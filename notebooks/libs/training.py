# Training related utilities

###########
# Imports #
###########

# # Python General
# import numpy as np
# import matplotlib.pyplot as plt

import gc
import os
import random
import shutil
import numpy as np
from libs.common import *
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import torch
import torchmetrics
import torch.nn as nn
from copy import deepcopy
from torchmetrics import Dice
import torch.nn.functional as F
import matplotlib.pyplot as plt


def load_tif_files(dir,title):
  '''
  Load tif files from a path, return a list of matching file paths
  '''    
  files = glob.glob(os.path.join(dir,'*.tif'))
  print('{} - {} image files in {}'.format(title,len(files),dir))
  return files
  
def create_dev_set(src_imgs_dir,src_masks_dir,tgt_imgs_dir,tgt_masks_dir,valid_num=50):
  '''
  Performs a Random split of specified number of images from the training set as a validation or test set
  '''
  if(os.path.isdir(tgt_imgs_dir)):
    print('{} Directory exists,\n exiting.(To recreate, move files back to original dirs BEFORE DELETING(!!!) and rerun)'.format(tgt_imgs_dir))
    target = os.listdir(tgt_imgs_dir)
    number_files = len(target)
    print(number_files)
  else:
    print('Creating ',tgt_imgs_dir)
    print('Creating ',tgt_masks_dir)
    os.makedirs(tgt_imgs_dir)
    os.makedirs(tgt_masks_dir)
    s2_source = src_imgs_dir
    esm_source = src_masks_dir
    files = os.listdir(s2_source)
    no_of_files = valid_num #int(len(files) *valid_pct)

    for file_name in random.sample(files, no_of_files):
        shutil.move(os.path.join(s2_source, file_name), tgt_imgs_dir)
        shutil.move(get_matching_mask_path(file_name,esm_source), tgt_masks_dir)


def scale_multi_bands_array(image_path,new_min=0.,new_max=1.,method='minmax',fillnan=False): # Default std and min-max
  '''
  A Min Max or Standard Multi Channel Scaler for images
  Scales each channel separately to a given range
  Returns: a tensor with the scaled multi channel image 
  '''
  with rio.open(image_path) as img:
    n_bands,height,width = img.count,img.shape[0],img.shape[1]
    norm_img = np.zeros((n_bands,height,width),dtype=np.float)
    if(method=='minmax'):
      scaler = MinMaxScaler()
    else:
      scaler = StandardScaler()

    for b in range(1,img.count+1):
      img_bnd = img.read(b)
      #Fill nans with mean of the channel
      if((fillnan) & (np.isnan(img_bnd).sum()>0)):
          not_nan =(np.isnan(img_bnd)==False)
          #try calculating mean from non nan values
          mean_chn = img_bnd[not_nan].mean()
          if(mean_chn==np.nan):
            #get global mean of the channel calculated in preprocessing
            mean_chn = get_global_chn_mean(b)
          img_bnd = np.nan_to_num(img_bnd,nan=mean_chn)
      if(np.isnan(img_bnd).sum()>0):
        print(image_path,'True',b,img_bnd.shape,img_bnd)

      if(method=='minmax'):
        v_min, v_max = np.min(img_bnd), np.max(img_bnd)
        norm_img[b-1,:,:] = (img_bnd - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
      else:
        #Standartization 
        scaler.fit(img_bnd)
        norm_img[b-1,:,:] = scaler.transform(img_bnd)
    assert(norm_img.shape==(n_bands,height,width))
    return torch.tensor(norm_img,dtype=torch.float)


def get_mask_as_array(img_name,mask_path,p2c,merge_bg=False):
  '''
  Returns a matching mask for an image as tensor Tuple: (file name,tensor)
  '''
  fn = get_matching_mask_path(img_name,mask_path)
  with rio.open(fn) as msk:
    mask_arr = msk.read(1)
    new_msk = np.zeros((mask_arr.shape[0],mask_arr.shape[0]),dtype=np.int)

    for i, val in enumerate(p2c):
      new_msk[mask_arr==p2c[i]] = val
    if(merge_bg):
      new_msk[msk==1] = 0 #merge 1 to 0
    return fn,torch.tensor(new_msk,dtype=int)


class S2ESMDataset(torch.utils.data.Dataset):
    '''
    A pytorch dataset for serving multi channel S1 images and ESM matching masks
    images - list of images to load (converted to tensors)
    mask_path - matching masks directory
    '''
    def __init__(self,images,mask_path,code2class,aug=None):
        super(S2ESMDataset, self).__init__()
        self.images = [(img,scale_multi_bands_array(img,method='minmax')) for img in tqdm(images)]
        # self.images = [(img,scale_multi_bands_array(img,method='std')) for img in tqdm(images)]
        self.masks = [get_mask_as_array(image[0],mask_path,code2class) for image in tqdm(self.images)]
        self.aug = aug

    def __len__(self):
        return len(self.images)

    def get_one(self,idx):
      return self.images[idx], self.masks[idx]
    
    def get_one_with_aug(self,idx):
      return self.__getitem__(idx)
    
    def __getitem__(self, idx):
      image = self.images[idx]
      img_nm = image[0]
      mask = self.masks[idx]
      msk_nm = mask[0]
      if (self.aug is not None):
        transformed = self.aug.encodes(image[1],mask[1])
        image = (img_nm,torch.tensor(transformed[0],dtype=torch.float))
        mask = (msk_nm,torch.tensor(transformed[1],dtype=torch.long))
      return image, mask

CMAP='brg_r'
def show_image(image_path,bands=[3,2,1],cm=CMAP,title=None,ax=None):
  '''
  Displaying images : either RBG images by Red,Green,Blue in this order or masks with a single band
  '''
  img = None
  with rio.open(image_path) as img1:
    img = img1.read(bands)
    if(bands==[1]):
      show(img, title=title,ax=ax,cmap=cm)
    elif(len(bands)>3):
      show(img, title=title,ax=ax)
    else:
      show(normalize(img), title=title,ax=ax)
  return img


def permute_4_display(img_arr):
  '''
  The channels need to be last in the shape for displaing with plt
  '''
  return torch.permute(normalize(torch.tensor(img_arr[1:4,:,:],dtype=float)), (1, 2, 0)).numpy()

def show_one(ds,index,images,preds=None,cmap=CMAP,with_aug=False):
  '''
  Shows one image from the dataset
  '''    
  if(with_aug):
    img1_arr,msk1_arr = ds.get_one_with_aug(index)
  else:
    img1_arr,msk1_arr = ds.get_one(index)
  f,axs = plt.subplots(1,4 if preds!=None else 3,figsize=(16,5))
  one_norm_img = permute_4_display(img1_arr[1])
  print(img1_arr[1].shape,msk1_arr[1].shape)
  one_mask =  msk1_arr[1]
  show_image(images[index],ax=axs[0])
  axs[0].set_title('Orig RGB')
  axs[1].imshow(one_norm_img)
  axs[1].set_title('Scaled')
  axs[2].imshow(one_mask,cmap=cmap)
  axs[2].set_title('True')
  if(preds!=None):
    axs[3].imshow(preds[0][index],cmap=cmap)
    axs[3].set_title('Pred')
  plt.suptitle('Augmented:{}'.format(with_aug))
  plt.show()



#################
#    UNET       #
#################
#Based on UNET from: https://github.com/pytorch/benchmark.git

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            ####
            #tried also a deeper version
            # nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),            
            # nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            ####
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        #TODO tryign to add dropout to fight overfitting
        x = nn.functional.dropout2d(x,0.3) #0.5
        return self.conv(x)
        
        
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,base_size=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        base = base_size
        self.inc = DoubleConv(n_channels, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base*8, base*16 // factor)
        self.up1 = Up(base*16, base*8 // factor, bilinear)
        self.up2 = Up(base*8, base*4 // factor, bilinear)
        self.up3 = Up(base*4, base*2 // factor, bilinear)
        self.up4 = Up(base*2, base, bilinear)
        self.outc = OutConv(base, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


#################
# Train loop    #
#################

def clear_cache():
  gc.collect()
  torch.cuda.empty_cache()

def end_epoch(e,train_loss,dev_loss,train_dice,dev_dice,verbose):
  if (verbose) & (e % 5 == 1):   #  report every 5 epochs
    print("Epoch {epoch_id}".format(epoch_id=e),end='\t- ')
    print("Train loss: {:.3f}".format(train_loss),end=' - ')
    print("Dev loss: {:.3f}".format(dev_loss),end=' - ')
    print("Train dice: {:.3f}".format(train_dice),end=' - ')
    print("Dev dice: {:.3f}".format(dev_dice))  
    clear_cache()  

def train_loop(model, n_epochs, train_dl, dev_dl,optimizer, loss_fn,device,
               early_stopping=15,verbose=False,n_classes=3,is_deeplab=False):
  '''
  Trains a pytorch model for number of epochs using traing and optional dev dataloaders
  Using a given optimizer and loss function
  Returns: arrays of losses for train and dev data, and arrays of 'dice' score for train and dev 
  (loss_train_arr , loss_dev_arr , all_epochs_dice_arr , all_epochs_dev_dice_arr)
  '''
  #Init once per training
  loss_train_arr,loss_dev_arr,dice_train_arr,dice_dev_arr,all_epochs_dice_arr,all_epochs_dev_dice_arr= [],[],[],[],[],[]
    
  epochs_without_improvement,best_dev_loss,best_dev_dice= 0,None,None

  model = model.to(device)
  criterion = loss_fn.to(device)
  dice = torchmetrics.Dice(average='weighted',num_classes=n_classes)

  # Train loop
  print('Training (with{} Dev set)'.format('out' if(dev_dl==None) else ''))
  for e in tqdm(range(1, n_epochs + 1)):
      train_loss   = 0
      train_dice     = 0
      
      for train_input, train_label in train_dl:
          optimizer.zero_grad()
          
          # Send tensors to GPU
          image_tensor   = train_input[1].to(device)
          mask_tensor  = train_label[1].to(device) 

          if(is_deeplab):
            pred = model(image_tensor)['out']
          else:
            pred = model(image_tensor)
            
          loss   = criterion(pred, mask_tensor)
          loss.backward()
          optimizer.step()
          # agg  batch results
          train_loss += loss.item()
          #find the max of every pixel prediction
          pred_cls =  np.argmax(pred.detach().cpu() , axis = 1)
          batch_dice = dice(mask_tensor.cpu(), pred_cls.cpu())
          dice_train_arr.append(batch_dice)
      
      # Agg epoch results 
      train_loss = train_loss / len(train_dl)            
      loss_train_arr.append(train_loss)
      train_dice = np.mean(np.array(dice_train_arr))
      all_epochs_dice_arr.append(train_dice)

      # Dev  evaluation 
      dev_loss ,dev_dice = 0,0
      
      #no validation set use only train
      if(dev_dl==None):
        # end_epoch(e,train_loss,dev_loss,train_dice,dev_dice,verbose)
        if not best_dev_loss or train_loss < best_dev_loss:  
        # if not best_dev_dice or train_dice > best_dev_dice:   
            best_train_loss = train_loss
            best_dev_loss = train_loss
            best_train_dice  = train_dice
            best_dev_dice    = train_dice
            epochs_without_improvement = 0
            best_state_dict = deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
        
        #end epoch reporting    
        end_epoch(e,train_loss,dev_loss,train_dice,dev_dice,verbose)
        
        if epochs_without_improvement == early_stopping:
            print ("\nEarly stoping after {} epochs. Dev loss did not imporve for more than {} epcochs".format(e,early_stopping))
            break
        #no dev set training
        continue
        
      with torch.no_grad():
          for dev_input, dev_label in dev_dl:
              # Send tensors to GPU
              dev_input   = dev_input[1].to(device)
              dev_label  = dev_label[1].to(device)

              # predict 
              if(is_deeplab):
                dev_pred = model(dev_input)['out']
              else:
                dev_pred = model(dev_input)
              #loss
              loss   = criterion(dev_pred, dev_label)
              
              dev_loss   += loss.item()
              # Agg results
              #find the max of every pixel prediction
              dev_pred_cls =  np.argmax(dev_pred.detach().cpu() , axis = 1)
              batch_dev_dice = dice(dev_label.cpu(), dev_pred_cls.cpu())
              dice_dev_arr.append(batch_dev_dice)

          # Agg epoch results 
          dev_loss = dev_loss / len(dev_dl)         
          loss_dev_arr.append(dev_loss)
          dev_dice = np.mean(np.array(dice_dev_arr))
          all_epochs_dev_dice_arr.append(dev_dice)
         
      # Early Stop and best model save 
      #if not best_dev_loss or dev_loss < best_dev_loss:  
      if not best_dev_dice or dev_dice > best_dev_dice:   
          best_train_loss = train_loss
          best_dev_loss   = dev_loss
          best_train_dice  = train_dice
          best_dev_dice    = dev_dice
          epochs_without_improvement = 0
          #print ("Achieved lower test loss  , save model  at epoch number {} ".format(e + 1) )
          best_state_dict = deepcopy(model.state_dict())
      else:
          epochs_without_improvement += 1

      if epochs_without_improvement == early_stopping:
          print ("\nEarly stoping after {} epochs. Dev loss did not imporve for more than {} epcochs".format(e,early_stopping))
          break

      #end epoch reporting    
      end_epoch(e,train_loss,dev_loss,train_dice,dev_dice,verbose)

                
  # Finish  training 
  model.load_state_dict(best_state_dict)
  if (verbose):
      print('\nFinished Training:')
      print('Train loss={:.3f}, Dev loss={:.3f}'.format(best_train_loss ,best_dev_loss ))
      print('Train dice={:.3f}, Dev dice={:.3f}'.format(best_train_dice ,best_dev_dice ))

  return loss_train_arr , loss_dev_arr , all_epochs_dice_arr , all_epochs_dev_dice_arr


def loss_graph(loss_train_arr , loss_dev_arr , metric_train_arr , metric_dev_arr , test_name = 'Dev',metric='Dice'):
    '''
    Plots the loss and dice score graphs after training
    '''
    fig, ax = plt.subplots(1, 2,figsize=(14, 5))

    ax[0].set_title('Loss vs Epoch')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].plot(loss_train_arr,label="Train loss")
    ax[0].plot(loss_dev_arr,  label="{} loss".format(test_name))
    ax[0].legend()
        
    ax[1].set_title('{} vs Epoch'.format(metric))
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('{}'.format(metric))
    ax[1].plot(metric_train_arr,  label="Train {}".format(metric))
    ax[1].plot(metric_dev_arr,  label="{} {}".format(test_name,metric))
    ax[1].legend()  
