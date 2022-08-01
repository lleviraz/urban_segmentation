# Evlaluation related utilities

###########
# Imports #
###########

import gc
# import os
# import math
# import random
# import shutil
# import statistics
import numpy as np
# from math import sqrt
# from scipy import stats
# from libs.common import *
# from libs.training import *
from libs.evaluation import *
# from sklearn.preprocessing import StandardScaler,MinMaxScaler
# from sklearn.metrics import classification_report,confusion_matrix
import torch
# import torchmetrics
# import torch.nn as nn
# from copy import deepcopy
# from torchmetrics import Dice
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

def predict(model, dl,device):
    '''
    Predict new masks using a model and a dataloader
    '''
    all_preds = []
    with torch.no_grad():
      for dev_input in tqdm(dl):
          # Send tensors to GPU
          dev_input   = dev_input[1].to(device)
          #predict
          dev_pred = model(dev_input)
          pred_cls =  np.argmax(dev_pred.detach().cpu() , axis = 1)
          all_preds.append(pred_cls.cpu())

    return all_preds
    

def plot_predictions(fnames,preds,image_dir,target_preds,cmap,code2class,class2desc,show_every=5,verbose=True):
  '''
  Saves and plots predictions
  '''
  cm=cmap
  nrows=max(1,int(len(preds)/show_every))
  if(verbose):
    f,axes = plt.subplots(nrows+1,2,figsize=(10,nrows*4))
    #row index
    j=0
  with rio.Env():
    for i in tqdm(range(len(preds))):
      pred_1 = preds[i]
      img_name = parse_file_name(fnames[i])
      
      pred=pred_1
      pred = pred.cpu()

      with rio.open(os.path.join(image_dir,img_name)) as matching_image:

        if (i%show_every==0):
          if(verbose):
            show(normalize(matching_image.read([3,2,1])),ax=axes[j,0],title=img_name)
          
            show(pred,ax=axes[j,1],cmap=cm)
            axes[j,1].set_title('Predicted mask')
            
            
            cls_values = [k for k in code2class.keys()] #[0,1,250,255]
            cls_colors = ['#00ff00','#fe0001','#0000ff']
            patches = [mpatches.Patch(color=cls_colors[i], 
                      label="{}-{}".format(cls_values[i],class2desc[code2class[i]])) for i in range(len(cls_values))]
            axes[j,1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=-1.)

            plt.tight_layout()
            j+=1
        
        pred_fname = 'pred_'+img_name
        #save images to target dir + copy the original MASK metadata including CRS
        #Opening one mask for as an example mask profile for projection

        #setting a profile for the new mask
        #starting by copying the matching image 'transform' and 'crs'
        new_mask_profile = matching_image.profile
        new_mask_profile['dtype'] = 'uint8'
        new_mask_profile['count'] = 1
        new_mask_profile['interleave'] = 'band'
        with rio.open(os.path.join(target_preds,pred_fname), 'w', **new_mask_profile) as dst2:
            dst2.write(pred.numpy().astype(rio.uint8), 1)
  print('pred masks saved to:',target_preds)
  if(verbose):
    plt.show()
  
class S2OnlyDataset(torch.utils.data.Dataset):
    '''
    A pytorch dataset for serving multi channel S1 images
    images - list of images to load (converted to tensors)
    '''
    def __init__(self,images):
        super(S2OnlyDataset, self).__init__()
        self.images = [(img,scale_multi_bands_array(img,method='minmax',fillnan=True)) for img in tqdm(images)]

    def __len__(self):
        return len(self.images)

    def get_one(self,idx):
      return self.images[idx]
    
    def __getitem__(self, idx):
      return self.images[idx]
