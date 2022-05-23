"""Common image segmentation metrics.
credits: some code was forked from: https://github.com/kevinzakka/pytorch-goodies
"""

import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from libs.preprocess import *
from libs.metrics import *
from libs.multichannel import *
# from libs.metrics import _fast_hist

def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])

EPS = 1e-10

def prep_hist(inp,targ):
  '''
  utility helper : prepare tensors confusion matrix for multiclass evaluation
  '''
  targ = targ.squeeze(1).cpu()
  n_clss=len(np.unique(targ.cpu()))
  pred=inp.argmax(dim=1).cpu()
  hist = torch.zeros((n_clss, n_clss)).cpu()
  for t, p in zip(targ, pred):
      hist += _fast_hist(t.flatten(), p.flatten(), n_clss)
  return hist
  

def _fast_hist(true, pred, num_classes):
    '''
    Utility helper : prepare tensors confusion matrix for multiclass evaluation
    '''
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist


def overall_pixel_accuracy(hist):
    """Computes the total pixel accuracy.

    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.

    Args:
        hist: confusion matrix.

    Returns:
        overall_acc: the overall pixel accuracy.
    """
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc


def per_class_pixel_accuracy(hist):
    """Computes the average per-class pixel accuracy.

    The per-class pixel accuracy is a more fine-grained
    version of the overall pixel accuracy. A model could
    score a relatively high overall pixel accuracy by
    correctly predicting the dominant labels or areas
    in the image whilst incorrectly predicting the
    possibly more important/rare labels. Such a model
    will score a low per-class pixel accuracy.

    Args:
        hist: confusion matrix.

    Returns:
        avg_per_class_acc: the average per-class pixel accuracy.
    """
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc


def jaccard_index(hist):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).

    Args:
        hist: confusion matrix.

    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard)
    return avg_jacc


def dice_coefficient(hist):
    """Computes the SÃ¸rensenâ€“Dice coefficient, a.k.a the F1 score.

    Args:
        hist: confusion matrix.

    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)
    return avg_dice


def eval_metrics(true, pred, num_classes):
    """Computes various segmentation metrics on 2D feature maps.

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        pred: a tensor of shape [B, H, W] or [B, 1, H, W].
        num_classes: the number of classes to segment. This number
            should be less than the ID of the ignored class.

    Returns:
        overall_acc: the overall pixel accuracy.
        avg_per_class_acc: the average per-class pixel accuracy.
        avg_jacc: the jaccard index.
        avg_dice: the dice coefficient.
    """
    hist = torch.zeros((num_classes, num_classes))
    for t, p in zip(true, pred):
        hist += _fast_hist(t.flatten(), p.flatten(), num_classes)
    overall_acc = overall_pixel_accuracy(hist)
    avg_per_class_acc = per_class_pixel_accuracy(hist)
    avg_jacc = jaccard_index(hist)
    avg_dice = dice_coefficient(hist)
    return overall_acc, avg_per_class_acc, avg_jacc, avg_dice


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

#CONSTANTS
BU_PROP='BU_prop'
c250_IN_BU = 'p250_in_bu'
c255_IN_BU ='p255_in_bu'

def calc_props(tensor_img,decode=False,code2class=None):
  '''
  Calculates the propertions of the forground classes given a mask tensor
  '''
  decoded_classes,counts = np.unique(tensor_img,return_counts=True)
  if(decode):
    decoded_classes = [code2class[c] for c in decoded_classes]

  cnt = {cl:cn for (cl,cn) in zip(decoded_classes,counts)}
  if(len(cnt)<4):
    if(0 not in cnt):
      cnt[0]=0.
    if(1 not in cnt):
      cnt[1]=0.
    if(250 not in cnt):
      cnt[250]=0.
    if(255 not in cnt):
      cnt[255]=0.

  total_cnt=sum(counts)
  c250,c255=0.,0.
  if(250 in cnt):
    c250=cnt[250]
  if(255 in cnt):
    c255=cnt[255]
  
  bu_prop=float((c250+c255)/total_cnt)
  cnt[BU_PROP]=bu_prop
  # cnt[c250_IN_BU]=float(a/total)
  cnt[c250_IN_BU]=float(c250/(c250+c255))
  # cnt[c255_IN_BU]=float(b/total)
  cnt[c255_IN_BU]=float(c255/(c250+c255))
  
  return cnt


def build_proportions_df(fnames,preds,mask_dir,code2class):
  '''
  Builds a proprtions dataframe for a list of image files and their predicted masks
  by counting the per class pixels, and calculating proprtions of fourgournd BU classes, 250 and 255,
  vs the other classes.
  '''
  numeric_cols = [0,1,250,255,BU_PROP,c250_IN_BU,c255_IN_BU]
  str_cols=['type','image_name','mask_name']

  reg_df=pd.DataFrame(columns=str_cols+numeric_cols)
  for fn,pred_msk in tqdm(zip(fnames,preds)):
    img_fn = fn.name
    mask = get_matching_mask_path(img_fn,mask_dir)
    msk_fn = mask.split('/')[-1]
    with rio.open(mask) as fmsk:
      #orig
      cnt1 = calc_props(fmsk.read(1),code2class=code2class)
      cnt1['type']='ORIG'

      #pred
      pred_arx = pred_msk.argmax(dim=0)
      cnt2 = calc_props(pred_arx,decode=True,code2class=code2class)
      cnt2['type']='PRED'

      #diff
      cnt3 = {}
      for i in numeric_cols:
        cnt3[i]=float(abs(cnt1[i]-cnt2[i]))
      cnt3['type']='DIFF'
    
      for c in [cnt1,cnt2,cnt3]:
        c['image_name']=img_fn
        c['mask_name']=msk_fn

      reg_df=reg_df.append(cnt1,ignore_index=True)
      reg_df=reg_df.append(cnt2,ignore_index=True)
      reg_df=reg_df.append(cnt3,ignore_index=True)
      
      
  return reg_df

mask_vals = {0:0,1:1,2:250,3:255}
def get_matching_mask_path(img_name,mask_dir=None):
    '''
    Return the matching mask of a given image from the masks directory
    '''
    suffix = img_name.split('_')
    mask_file = glob.glob(mask_dir + '/*'+'_'+suffix[-1])
    return mask_file[0]

def plot_final_results(test_fnames,preds,pred_prop_df,image_dir,mask_dir,show_every=3):
  '''
  Plots the final result images, masks, predicted masks, and statistics
  Param: test_fnames - images file names list 
  Param: pred_prop_df - proprtions dataframe calculated using `build_proportions_df()`
  Param: image_dir - the source images dircetory
  Param: mask_dir - the original mask directory
  Param: show_every - show a plot of an image every X iterations (default 3)
  '''
  cm='Accent'
  nrows=int(len(test_fnames)/show_every)+1
  f,axes = plt.subplots(nrows,5,figsize=(32,nrows*6))
  j=0
  with rio.Env():
    with rio.open(get_matching_mask_path(test_fnames[0].name,mask_dir),'r') as src:
      profile = src.profile

      for i in tqdm(range(len(test_fnames))):
        pred_1 = preds[0][i]
        pred_arx = pred_1.argmax(dim=0)

        msk = get_msk(test_fnames[i].name,mask_dir,mask_vals)

        if pred_arx.shape[0]!=1:
          pred=pred_arx.unsqueeze_(0)
        true=transforms.ToTensor()(msk)
        if true.shape[0]!=1:
          true.unsqueeze_(0)
        true = true.cpu()
        pred = pred.cpu()
        pred_arx = pred_arx.cpu()

        if (i%show_every==0):
          n_clss=len(np.unique(true))
          overall_acc, avg_per_class_acc, avg_jacc, avg_dice = eval_metrics(true,pred,n_clss)
          a_s = pred_1.argmax(dim=0).unsqueeze(0).to(torch.float)
          b_s = transforms.ToTensor()(msk).unsqueeze_(0).to(torch.float)
          
          stats = calc_pred_stats(pred_arx,str(i)+' pred',mask_vals)
          orig_stats = calc_pred_stats(true,str(i)+' true',mask_vals)

          # stats_diff,abs_err,abs_fg_err = get_stats_diff(stats,orig_stats)

          with rio.open(os.path.join(image_dir,test_fnames[i].name)) as matching_image:
            show(normalize(matching_image.read([3,2,1])),ax=axes[j,0],title=test_fnames[i].name)
          
          show(msk,ax=axes[j,1],cmap=cm)
          msk_name = get_matching_mask_path(test_fnames[i].name, mask_dir).split(os.path.sep)[-1]
          axes[j,1].set_title('True mask ' + msk_name)

          show(pred_arx.squeeze(0),ax=axes[j,2],cmap=cm)
          axes[j,2].set_title('Predicted mask')


          #scores panel
          width = 0.9
          scores = ['per_cls_acc','IoU','Dice']
          x = np.arange(len(scores))
          values = [avg_per_class_acc,avg_jacc,avg_dice]
          axes[j,3].bar(x, values , width ,color=[ 'orange', 'green', 'red'])

          axes[j,3].set_xticks(x)
          axes[j,3].set_yticks(np.arange(0.0,1.1,0.1))
          axes[j,3].set_xticklabels(scores,rotation=0)

          #Text Panel
          axes[j,4].axis('off')
          with pd.option_context('display.float_format', '{:,.3f}'.format):
            axes[j,4].text(0.0, 0.25,
                           '=======================\
                           \n-------------------Classes---------------------- \
                           \n0:Green,1:Orange,250:Pink,255:Gray \
                           \n-------------------Scores----------------------- \
                           \nFG Acc={:.3f} IoU={:.3f} Dice={:.3f} \
                           \n----------------------------------------------------- \
                           \n {} \
                           \n======================='.format(
                           avg_per_class_acc, avg_jacc, 
                           avg_dice,pred_prop_df[pred_prop_df['mask_name']==msk_name][['type','BU_prop','250_prop','255_prop']]),
                           fontsize=18)
  
          f.tight_layout()

          j+=1

          #save images to target dir
          with rio.open(os.path.join(PREDS_DIR,test_fnames[i].name), 'w', **profile) as dst1:
              dst1.write(true[0].numpy().astype(rio.uint8), 1)
          with rio.open(os.path.join(PREDS_DIR,'pred_'+test_fnames[i].name), 'w', **profile) as dst2:
              dst2.write(pred_arx[0].numpy().astype(rio.uint8), 1)

  plt.show()
  
