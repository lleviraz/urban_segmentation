# Evlaluation related utilities

###########
# Imports #
###########

import gc
import os
import math
import random
import shutil
import statistics
import numpy as np
from math import sqrt
from scipy import stats
from libs.common import *
from libs.training import *
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix
import torch
import torchmetrics
import torch.nn as nn
from copy import deepcopy
from torchmetrics import Dice
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def evaluate(model, dl , caption,plot_cm , print_scores,device,class2desc,code2class,max_batches=10):
    '''
    Evaluate the models using a classification report and confusion matrix
    This works well in the case of unbalanced labels in the datasets
    '''
    dice = torchmetrics.Dice(average='weighted',num_classes=model.n_classes)#,ignore_index=0)
    dice_arr = []

    if print_scores == True:
        print(caption,"\n")
          
        # Dev  evaluation 
        y_true       = []
        y_pred       = []
        all_preds = []
        
        with torch.no_grad():
            jj=0
            for dev_input, dev_label in tqdm(dl):
                # Send tensors to GPU
                dev_input   = dev_input[1].to(device)
                dev_label  = dev_label[1].to(device) 
                #predict
                dev_pred = model(dev_input)

                pred_cls =  np.argmax(dev_pred.detach().cpu() , axis = 1)
                #flatten tensors for the classification report
                y_true.extend(dev_label.cpu().flatten().numpy()) 
                y_pred.extend(pred_cls.cpu().flatten().numpy()) 
                all_preds.append(pred_cls.cpu())
                batch_dice = dice(dev_label.cpu(), pred_cls.cpu())
                btc = pred_cls.cpu()

                dice_arr.append(batch_dice)
                if(jj > max_batches):
                  break
                jj+=1
                          
    mean_dice = np.mean(np.array(dice_arr))
    tick_names = list(class2desc.values())
    lbls= list(code2class.keys())

    ## Confusion Matrix - Multi class ## 
    if plot_cm == True:
        cnf_matrix = confusion_matrix(y_true, y_pred)
        fig, ax    = plt.subplots(figsize=(8,4))
        tick_marks = np.arange(len(tick_names))
        sns.heatmap(pd.DataFrame(data=cnf_matrix, index=tick_names, columns=tick_names), annot=True, cmap="YlGnBu",fmt=".1f")
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion Matrix (\'All Classes\')', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        
      

    # Scores #
    m1 = classification_report(y_true, y_pred, target_names  = tick_names, output_dict=True,digits=3 )
    f1_all      = m1['weighted avg']['f1-score']
    
    if print_scores == True:
        print(classification_report(y_true, y_pred, target_names  = tick_names,digits=3 ))
        print('**************{} Mean Dice score:{:.3}**************'.format(caption,mean_dice))
    
    return f1_all , all_preds,mean_dice

    
def parse_file_name(full_path):
  return full_path.split(os.path.sep)[-1]

def get_file_name_id(img_path):
  spl = img_path.split(os.path.sep)
  fname = spl[0]
  fid   = spl[0].split('.')[0].split('_')[-1]
  return fname,fid
 
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
  cnt[c250_IN_BU]=float(c250/(c250+c255))
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
    img_fn = fn#.name
    mask = get_matching_mask_path(img_fn,mask_dir)
    
    msk_fn = mask.split('/')[-1]
    with rio.open(mask) as fmsk:
      #orig
      cnt1 = calc_props(fmsk.read(1),code2class=code2class)
      cnt1['type']='ORIG'

      #pred
      cnt2 = calc_props(pred_msk,decode=True,code2class=code2class)
      cnt2['type']='PRED'

      #diff
      cnt3 = {}
      for i in numeric_cols:
        cnt3[i]=float(abs(cnt1[i]-cnt2[i]))
      cnt3['type']='DIFF'
    
      for c in [cnt1,cnt2,cnt3]:
        c['image_name']=parse_file_name(img_fn)
        c['mask_name']=msk_fn

      reg_df=reg_df.append(cnt1,ignore_index=True)
      reg_df=reg_df.append(cnt2,ignore_index=True)
      reg_df=reg_df.append(cnt3,ignore_index=True)
      
      
  return reg_df
  
###########
# Metrics #
###########

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


def eval_metrics(true, pred, num_classes,class2code):
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
    avg_per_class_acc = per_class_pixel_accuracy(hist)
    nc=num_classes+1 if num_classes==len(class2code)-1 else num_classes
    dice = Dice(average='weighted',num_classes=nc)
    avg_dice = dice(true.cpu(), pred.cpu())
    return 0, avg_per_class_acc, 0, avg_dice



####################
# Plotting Results #
####################
BU_PROP='BU_prop'
c250_IN_BU = 'p250_in_bu'
c255_IN_BU ='p255_in_bu'

def plot_final_results(test_fnames,preds,pred_prop_df,image_dir,mask_dir,
                        cmap,code2class,class2code,class2desc,preds_dir,
                        show_every=5,scores_df=None,show_only=None):
  '''
  Plots the final result images, masks, predicted masks, and statistics
  Param: test_fnames - images file names list 
  Param: pred_prop_df - proprtions dataframe calculated using `build_proportions_df()`
  Param: image_dir - the source images dircetory
  Param: mask_dir - the original mask directory
  Param: show_every - show a plot of an image every X iterations (default 3)
  '''
  cm=cmap
  if(show_only!=None):
    nrows=max(1,int(len(show_only)/show_every))
  else:
    nrows=max(1,int(len(preds)/show_every))
  f,axes = plt.subplots(nrows+1,5,figsize=(24,nrows*4))
  #row index
  j=0
  with rio.Env():
    #Opening one mask for as an example mask profile for projection
    for i in tqdm(range(len(preds))):
      pred_1 = preds[i]
      img_name = parse_file_name(test_fnames[i])
      
      #filter if limited list of images was sent
      if(show_only!=None):
        if(img_name not in show_only):
          continue

      mask_path=get_matching_mask_path(img_name, mask_dir)
      msk = get_mask_as_array(img_name,mask_dir,code2class)
      msk_name = parse_file_name(mask_path)

      pred=pred_1

      true=msk[1]
      true = true.cpu()
      pred = pred.cpu()

      n_clss=len(np.unique(true))
      _, avg_per_class_acc, _, avg_dice = eval_metrics(true,pred,n_clss,class2code)
      if (i%show_every==0):

        with rio.open(os.path.join(image_dir,img_name)) as matching_image:
          show(normalize(matching_image.read([3,2,1])),ax=axes[j,0],title=img_name)
        
        show(true,ax=axes[j,1],cmap=cm)
        axes[j,1].set_title('True mask ' + msk_name)

        show(pred,ax=axes[j,2],cmap=cm)
        axes[j,2].set_title('Predicted mask')
        
        
        cls_values = [k for k in code2class.keys()]
        # Set a legend on the predicted image
        if(n_clss==4):
          cls_colors = ['#7fc97f','#fdc086','#e51b60','#666666']
        else:
          cls_colors = ['#00ff00','#fe0001','#0000ff']
        patches = [mpatches.Patch(color=cls_colors[i], 
                   label="{}-{}".format(cls_values[i],class2desc[code2class[i]])) for i in range(len(cls_values))]
        axes[j,1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=-1.)

        #scores panel
        width = 0.9
        scores = ['FG Acc','Dice']
        x = np.arange(len(scores))
        values = [avg_per_class_acc,avg_dice]
        axes[j,3].bar(x, values , width ,color=[ 'orange', 'cyan'])

        axes[j,3].set_xticks(x)
        axes[j,3].set_yticks(np.arange(0.0,1.1,0.1))
        axes[j,3].set_xticklabels(scores,rotation=0)
        def annotate_bars(some_ax,font_size=18):
          for j,p in enumerate(some_ax.patches):
            some_ax.annotate('{:.3f}'.format(p.get_height()), (p.get_x(), p.get_height() * .90),size=font_size)
        annotate_bars(axes[j,3])

        #Text Panel
        axes[j,4].axis('off')
        def get_vals(row_type):
          return pred_prop_df.loc[(pred_prop_df['mask_name']==msk_name) & (pred_prop_df['type']==row_type)][[BU_PROP,c250_IN_BU,c255_IN_BU]].values

        true_vals = get_vals('ORIG')[0]
        pred_vals = get_vals('PRED')[0]
        with pd.option_context('display.float_format', '{:,.3f}'.format):
          axes[j,4].text(0., .2,'BU Area:\nTrue:{:,.3f} Pred:{:,.3f}\n250-Non Residential:\nTrue:{:,.3f} Pred:{:,.3f}\n255-Residential:\nTrue:{:,.3f} Pred:{:,.3f}'.format(
                                  true_vals[0],pred_vals[0],
                                  true_vals[1],pred_vals[1],
                                  true_vals[2],pred_vals[2]),fontsize=24)
        plt.tight_layout()
        j+=1
      
      pred_fname = 'pred_'+img_name
      scores_df.loc[len(scores_df)] = [img_name,msk_name,pred_fname,
                                   avg_per_class_acc.item(), avg_dice.item()]

      #save images to target dir + copy the original MASK metadata including CRS
      with rio.open(mask_path, 'r') as src1:
        with rio.open(os.path.join(preds_dir,pred_fname), 'w', **src1.profile) as dst2:
            dst2.write(pred.numpy().astype(rio.uint8), 1)

  plt.show()
  return scores_df
  

def plot_scatter_diff(orig_df,pred_df,title,filter_index,thresh,vl=False):
  f,ax1 = plt.subplots(1,1,figsize=(18,6))
  x1=orig_df[filter_index].index
  y1=orig_df[filter_index].values
  ax1.scatter(x1,y1,alpha=0.4,marker='>',color='red')

  x2=pred_df[filter_index].index
  y2=pred_df[filter_index].values
  ax1.scatter(x2,y2,alpha=0.4,marker='<',color='navy')
  ax1.set_xlabel('Image id',fontsize=18)
  ax1.set_ylabel('Ratio of BU area',fontsize=18)
  ax1.set_yticks(np.arange(0, 1, 0.05))
  if(vl):
    ax1.vlines(x1[0], y1[0], y2[0], linestyles='dotted')
  ax1.set_title(title.format(int(thresh*100),len(x2),len(orig_df)),fontsize=18)
  plt.legend({'Original Masks','Predicted Masks'},fontsize=14)
  plt.show()


def show_diag_plot(om,nm,title='Dev'):
  '''
  Shows a diagonal plot of the target (diagonal) vs predicted scattered mean values 
  '''
  _,ax0 = plt.subplots(1,1,figsize=(14,6))
  # ax0.plot([0, 1], [0, 1], ls="--")
  ident = [min(min(om),min(nm)), max(max(om),max(nm))]
  # plt.plot(ident,ident,ls='--',color='r')
  ax0.scatter(om,nm)
  ax0.set_xlabel('Original')
  ax0.set_ylabel('Predicted')
  spr_corr = stats.spearmanr(om,nm)
  m, b = np.polyfit(om, nm, 1)
  #best fit line
  ax0.plot(om, nm, 'o')
  ax0.plot(om, m*om + b)
  plt.title(title + ' set Predicted vs. Original BU Area Ratio (Spearman:{:.2f})'.format(spr_corr[0]))
  # plt.legend()
  plt.show()
  
def plot_confidence_interval(x, values, z=1.96, color='#2187bb', horizontal_line_width=0.25):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / sqrt(len(values))
  
    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, 'o', color='#f44336')
    plt.ylabel('% error')

    return mean, confidence_interval
    
#1.96=95% #1.64=90% #2.33=98% #2.58=99%
def calc_conf_intervals(error,z=1.96,title='na'):
  n=len(error)
  mean_error=np.mean(error)
  # print(error)
  interval = z * math.sqrt( (mean_error * (1 - mean_error)) / n)
  lower = mean_error-interval
  upper = mean_error+interval
  print(title)
  print('mean=%.3f, lower=%.3f, upper=%.3f' % (mean_error,lower, upper))