#General Python imports
import os
import re
import sys
import glob
import numpy as np
import pprint as pp
from pathlib import Path
from tqdm.notebook import tqdm

#plotting and image processing
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

#rasterio for multi channael image processing
import rasterio as rio
from rasterio.transform import Affine
from rasterio.plot import show_hist,show
from fastai.torch_core import *

#metrics
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


def calc_similiarity(imgA,imgB):
  '''
  Calculates the MSE between two images (usually not a good metric, but it used as a reference)
  '''
  mse_score = mean_squared_error(imgA, imgB)
  ssim_score = ssim(imgA, imgB, data_range=imgA.max() - imgA.min())
  return mse_score,ssim_score


def normalize(image,vmin=0.0,vmax=0.3):
  '''
  RGB data was exported in the range between 0 and 0.3
  '''
  return (image - vmin)/(vmax-vmin)
  
  


def get_matching_img_path(fname,src_img_dir=None):
    '''
    Return the matching image for a given mask from an image directory
    '''
    suffix = fname.split('_')
    img_file = glob.glob(src_img_dir + '/*'+suffix[-1])
    return img_file[0]
    
    
def get_matching_mask_path(img_name,mask_dir=None):
    '''
    Return the matching mask of a given image from the masks directory
    '''
    suffix = img_name.split('_')
    mask_file = glob.glob(mask_dir + '/*'+'_'+suffix[-1])
    # print(img_name,'>>>\n',suffix[-1],'>>>\n',mask_file)
    return mask_file[0]


def display_image(image_path,bands=[3,2,1],title=None,ax=None,cm=None,verbose=True):
  '''
  Displaying RBG images by Red,Green,Blue in this order
  '''
  img =  rio.open(image_path)
  show(normalize(img.read(bands)), title=title,ax=ax,cmap=cm)
  if(verbose):
    print(title,len(img.indexes),'bands')
  return img
  
    
def print_mask_stats(rasterio_img,title=None):
  '''
  Prints statistics about a mask classes, class counts and proportions
  '''
  classes,counts = np.unique(rasterio_img.read(1),return_counts=True)
  print(title + ' classes:',classes)
  sumc = sum(counts)
  pp.pprint(['class:{},count:{},{:.2f}%:'.format(classes[i],c*100,c*100/sumc) for i,c in enumerate(counts)])
  print(len(rasterio_img.indexes),' bands')
  
  
def calc_pred_stats(arr,title='pred sample',mask_vals=None):
  '''
  Calculates statistics about a mask classes
  '''
  final_classes = mask_vals.copy()
  classes,counts = np.unique(arr,return_counts=True)
  for j,cls in enumerate(classes):
    if(j==cls):
      final_classes[j]=counts[j]
    else:
      final_classes[j]=0.0
  sumc = sum(counts)
  stats = ['Class:{:5d}, count:{}, {:.2f}%'.format(item[0],item[1],item[1]*100/sumc) for item in final_classes.items()]

  return stats
  

def get_stats_diff(stats,orig_stats):
  '''
  Calculates the difference in proportions of two stats lists calculated before
  '''
  abs_err, abs_fg_err, diffs = 0.,0.,[]
  for i,(stat,orig) in enumerate(zip(stats,orig_stats)):
    
    cls=stat.split(',')[0]
    s=stat.split(',')[2]
    st=re.sub('[^0-9.]+', '', s)
    
    os=orig.split(',')[2]
    oss=re.sub('[^0-9.]+', '', os)
    
    diff=float(st)-float(oss)
    diffs.append('{} =   {:.1f}%'.format(cls,diff))
    abs_err+=abs(diff)
    if (i>1):
      abs_fg_err+=abs(diff)
  
  diffs.append('Abs  Error =   {:.1f}'.format(abs_err))
  diffs.append('Abs FG Err =   {:.1f}'.format(abs_fg_err))
  return diffs, abs_err, abs_fg_err
  
def print_image_metadata(rasterio_img,title=None):
  print(title + ' Metadata:',rasterio_img.meta)
  print(title + ' Transform:\n',rasterio_img.transform)
  
  
def validate_dataset(s2_resized_dir,esm_aligned_dir,esm_orig_dir,df_orig,df_new):
  '''
  Validates the dataset after preprocessing, plots some samples and their statistics 
  '''
  cnt=0
  src_files = glob.glob(esm_aligned_dir + '/*.tif')
  for i,fnm in tqdm(enumerate(src_files)):
    # fnm = src_files[0]
    orig_fnm = Path(fnm).name
    orig_image = get_matching_img_path(orig_fnm,s2_resized_dir)
    
    with rio.open(os.path.join(esm_orig_dir , orig_fnm)) as orig_mask:
      with rio.open(fnm) as new_mask:
        with rio.open(orig_image) as orig_img:
          #plot a few trios
          if(i%175==0):
            #plot trio
            fig, (axorig,axmod,axmod2) = plt.subplots(1,3, figsize=(14,6))
            show(orig_mask,ax=axorig,title='{} ({},{})'.format(Path(orig_mask.name).name,orig_mask.width,orig_mask.height),cmap='Accent')
            show(new_mask,ax=axmod,title='New {} ({},{})'.format(Path(new_mask.name).name,new_mask.width,new_mask.height),cmap='Accent')
            show(normalize(orig_img.read([3,2,1])),ax=axmod2,title='{}({},{})'.format(Path(orig_img.name).name,orig_img.width,orig_img.height))
            #plot hist
            fig, (axh1, axh2) = plt.subplots(1,2, figsize=(12,4))
            axh1.set_xlim(-5, 275)
            show_hist(orig_mask, ax=axh1,bins=150,  title="Orig Mask Histogram (classes)")
            axh2.set_xlim(-5, 275)
            show_hist(new_mask, ax=axh2,bins=150,  title="New Mask Histogram (classes)")
            plt.show()
            print_mask_stats(orig_mask,orig_mask.name)
            print_mask_stats(new_mask,new_mask.name)
          #TESTS to see the generated image is as expected
          assert new_mask.driver==orig_mask.driver=='GTiff'
          assert new_mask.dtypes[0]==orig_mask.dtypes[0]=='uint8'
          assert new_mask.count==orig_mask.count==1

          assert new_mask.width==orig_img.width
          assert new_mask.height==orig_img.height
          assert new_mask.crs==orig_img.crs
          assert new_mask.transform==orig_img.transform
          if(orig_img.height!=orig_img.width):
            cnt+=1
          add_cls_counts(df_orig,orig_mask)
          add_cls_counts(df_new,new_mask)
  #Number of images with different width and height should be 0
  assert(cnt==0)
  return orig_mask,new_mask,orig_img


#calculate and display the distribution of classes
def add_cls_counts(df,mask,classes_dic={'0':0,'1':0,'250':0,'255':0}):
    '''
    Counting classes in masks to plot histograms and calculate dataset imbalance
    '''
    if isinstance(mask,TensorBase):
        classes,counts = np.unique(mask,return_counts=True)
    else:
        classes,counts = np.unique(mask.read(1),return_counts=True)
    new_row = list(counts)

    if len(classes)<4:
      new_row = classes_dic
      # print(len(df),classes,counts)
      for i,c in enumerate(classes):
        new_row[str(c)]=counts[i]
      
    df.loc[len(df)] = new_row


def get_bu_area_ratio(df,with_no_data=False,cols = ['0','1','2','3']):
  '''
  Calculates the Built-up area ratio (250+255)/(0+1+250+255), class 0 is optional
  '''
  c_255=df[cols[3]]
  c_250=df[cols[2]]
  c_one=df[cols[1]]
  c_zero=df[cols[0]]

  if(with_no_data):
    total=c_zero+c_one+c_250+c_255
  else:
    total=c_one+c_250+c_255
  return (c_250+c_255)/total
 

#Plotting
def plot_hist(df,title,axx,c,lbl):
  sns.distplot(df,bins=125,ax=axx,color=c,hist_kws=dict(alpha=0.3),label=lbl)
  axx.set_title(title)
  axx.set_ylabel('Number of images')
  axx.set_xlabel('Proportion of 250+255 / All classes')
  axx.legend()
  axx.set_xlim(0.05,0.9)
  
def annotate(ax):
  for p in ax.patches:
    ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x() * 1.010, p.get_height() * 1.010))

def plot_bar(df,title,axx,pct=True):
  if(pct==True):
    total_counts=df.sum(axis=0).sum()
  else:
    total_counts = 1
  (df.sum(axis=0)/total_counts).plot.bar(ax=axx,title=title,color=[ 'cyan','orange', 'green', 'red'],alpha=0.7)