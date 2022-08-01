###########
# Imports #
###########
import os
import glob
import scipy
import numpy as np
import pprint as pp
import pandas as pd
import seaborn as sns
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import rasterio as rio
from rasterio.plot import show,show_hist
from rasterio.warp import reproject, Resampling
#metrics
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

#our utils
from libs.bands import *

####################
# Image Processing #
####################

def normalize(image,vmin=0.0,vmax=0.3):
  '''
  RGB data was exported in the range between 0 and 0.3
  '''
  return (image - vmin)/(vmax-vmin) + vmin


def display_image(image_path,bands=[3,2,1],title=None,ax=None):
  '''
  Displaying RBG images by Red,Green,Blue in this order
  '''
  img =  rio.open(image_path)
  show(normalize(img.read(bands)), title=title,ax=ax)
  return img
  
  
def print_mask_stats(rasterio_img,title=None):
  '''
  Prints statistics about a mask classes
  '''
  classes,counts = np.unique(rasterio_img.read(1),return_counts=True)
  print(title + ' classes:',classes)
  sumc = sum(counts)
  pp.pprint(['class:{},count:{},{:.2f}%:'.format(classes[i],c*100,c*100/sumc) for i,c in enumerate(counts)])
  print(len(rasterio_img.indexes),' bands')
  
  
def print_image_metadata(rasterio_img,title=None):
  '''
  Prints rasterio image metadata
  '''    
  print(title + ' Metadata:',rasterio_img.meta)
  print(title + ' Transform:\n',rasterio_img.transform)
  
  
def rename_images(s2_path):
  '''
  Rename images in path - removed spaces from image names
  '''        
  s2_files = glob.glob(s2_path + '/*.tif')
  for path in tqdm(s2_files):
      newname =  path.replace(' ', '_')
      # print(newname,path)
      if newname != path:
          os.rename(path,newname)
          

def batch_resize(src_dir,target_dir,n_bands,rsmpl=Resampling.nearest):
  '''
  Lazy batch resize multi band TIF images  band into (300,300) + adding 4 bands
  Using Rasterio reprojection
  '''
  with rio.Env():
    print('Source dir:',src_dir)
    print('Target dir:',target_dir)

    if(os.path.isdir(target_dir)):
      print('{} Directory exists, exiting(DELETE it and rerun to force regeneration)'.format(target_dir))
      target = os.listdir(target_dir)
      number_files = len(target)
      print(number_files)
      return

    print('Creating ',target_dir )
    os.makedirs(target_dir)
    
    target_size = (n_bands,300,300)
    src_files = glob.glob(src_dir + '/*.tif')
    print(len(src_files),'images to resize + additional 4 band to ',target_size)

    for src_file in tqdm(src_files):
        fname = Path(src_file).name

        with rio.open(os.path.join(src_dir,fname)) as src:
          rows, cols = src_shape = src.shape
          src_transform = src.transform
          src_crs = src.crs
          source = src.read()
          #adding NDVI band
          ndvi = get_ndvi(src)
          source = np.concatenate((source,ndvi),axis=0)
          #adding  band
          ndti = get_ndti(src)
          source = np.concatenate((source,ndti),axis=0)
          # #adding  band
          ndvire = get_ndvire(src)
          source = np.concatenate((source,ndvire),axis=0)
          # #adding  band
          mndwi = get_mndwi(src)
          source = np.concatenate((source,mndwi),axis=0)

          dst_shape = target_size
          dst_transform = src.transform
          dst_crs = src.crs 
          destination = np.zeros(dst_shape, src.dtypes[0])
          
          reproject(
            source,
            destination,
            dst_width=target_size[1],
            dst_height=target_size[2],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=rsmpl)

          # Write it out to a file.
          with rio.open(
                  os.path.join(target_dir,fname),'w',
                  driver='GTiff',
                  width=dst_shape[1],
                  height=dst_shape[2],
                  count=n_bands,#src.count+1,
                  dtype=src.dtypes[0],
                  nodata=0,
                  transform=dst_transform,
                  crs=dst_crs) as dst:
                      dst.write(destination, indexes=range(1,n_bands+1))


def get_matching_img_path(fname,src_img_dir):
    '''
    Return the matching image for a given mask
    '''
    suffix = fname.split('_')
    img_file = glob.glob(src_img_dir + '/*_'+suffix[-1])
    return img_file[0]


def get_matching_mask_path(img_name,mask_dir=None):
    '''
    Return the matching mask of a given image from the masks directory
    '''
    suffix = img_name.split('_')
    mask_file = glob.glob(mask_dir + '/*_'+suffix[-1])
    return mask_file[0]
    

def batch_projection(src_dir,target_dir,orig_img_dir,rsmpl=Resampling.nearest):
  '''
  Lazy batch transform of ESM MASKs with 1 band into the the Sentinel2 image CRS
  Using Rasterio reprojection
  '''
  with rio.Env():
    print('Source dir:',src_dir)
    print('Target dir:',target_dir)
    print('Orig images dir:',orig_img_dir)

    if(os.path.isdir(target_dir)):
      print('{} Directory exists, exiting(DELETE it and rerun)'.format(target_dir))
      target = os.listdir(target_dir)
      number_files = len(target)
      print(number_files)
      return

    print('Creating ',target_dir )
    os.makedirs(target_dir)

    # dst_crs = 'EPSG:3857'
    src_files = glob.glob(src_dir + '/*.tif')
    print(len(src_files),'images to transform')

    for src_file in tqdm(src_files):
      fname = Path(src_file).name

      with rio.open(get_matching_img_path(fname,orig_img_dir)) as matching_image:
        
        with rio.open(os.path.join(src_dir,fname)) as src:
          rows, cols = src_shape = src.shape
          src_transform = src.transform
          src_crs = src.crs

          # print(src,rows,cols,src_transform,src_crs)
          source = src.read(1)

          dst_shape = matching_image.shape
          dst_transform = matching_image.transform
          dst_crs = matching_image.crs 
          destination = np.zeros(dst_shape, np.uint8)
          
          reproject(
          source,
          destination,
          src_transform=src_transform,
          src_crs=src_crs,
          #  nodata=0,
          dst_transform=dst_transform,
          dst_crs=dst_crs,
          resampling=rsmpl)

          # Write it out to a file.
          with rio.open(
                  os.path.join(target_dir,fname),
                  'w',
                  driver='GTiff',
                  width=dst_shape[1],
                  height=dst_shape[0],
                  count=1,
                  dtype=np.uint8,
                  nodata=0,
                  transform=dst_transform,
                  crs=dst_crs) as dst:
              dst.write(destination, indexes=1)


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
    if isinstance(mask,torch.Tensor): #Used to be TensorBase
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


def calc_similiarity(imgA,imgB):
  '''
  Calculates the MSE and Similiarity between two images (usually not a good metric, but it used as a reference)
  '''
  mse_score = mean_squared_error(imgA, imgB)
  ssim_score = ssim(imgA, imgB, data_range=imgA.max() - imgA.min())
  return mse_score,ssim_score

  
##################
# Plotting utils #
##################
def plot_hist(df,title,axx,c,lbl):
  sns.distplot(df,bins=125,ax=axx,color=c,hist_kws=dict(alpha=0.3),label=lbl)
  axx.set_title(title)
  axx.set_ylabel('Number of images')
  axx.set_xlabel('Proportion of 250(Non Resi) + 255 (Resi) / All classes')
  axx.legend()
  axx.set_xlim(0.05,0.9)


def annotate(ax):
  for p in ax.patches:
    ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x() * 1.010, p.get_height() * 1.010))
  
def plot_simple_bar(df,title,axx,pct=True):
  if(pct==True):
    total_counts=df.sum(axis=0).sum()
  else:
    total_counts = 1
  (df.sum(axis=0)/total_counts).plot.bar(ax=axx,title=title,color=[ 'cyan','orange', 'green', 'red'],alpha=0.7)


def calc_stats_dfs(img_path,n_bands):
  '''
  calculates the per band mean and STD and return 2 dataframes with this info
  '''
  stats_mean_df = pd.DataFrame(columns=range(1,n_bands))
  stats_std_df = pd.DataFrame(columns=range(1,n_bands))
  s2_images = glob.glob(img_path+ '/*.tif')
  for row,image in tqdm(enumerate(s2_images),total=len(s2_images)):
    with rio.open(image,'r') as im:
      for c in range(1,n_bands+1):#im.count+1):
        chn=np.array(im.read(c))
        #imputing mean and STD of  exported images with the mean and STD of the other non nans values in the channel
        if(np.isnan(chn).sum()>0):
          not_nan =(np.isnan(chn)==False)
          stats_mean_df.loc[row,c] =  chn[not_nan].mean()
          stats_std_df.loc[row,c] = chn[not_nan].std()
          # print(np.isnan(chn).sum())
        else:
          stats_mean_df.loc[row,c] = chn.mean()
          stats_std_df.loc[row,c] = chn.std()
  return stats_mean_df,stats_std_df
  
def plot_bands_dist(n_bands,mean_df,std_df,mean_df2=[],std_df2=[]):
  '''
  Plots a comparison between 2 datasets bands distribution
  '''
  f,axs = plt.subplots(1,n_bands,figsize=(24,2))
  x = np.arange(-1.1,1.1, 1e-04)
  for i in range(1,n_bands+1):
    mean_of_band = mean_df.mean()[i]
    std_of_band = std_df.mean()[i]
    axs[i-1].plot(x, scipy.stats.norm.pdf(x, mean_of_band,std_of_band ),label=i,color='blue',alpha=0.5)
    axs[i-1].set_xlim([mean_of_band-(4*std_of_band),mean_of_band+(4*std_of_band)])
    # mean_df[i].plot.box(ax=axs[i-1],color='blue')
    if(len(mean_df2)>0):
      mean_of_band2 = mean_df2.mean()[i]
      std_of_band2 = std_df2.mean()[i]
      axs[i-1].plot(x, scipy.stats.norm.pdf(x, mean_of_band2,std_of_band2 ),label=i,color='red',alpha=0.5)
      axs[i-1].set_xlim([mean_of_band2-(4*std_of_band2),mean_of_band2+(4*std_of_band2)])
    axs[i-1].set_title(i)
    plt.tight_layout()