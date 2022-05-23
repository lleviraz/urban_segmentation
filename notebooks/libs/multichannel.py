#This file holds the multichannel relevant code snippents and functions

# Python General
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import glob
from tqdm.notebook import tqdm
import pprint as pp
from PIL import Image
import numpy as np

#Torch
from torch import nn
import torch
import torch.nn.functional as F
from torchvision.models.resnet import resnet18,resnet34

#rasterio
import rasterio as rio
from rasterio.transform import Affine
from rasterio.plot import show_hist,show
from rasterio.warp import calculate_default_transform, reproject, Resampling

#fastai
from fastai.vision.all import *
from fastcore.xtras import Path
from fastai.callback.hook import summary
from fastai.callback.progress import ProgressCallback
from fastai.callback.schedule import lr_find, fit_flat_cos
from fastai.data.block import DataBlock
from fastai.data.external import untar_data, URLs
from fastai.data.transforms import get_image_files, FuncSplitter, Normalize
from fastai.layers import Mish
from fastai.losses import BaseLoss,DiceLoss,FocalLossFlat
from fastai.optimizer import ranger,Adam
from fastai.torch_core import tensor
from fastai.vision.augment import aug_transforms
from torchvision.transforms import RandomHorizontalFlip,RandomRotation,ColorJitter
from fastai.vision.learner import unet_learner
from fastai.basics import *
from fastai.vision import *
from fastai.vision.core import *

from fastai.vision.data import *
from fastai.data import *

@patch
@delegates(subplots)
def plot_metrics(self: Recorder, nrows=None, ncols=None, figsize=None, final_losses=True, perc=.5, **kwargs):
    '''
    Plots the final metrics plots after training
    '''
    n_values = len(self.recorder.values)
    if n_values < 2:
        print('not enough values to plot a chart')
        return
    metrics = np.stack(self.values)
    names = self.metric_names[1:-1]
    metric_names = [m.replace("valid_", "") for m in self.metric_names[1:-1] if 'loss' not in m and 'train' not in m]
    if final_losses:
        sel_idxs = int(round(n_values * perc))
        if sel_idxs < 2:
            final_losses = False
        else:
            names = names + ['train_final_loss', 'valid_final_loss']
            self.loss_idxs = L([i for i,n in enumerate(self.metric_names[1:-1]) if 'loss' in n])
            metrics = np.concatenate([metrics, metrics[:, self.loss_idxs]], -1)

    n = int(1 + final_losses + len(self.metrics))
    if nrows is None and ncols is None:
        if n <= 3:
            nrows = 1
        else:
            nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None: nrows = int(np.ceil(n / ncols))
    elif ncols is None: ncols = int(np.ceil(n / nrows))
    figsize = figsize or (ncols * 6 + ncols - 1, nrows * 4 + nrows - 1)
    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = axs.flatten()[:n]
    for i,name in enumerate(names):
        xs = np.arange(0, len(metrics))
        if name in ['train_loss', 'valid_loss']:
            ax_idx = 0
            m = metrics[:,i]
            title = 'losses'
        elif name in ['train_final_loss', 'valid_final_loss']:
            ax_idx = 1
            m = metrics[-sel_idxs:,i]
            xs = xs[-sel_idxs:]
            title = 'final losses'
        else:
            ax_idx = metric_names.index(name.replace("valid_", "").replace("train_", "")) + 1 + final_losses
            m = metrics[:,i]
            title = name.replace("valid_", "").replace("train_", "")
        if 'train' in name:
            color = '#1f77b4'
            label = 'train'
        else:
            color = '#ff7f0e'
            label = 'valid'
            axs[ax_idx].grid(color='gainsboro', linewidth=.5)
        axs[ax_idx].plot(xs, m, color=color, label=label)
        axs[ax_idx].set_xlim(xs[0], xs[-1])
        axs[ax_idx].legend(loc='best')
        axs[ax_idx].set_title(title)
    plt.show()


@patch
@delegates(subplots)
def plot_metrics(self: Learner, **kwargs):
    self.recorder.plot_metrics(**kwargs)


def _using_attr(f, attr, x):
    return f(getattr(x,attr))

def using_attr(f, attr):
    "Change function `f` to operate on `attr`"
    return partial(_using_attr, f, attr)    

#########################################################################
# A combined Focal and Dice Loss for segmenation - based on fastai docs #
#########################################################################
class CombinedLoss:
    '''
    Dice and Focal combined https://docs.fast.ai/losses.html
    '''
    def __init__(self, axis=1, smooth=1e-06, alpha=1.,gamma=2.0,reduction='mean', sin=False):
        self.axis,self.smooth,self.alpha,self.gamma,self.reduction,self.sin = \
          axis, smooth, alpha,gamma,reduction, sin
        store_attr()
        self.focal_loss = FocalLossFlat(axis=axis,gamma=gamma,reduction=reduction)
        self.dice_loss =  DiceLoss(axis=axis, smooth=smooth,reduction=reduction, square_in_union=sin)
        
    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)
    
    def decodes(self, x):
      return x.argmax(dim=self.axis)

    def activation(self, x): 
      return F.softmax(x, dim=self.axis)

    def display(self):
      return 'DiceFocalLoss({},{},{},{},{},{})'.format(self.axis,self.smooth,
                                                       self.alpha,self.gamma,
                                                       self.reduction,self.sin)
                                                       

def get_msk(img_name,mask_dir, p2c={0:0,1:1,2:250,3:255},merge_bg=False):
  '''
  Returns a matching mask for an image using the codes decoder dictionary (0,1,2,3)->(0,1,250,255)
  '''
  fn = get_matching_mask_path(img_name,mask_dir)
  msk = np.array(PILMask.create(fn))

  for i, val in enumerate(p2c):
    msk[msk==p2c[i]] = val
  if(merge_bg):
    msk[msk==0] = 1
  msk[0,0]=0 #save one pixel as 0 to preseve the "colors" when comparing
  return PILMask.create(msk)
  

# def get_msk(img_name,mask_dir, p2c):
#   '''
#   Returns a matching mask for an image using the codes decoder dictionary (0,1,2,3)->(0,1,250,255)
#   '''
#   fn = get_matching_mask_path(img_name,mask_dir)
#   msk = np.array(PILMask.create(fn))
#   for i, val in enumerate(p2c):
#     msk[msk==p2c[i]] = val
#   return PILMask.create(msk)
  
def open_geotiff(fn, chans=None):
    with rio.open(str(fn)) as f:
        data = f.read()
        data = data.astype(np.float32)
    im = torch.from_numpy(data)
    if chans is not None: im = im[chans]
    return im

class MultiChannelTensorImage(TensorImage):
    _show_args = ArrayImageBase._show_args
    def show(self, channels=[1], ctx=None, vmin=None, vmax=None, **kwargs):
        if len(channels) >= 3: 
            return show_composite(self, channels=channels, ctx=ctx, vmin=vmin, vmax=vmax,
                                  **{**self._show_args, **kwargs})
        elif len(channels) == 1: 
            return show_single_channel(self, channel=channels[0], ctx=ctx, 
                                       **{**self._show_args, **kwargs})

    @classmethod
    def create(cls, fn:(Path,str,ndarray), chans=None,  **kwargs) ->None:
        if isinstance(fn, Tensor): fn = fn.numpy()
        if isinstance(fn, ndarray): 
            im = torch.from_numpy(fn)
            if chans is not None: im = im[chans]
            return cls(im)
        if isinstance(fn, Path) or isinstance(fn, str):
            return cls(open_geotiff(fn=fn, chans=chans))
        
    def __repr__(self): return f'{self.__class__.__name__} size={"x".join([str(d) for d in self.shape])}'
    
    def __str__(self): return f'{self.__class__.__name__} size={"x".join([str(d) for d in self.shape])}'
    
MultiChannelTensorImage.create = Transform(MultiChannelTensorImage.create) 
        
def show_composite(img, channels, ax=None, figsize=(3,3), title=None, scale=True,
                   ctx=None, vmin=None, vmax=None, scale_axis=(0,1), **kwargs)->plt.Axes:
    "Show three channel composite so that channels correspond to R, G and B"
    ax = ifnone(ax, ctx)
    if ax is None: _, ax = plt.subplots(figsize=figsize)    
    r, g, b = channels[0],channels[1],channels[2]
    tempim = img.data.cpu().numpy()
    im = np.zeros((tempim.shape[1], tempim.shape[2], 3))
    im[...,0] = tempim[r]
    im[...,1] = tempim[g]
    im[...,2] = tempim[b]

    if scale: im = norm(im, vmin, vmax, scale_axis)
    ax.imshow(im, **kwargs)
    ax.axis('off')
    if title is not None: ax.set_title(title)
    
    return ax

def show_single_channel(img, channel, ax=None, figsize=(3,3), ctx=None, 
                        title=None, **kwargs) -> plt.Axes:
    ax = ifnone(ax, ctx)
    if ax is None: _, ax = plt.subplots(figsize=figsize)    
    tempim = img.data.cpu().numpy()
    ax.imshow(norm(tempim[channel], vmin=tempim[channel].min(), vmax=tempim[channel].max()), **kwargs)
    ax.axis('off')
    if title is not None: ax.set_title(title)
    return ax

def norm(vals, vmin=None, vmax=None, axis=(0,1)):
    """
    For visualization purposes scale image with `(vals-vmin)/(vmax-vmin), 
    with vmin and vmax either specified or within 0.01 and 0.99 quantiles of all values
    """
    vmin = ifnone(vmin, np.quantile(vals, 0.01, axis=axis))
    vmax = ifnone(vmax, np.quantile(vals, 0.99, axis=axis))
    ret_im = (vals - vmin)/(vmax-vmin)
    ret_im[ret_im < 0] = 0
    ret_im[ret_im > 1] = 1
    return ret_im
    
def MultiChannelImageBlock(cls=MultiChannelTensorImage, chans=None): 
    "By default all 12 channels are loaded"
    return TransformBlock(partial(cls.create, chans=chans))
    
    
class MCISegmentationDataLoaders(DataLoaders):
    '''
    A wrapper around several `DataLoader`s with factory methods for segmentation of multi channel tiff images
    This is an extension to fastai's SegmentationDataLoaders (https://docs.fast.ai/vision.data.html#SegmentationDataLoaders )
    to support multi channel Tiff images.
    '''
    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_label_func_segm(cls, path, fnames, label_func, chans=None, 
                         extensions=['.tif'], valid_pct=0.2, seed=None, 
                         codes=None, item_tfms=None, batch_tfms=None, **kwargs):
        '''
        Create from list of `fnames` in `path`s with `label_func`.
        This method creates a new fastai (PyTorch) data loader from two directories of images and matching masks,
        It was needed to handle the Sentinel2 multi channel Tiff images (Default implementation supports RGB with 3 channels only)
        It is basically a copy of the original fastai class method: https://github.com/fastai/fastai/blob/master/fastai/vision/data.py#L180 
        except if it using the custom `MultiChannelImageBlock` that was defined above.
        - path (str): full path to train images
        - bs (int): the batch size to use with the data loader - default 5
        - codes (list): a list of all possible target classes in the mask
        - fnames (list): a list of all the image file names without their path
        - label_func (func): a function that returns the matching mask of an image
        - valid_pct (float 0 to 1): the precentage of images for validation during training - default 0.1
        - extensions (list):  list of image file fexts to use when loading image and mask files - default ['.tif']
        - channels (list of integers): the channels to read from the input image, and int from 1 to 12
        - seed (int): random seed for reproducing results (for numpy pytorch)
        - item_tfms (list of transforms): list of transformations (augmentations)  - (fastai) Item transforms are for collating and preparing for a batch and theyÃ¢â‚¬â„¢re run on the CPU
        - batch_tfms (list of transforms): list of transformations (augmentations) -  (fastai) Batch transforms are applied after everything is resized and batched up and done on the GPU
        '''        
        dblock = DataBlock(blocks=(MultiChannelImageBlock(chans=chans), 
                                   MaskBlock(codes=codes)),
                           splitter=RandomSplitter(valid_pct, seed=seed),
                           get_y=label_func,
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)
        res = cls.from_dblock(dblock, fnames, path=path, **kwargs)
        return res