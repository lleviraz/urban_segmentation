###########
# Imports #
###########
# import os
# import glob
# import scipy
import numpy as np
# import pprint as pp
# import pandas as pd
# import seaborn as sns
# from pathlib import Path
# from tqdm.notebook import tqdm
# import matplotlib.pyplot as plt
# from fastai.torch_core import TensorBase
# import rasterio as rio
# from rasterio.plot import show,show_hist
# from rasterio.warp import reproject, Resampling
# #metrics
# from skimage.metrics import mean_squared_error
# from skimage.metrics import structural_similarity as ssim

####################
# Extra bands      #
####################

def get_ndvi(img):
  '''
  NDVI : Normalized difference vegetation index - Added as **B12**
  (B7-B3)/(B7+B3)
  '''
  B7 = img.read(7)
  B3 = img.read(3)
  NDVI = (B7-B3)/(B7+B3)
  return np.expand_dims(NDVI, axis=0)
  
def get_ndti(img):
  '''
  NDTI:Normalized Difference Tillage Index  - Added as **B13**
  (B10-B11)/(B10+B11)
  '''
  B10 = img.read(10)
  B11 = img.read(11)
  NDTI = (B10-B11)/(B10+B11)
  return np.expand_dims(NDTI, axis=0)
 
def get_ndvire(img):
  '''
  NDVIre: Normalized Difference Vegetation Index  - Added as **B14**
  ((B4 - B3)) / ((B4 + B3))
  '''
  B4 = img.read(4)
  B3 = img.read(3)
  NDVIre = (B4-B3)/(B4+B3)
  return np.expand_dims(NDVIre, axis=0)
  
def get_mndwi(img):
  '''
  MNDWI: Modified Normalized Difference Water Index  - Added as **B15**
  ((B2 - B10)) / ((B2 + B10))
  '''
  B2 = img.read(2)
  B10 = img.read(10)
  MNDWI = (B2-B10)/(B2+B10)
  return np.expand_dims(MNDWI, axis=0)