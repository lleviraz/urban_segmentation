# urban_segmentation
Urban segmentation on multi channel satellite images to identify Residential and Industrial built up areas

MLDS MSc Final Project - Reichman University 2022

![urban segmentation pipeline](urb_seg_ppln.png) 

## Intro
The repo contains the code implementation and trained models for the pipeline described above.
The model was trained on 700 [Sentinel2 GeoTiff images](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR) and 700 matching [ESM masks](https://land.copernicus.eu/pan-european/GHSL/european-settlement-map/esm-2015-release-2019?tab=download)
A sample input image (before and after preprocessing) and matching ESM mask:
![sample image](sample_input.png)

Our model scored a [Dice]([https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2#:~:text=The%20Dice%20coefficient%20is%20very,similarity%20between%20predicted%20and%20truth](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2)) score of 0.83 on test images and was successfully used to predict images on another part of the world:
![sample pred](sample_pred.png)

## Code
See the [notebooks](notebooks/) folder for all relevant notebooks:
- [NB0](notebooks/NB0%20Exporting%20Sentinel%202%20Images%20from%20GEE.ipynb) - Download new images from Google Earth Engine
- [NB1](notebooks/NB1%20Preprocess%20and%20Predict.ipynb) - Preprocessing and inference only
- [NB2](notebooks/NB2%20EDA%20and%20Preprocessing%20for%20Training.ipynb) - EDA and Preprocessing for Training
- [NB3](notebooks/NB3%20Preprocessing%20for%20Inference.ipynb) - Minimal Preprocessing for Training or Inference
- [NB4](notebooks/NB4%20Urban%20Segmentation%20Training%20Pipeline%20(PyTorch%20UNET).ipynb) - Training a new UNET segmentation model
- [NB5](notebooks/NB5%20Urban%20Segmentation%20Pipeline%202%20(PyTorch%20DeepLabV3).ipynb) - Training a new DeepLabv3 segmentation model
- [NB6](notebooks/NB6%20Inference%20Pipeline%20(PyTorch%20UNET).ipynb) - Inference only
- [NB7](notebooks/NB7%20Experiments%20Results%20Analysis.ipynb) - Training results analysis

**Note:** here is a link to a Youtube playlist with some "walkthrough" videos for the download and inference parts [click here](https://youtube.com/playlist?list=PLQSmztpAE0KeKD9sWmX5D3PNuLWXZ46Tj)

# How to use this repo
## Downloading new Satellite images for prediction
- Use [NB0](notebooks/NB0%20Exporting%20Sentinel%202%20Images%20from%20GEE.ipynb) to download new images, you will need a [GEE account](https://code.earthengine.google.com/) and a Google Drive with free space for storing the images.
- The output will be a directory in google drive with your downloaded images.

## Inference
- Use [NB1](notebooks/NB1%20Preprocess%20and%20Predict.ipynb) to preprocess and predict new masks for your downloaded images
- Or if you prefer you can preprocess using [NB3](notebooks/NB3%20Preprocessing%20for%20Inference.ipynb) and then do Inference separately using [NB6](notebooks/NB6%20Inference%20Pipeline%20(PyTorch%20UNET).ipynb)
- The output will be a directory with the predicted masks

## Training a new model for your data
- Use [NB2](notebooks/NB2%20EDA%20and%20Preprocessing%20for%20Training.ipynb) (or [NB3](notebooks/NB3%20Preprocessing%20for%20Inference.ipynb)) for Preprocessing and [NB4](notebooks/NB4%20Urban%20Segmentation%20Training%20Pipeline%20(PyTorch%20UNET).ipynb) (or [NB5](notebooks/NB5%20Urban%20Segmentation%20Pipeline%202%20(PyTorch%20DeepLabV3).ipynb)) for Training
- Use [NB7](notebooks/NB7%20Experiments%20Results%20Analysis.ipynb) for analyzing
- The output will be a PyTorch model, and a directory with the predicted masks used for testing


