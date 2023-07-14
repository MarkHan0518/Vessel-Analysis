"""
This module contains the code for "hysteresis thresholding" step needed for a vessel segmentation pipeline.
The user needs to define filepaths at the beginning of the main script.
"""


import numpy as np
from tifffile import imsave
from skimage import filters
import h5py as h5
import tifffile


filedir = '' # Filepath of the HDF5 dataset.
save_path = '' # Filepath where the Multi_otsu.tif is saved and hyst.tif will be saved


ch = 't00000/s00/0/cells' # use level 0 (original) dataset
with h5.File(filedir, 'r') as f:
    image_3d = f[ch]
    d,h,w = image_3d.shape


images = tifffile.imread(save_path + 'Multi_otsu.tif', key=range(0, d, 1))
lowt = 1
hight = 3


hyst = filters.apply_hysteresis_threshold(images, lowt, hight).astype(np.uint8)*255
imsave(save_path +'hyst.tif', hyst)