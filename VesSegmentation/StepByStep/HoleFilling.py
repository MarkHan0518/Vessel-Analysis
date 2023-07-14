"""
This module contains the code for "hole filling" step needed for a vessel segmentation pipeline.
The user needs to define filepaths at the beginning of the main script.
"""


import numpy as np
from tifffile import imsave
from scipy import ndimage as ndi
import time
import tifffile


save_path = '' # Filepath where the hyst.tif is saved and hole_filling.tif will be saved
hyst = tifffile.imread(save_path + 'hyst.tif')

fill_start = time.time()
hole_filling = ndi.binary_fill_holes(hyst).astype(np.uint8)
imsave(save_path +'hole_filling.tif', hole_filling)

fill_end = time.time()
fill_duration = fill_end-fill_start
print("Fill hole time: " + str(round(fill_duration)) + ' s')