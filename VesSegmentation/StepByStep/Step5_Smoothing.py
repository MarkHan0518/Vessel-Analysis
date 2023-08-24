"""
This module contains the code for "smoothing" step needed for a vessel segmentation pipeline.
The user needs to define filepaths at the beginning of the main script.
"""


from tifffile import imsave, imread
import time
from scipy.ndimage import median_filter


save_path = '' # Filepath where the remove_region.tif is saved and smoothing.tif will be saved
remove_region = imread(save_path + 'remove_region.tif')

smooth_start = time.time()

size = 3
smoothed = median_filter(remove_region, size)
imsave(save_path + 'smoothing.tif', smoothed)

smooth_end = time.time()
smooth_duration = smooth_end-smooth_start
print("Smooth time: " + str(round(smooth_duration)) + ' s')