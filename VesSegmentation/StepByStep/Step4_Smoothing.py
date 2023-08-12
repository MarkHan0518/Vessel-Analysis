"""
This module contains the code for "smoothing" step needed for a vessel segmentation pipeline.
The user needs to define filepaths at the beginning of the main script.
"""


from tifffile import imsave
import time
import tifffile
from scipy.ndimage import median_filter

save_path = '' # Filepath where the hole_filling.tif is saved and smoothing.tif will be saved
fill_hole = tifffile.imread(save_path + 'hole_filling.tif')

smooth_start = time.time()

size = 3
smoothed = median_filter(fill_hole, size)
imsave(save_path + 'smoothing.tif', smoothed)

smooth_end = time.time()
smooth_duration = smooth_end-smooth_start
print("Smooth time: " + str(round(smooth_duration)) + ' s')