"""
This module contains the code for "remove small regions" step needed for a vessel segmentation pipeline.
The user needs to define filepaths at the beginning of the main script.
"""


import time
import numpy as np
from skimage import morphology
from tifffile import imsave, imread


##### ---------- Define Parameters ---------- #####
save_path = '' # Filepath where the hole_filling.tif is saved and smoothing.tif will be saved
object_min_size = 40
###################################################

hole_filling = imread(save_path + 'hole_filling.tif')
remove_start = time.time()

hole_filling_bool = hole_filling.astype(bool)
remove_region_bool = morphology.remove_small_objects(hole_filling_bool, object_min_size)
remove_region = remove_region_bool.astype(np.uint8)

remove_end = time.time()
remove_duration = remove_end-remove_start
imsave(save_path + 'remove_region.tif', remove_region)
print("Remove small regions time: " + str(round(remove_duration)) + ' s')