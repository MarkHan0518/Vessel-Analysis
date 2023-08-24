"""
This module contains the code for "chunk edge smoothing" step needed for a vessel segmentation pipeline.
The user needs to define filepaths and parameters at the beginning of the main script.
"""


import time
import tifffile
import numpy as np


save_path = '' # Filepath where the otsu_3d.tif and count_3d.tif are saved and hyst.tif will be saved

otsu_3d = tifffile.imread(save_path + 'otsu_3d.tif')
count_3d = tifffile.imread(save_path + 'count_3d.tif')
d, w, h = otsu_3d.shape

start = time.time()
data = (np.divide(otsu_3d, count_3d)).astype(np.uint8)
tifffile.imsave(save_path + 'divide.tif', data)

end = time.time()
duration = end - start
print("Dividing time: " + str(round(duration, 2)) + ' s')