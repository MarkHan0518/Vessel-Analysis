"""
This module contains the code for "hole filling" step needed for a vessel segmentation pipeline.
The user needs to define filepaths at the beginning of the main script.
"""


import time
import tifffile
import numpy as np
from tqdm import tqdm
from tifffile import imsave
from scipy import ndimage as ndi


save_path = '' # Filepath where the hyst.tif is saved and hole_filling.tif will be saved
hyst = tifffile.imread(save_path + 'hyst_overlap.tif')
fill_start = time.time()
hole_filling = np.zeros(hyst.shape, dtype = np.uint8)

for frame_idx in tqdm(range(hyst.shape[0]), desc="Hole filling x-y plane: "):
    hole_filling[frame_idx, :, :] = ndi.morphology.binary_fill_holes(hyst[frame_idx,:,:]).astype(np.uint8)

for frame_idx in tqdm(range(hyst.shape[1]), desc="Hole filling x-z plane: "):
    hole_filling[:, frame_idx, :] = ndi.morphology.binary_fill_holes(hole_filling[:,frame_idx,:]).astype(np.uint8)

for frame_idx in tqdm(range(hyst.shape[2]), desc="Hole filling y-z plane: "):
    hole_filling[:, :, frame_idx] = ndi.morphology.binary_fill_holes(hole_filling[:,:,frame_idx]).astype(np.uint8)
hole_filling = hole_filling.astype(np.uint8)

fill_end = time.time()
fill_duration = fill_end-fill_start
print("Hole filling time: " + str(round(fill_duration)) + ' s')
imsave(save_path +'hole_filling.tif', hole_filling)