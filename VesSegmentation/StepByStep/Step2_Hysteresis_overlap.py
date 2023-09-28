"""
This module contains the code for "hysteresis thresholding" step needed for a vessel segmentation pipeline.
The user needs to define filepaths and parameters at the beginning of the main script.
"""


import tifffile
import numpy as np
from skimage import filters

##### ---------- Define Parameters ---------- #####
save_path = '' # Filepath where the divide.tif is saved and hyst.tif will be saved
lowt = 2 # Lower threshold for hysteresis thresholding
hight = 3 # Higher threshold for hysteresis thresholding
ENOUGH_MEMORY = False
###################################################

images = tifffile.imread(save_path + 'divide.tif')
d, h, w = images.shape

if ENOUGH_MEMORY:
    hyst = filters.apply_hysteresis_threshold(images, lowt, hight).astype(np.uint8)
    tifffile.imsave(save_path + 'hyst.tif', hyst)
else:
    # Process the data in chunks with overlapping regions
    chunk_size = 1000  # You can adjust this based on your memory capacity
    num_chunks = (d + chunk_size - 1) // chunk_size

    # Define the size of the overlapping region between chunks
    overlap_size =  int(chunk_size / 5) # You can adjust this based on your requirements and the chunk size

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, d)

        # Extend the start and end indices to include the overlapping region
        start_idx_with_overlap = max(0, start_idx - overlap_size)
        end_idx_with_overlap = min(d, end_idx + overlap_size)

        # Load the chunk with overlapping region
        chunk_with_overlap = images[start_idx_with_overlap:end_idx_with_overlap, :, :]

        # Apply hysteresis thresholding on the chunk (including the overlapping region)
        hyst = filters.apply_hysteresis_threshold(chunk_with_overlap, lowt, hight).astype(np.uint8) * 255

        if i == 0:
        # Trim the overlapping region to get the final result for the chunk
            hyst_result = hyst[0 : end_idx - start_idx]
        else:
            hyst_result = hyst[overlap_size : overlap_size + (end_idx - start_idx)]

        # Save the result for this chunk
        tifffile.imsave(save_path + f'hyst_overlap_{i}.tif', hyst_result)

print("Hysteresis thresholding with overlapping regions completed!")