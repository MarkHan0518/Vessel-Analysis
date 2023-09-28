"""
This module contains the code for "keeping the largest region" step.
It is an OPTIONAL step.
The user needs to define filepaths at the beginning of the main script.
"""


import time
import numpy as np
from skimage import measure
from tifffile import imsave, imread


def keep_largest_region(image):
    """Keep the Largest Region (Object) in a 3D Image

    This function takes a 3D image containing segmented regions and keeps only the largest region (object).

    Parameters:
    ----------
    image: ndarray
        A 3D numpy array containing segmented regions.

    Returns:
    ----------
    ndarray
        A 3D numpy array with only the largest region (object) retained.
    """
    labels = measure.label(image)
    properties = measure.regionprops(labels)

    # Find the largest region (object)
    largest_area = 0
    largest_label = 0
    for props in properties:
        if props.area > largest_area:
            largest_area = props.area
            largest_label = props.label

    # Create a mask to keep only the largest region (object)
    largest_region_mask = labels == largest_label

    # Apply the mask to the input image
    largest_region_image = image * largest_region_mask

    return largest_region_image


##### ---------- Define Parameters ---------- #####
save_path = '' # Filepath where the hole_filling.tif is saved and largest_region.tif will be saved
###################################################

remove_start = time.time()

hole_filling = imread(save_path + 'hole_filling.tif')
largest_region = keep_largest_region(hole_filling)
largest_region = largest_region.astype(np.uint8)
imsave(save_path + 'largest_region.tif', largest_region)
print('Largest region retained.')

labels = measure.label(largest_region)
properties = measure.regionprops(labels)
len_after = len(properties)
print(len_after)

remove_end = time.time()
remove_duration = remove_end - remove_start
print("Remove small regions time: " + str(round(remove_duration)) + ' s')