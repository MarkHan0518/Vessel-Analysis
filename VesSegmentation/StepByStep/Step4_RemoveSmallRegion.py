"""
This module contains the code for "remove small regions" step needed for a vessel segmentation pipeline.
The user needs to define filepaths at the beginning of the main script.
"""


import time
import numpy as np
from skimage import measure
from tifffile import imsave, imread


def remove_small_region(image, border = 16, area_size=100, extent=0.4, small_region_size=40) -> np.ndarray:
    """Remove Small Regions and Non-Vascular Regions from a 3D Image

    This function takes a 3D image containing segmented regions and removes small
    and non-vascular regions based on their area and extent. The result is an image with the targeted regions removed.

    Parameters:
    ----------
    image: ndarray
        A 3D numpy array containing segmented regions.
    area_size: int, optional
        Threshold for small region size. Default is 100.
    extent: float, optional
        Threshold for region extent to determine non-vascular regions. Default is 0.4.
    small_region_size: int, optional
        Threshold for very small region size. Default is 40.

    Returns:
    ----------
    ndarray
        A 3D numpy array with small and non-vascular regions removed.

    Notes:
    ----------
        - 'extent' here refers to the ratio of isolated region size to the external bounding box size.
        - The function uses skimage's regionprops to analyze and filter regions.
    
    Reference:
    ----------
    - HP-VSP Library (source of the code):
    https://github.com/visionlyx/HP-VSP/blob/main/vascular%20segmentation%20pipeline/mpi_block_fusion.py
    """
    labels = measure.label(image)
    properties = measure.regionprops(labels)
    
    for props in properties:
        print(f"Label: {props.label}, Area: {props.area}, Centroid: {props.centroid}, Extent: {props.extent}")

    for i in range(0, len(properties)):
        if ((properties[i].area < area_size and properties[i].extent >= extent
             and properties[i].centroid[0] >= border and properties[i].centroid[0] < image.shape[0]-border
             and properties[i].centroid[1] >= border and properties[i].centroid[1] < image.shape[1]-border
             and properties[i].centroid[2] >= border and properties[i].centroid[2] < image.shape[2]-border)
            or properties[i].area < small_region_size):

            temp = ~(labels == properties[i].label)
            temp = temp.astype(np.uint8)
            image = image * temp

    return image

##### ---------- Define Parameters ---------- #####
save_path = '' # Filepath where the hole_filling.tif is saved and smoothing.tif will be saved
###################################################

hole_filling = imread(save_path + 'hole_filling.tif')
remove_start = time.time()
remove_region = remove_small_region(hole_filling).astype(np.uint8)
imsave(save_path + 'remove_region.tif', remove_region)
remove_end = time.time()
remove_duration = remove_end-remove_start
print("Remove small regions time: " + str(round(remove_duration)) + ' s')