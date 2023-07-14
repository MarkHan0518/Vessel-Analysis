"""
This module contains useful functions for feature extraction.
"""

from skimage.morphology import skeletonize_3d
import numpy as np
from scipy import ndimage as ndi


def extract_centerlines(segmentation):
    '''
    Parameter:
    - segmentation: 3D binary mask

    Function:
    Compute the skeleton of a 3D binary image

    Return:
    A ndarray represents the thinned image
    '''

    skeleton = skeletonize_3d(segmentation)
    skeleton.astype(dtype='uint8', copy=False)
    return skeleton

def extract_bifurcations(centerlines):
    '''
    Parameter:
    - centerlines: skeleton of the 3D binary mask

    Function:
    Find the bifurcations

    Return:
    A ndarray represents the point of bifurcations

    Important:
    Must apply "ImageJ" -> "Analyze" -> "3D Object Counter" to the resulted
    bifurcation mask to obtain correct number of bifucations
    '''

    a = centerlines
    a.astype(dtype='uint8', copy=False)
    sh = np.shape(a)
    bifurcations = np.zeros(sh,dtype='uint8')

    for x in range(1,sh[0]-1):
        for y in range(1,sh[1]-1):
            for z in range(1,sh[2]-1):
                if a[x,y,z]== 1:
                    local = np.sum([a[ x-1,  y-1,  z-1],
                    a[ x-1,  y-1,  z],
                    a[ x-1,  y-1,  z+1],
                    a[ x-1,  y,  z-1],
                    a[ x-1,  y,  z],
                    a[ x-1,  y,  z+1],
                    a[ x-1,  y+1,  z-1],
                    a[ x-1,  y+1,  z],
                    a[ x-1,  y+1,  z+1],
                    a[ x,  y-1,  z-1],
                    a[ x,  y-1,  z],
                    a[ x,  y-1,  z+1],
                    a[ x,  y,  z-1],
                    a[ x,  y,  z],
                    a[ x,  y,  z+1],
                    a[ x,  y+1,  z-1],
                    a[ x,  y+1,  z],
                    a[ x,  y+1,  z+1],
                    a[ x+1,  y-1,  z-1],
                    a[ x+1,  y-1,  z],
                    a[ x+1,  y-1,  z+1],
                    a[ x+1,  y,  z-1],
                    a[ x+1,  y,  z],
                    a[ x+1,  y,  z+1],
                    a[ x+1,  y+1,  z-1],
                    a[ x+1,  y+1,  z],
                    a[ x+1,  y+1,  z+1]])

                    if local > 3*1:
                        bifurcations[x,y,z] = 1
    
    bifurcations.astype(dtype='uint8', copy=False)
    return bifurcations


def extract_radius(segmentation, centerlines):
    '''
    Parameters:
    - segmentation: 3D binary mask
    - centerlines: skeleton of the 3D binary mask

    Function:
    Exact Euclidean distance transform and convolve it with the centerlines

    Return:
    Similar to centerlines, but each pixel represents the Euclidean distance instead of just 0 and 1

    Important:
    Must multiply the Euclidean distance with the voxel size to obtain the actual radius of the vessel
    '''

    transf = ndi.distance_transform_edt(segmentation)
    radius_matrix = transf*centerlines
    return radius_matrix