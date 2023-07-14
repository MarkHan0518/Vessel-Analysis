"""
This module converts the center of mass (COM) of the bifurcation mask into a binary dataset.
It is needed for bifurcation quantification in FeatureQuantification.py.
The user needs to define filepaths at the beginning of the main script.
"""


from tifffile import imsave, imread
import numpy as np

READPATH = '' # filepath of the COM_bifurcation.tif after ImageJ 3D Obejcts Counter
SAVEPATH = '' # filepath where the COM_bifurcation_binary.tif is saved
mask = imread(READPATH)

def threshold_and_binarize(array_3d, threshold):
    return (array_3d > threshold).astype(np.uint8)

threshold = 0
data = threshold_and_binarize(mask, threshold)
imsave(SAVEPATH + 'COM_bifurcations_binary.tif', data)