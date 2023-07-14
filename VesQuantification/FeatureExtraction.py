"""
This module contains code for feature extraction.
The user needs to define filepaths at the beginning of the main script.
"""


from tifffile import imsave, imread
import Functions as functions
import time
import numpy as np


##### ----- Parameters ----- #####
READPATH = '' # filepath of the vessel mask after segmentation
SAVEPATH = '' # filepath where the masks for each feature are saved
##################################


segmentation = imread(READPATH)
##### ----- Centerlines ----- #####
centerlines = functions.extract_centerlines(segmentation)
imsave(SAVEPATH + 'centerlines.tif', centerlines)


##### ----- Bifurcations ----- #####
bif_start = time.time()
bifurcations = functions.extract_bifurcations(centerlines).astype(np.uint8)*255
bif_end = time.time()
bif_duration = bif_end-bif_start
print("Extract bifurcations duration: " + str(round(bif_duration, 2)) + " s")
imsave(SAVEPATH + 'bifurcations.tif', bifurcations)


##### ----- Radius ----- #####
rad_start = time.time()
radius = functions.extract_radius(segmentation, centerlines).astype(np.uint8)
rad_end = time.time()
rad_duration = rad_end-rad_start
print("Extract radius duration: " + str(rad_duration) + " s")
imsave(SAVEPATH + 'radius.tif', radius)