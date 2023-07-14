"""
This module contains code for feature quantification.
The user needs to define filepath and parameters at the beginning of the main script.
"""


from tifffile import imread
import numpy as np


##### ----- Parameters ----- #####
READPATH = '' # filepath where the masks for each feature are saved
VOXEL_SIZE = 2.68 # 2x downsampling
LARGE_VESSEL_RADIUS = 40 # um
INTERMEDIATE_VESSEL_RADIUS = 15 # um
MICRO_VESSEL_RADIUS = VOXEL_SIZE # um
##################################


# Centerlines Quantifications
centerlines = imread(READPATH + 'centerlines.tif')

# 1. local vessel length & density (m per mm^3)
total_length_pixels = np.sum(centerlines) # pixels
total_length_um = total_length_pixels * VOXEL_SIZE # um
total_length_m = total_length_um / 1000000 # m
d, h, w = centerlines.shape
total_volume_voxels = d * h * w # voxels
total_volume_mm3 = total_volume_voxels * VOXEL_SIZE ** 3 # um^3
total_volume_m3 = total_volume_mm3 / 1000000000 # mm^3
length_density =total_length_m / total_volume_m3 # m per mm^3
print('local vessel length: ' + str(round(total_length_m, 2)) + ' m')
print('local vessel density: ' + str(round(length_density, 2)) + ' m per mm^3')


# Bifurcations Quantifications
# COM_bifurcations is obtained from applying "ImageJ" -> "Analyze" -> "3D Object Counter" to the bifurcation result
COM_bifurcations = imread(READPATH + 'COM_bifurcations_binary.tif')

# 1. local bifurcation density (count per mm^3)
total_bif = np.sum(COM_bifurcations)
bif_density =total_bif / total_volume_m3 # count per mm^3
print('local bifurcation density: ' + str(int(bif_density)) + ' count per mm^3')


# Radius Quantification
radius = imread(READPATH + 'radius.tif')

# 1. local vessel radius: max and mean (um)
av_rad_pixels = np.true_divide(radius.sum(),(radius!=0).sum()) # pixels
av_rad_um = av_rad_pixels * VOXEL_SIZE # um
max_rad_pixels = np.max(radius) # pixels
max_rad_um = max_rad_pixels * VOXEL_SIZE # um
print('Maximum radius: ' + str(round(max_rad_um, 2)) + ' um')
print('Mean radius: ' + str(round(av_rad_um, 2)) + ' um')

# 2. local vessel radius segmentation: large, intermediate, micro binary masks
shapes = np.shape(radius)
large_intermediate_trsh = LARGE_VESSEL_RADIUS / VOXEL_SIZE
intermediate_micro_trsh = INTERMEDIATE_VESSEL_RADIUS / VOXEL_SIZE
large_vessels = np.greater_equal(radius, large_intermediate_trsh).astype(np.uint8)
intermediate_vessels = np.logical_and(radius<large_intermediate_trsh, radius>=intermediate_micro_trsh).astype(np.uint8)
small_vessels = np.logical_and(radius<intermediate_micro_trsh, radius>=1).astype(np.uint8)
# imsave(SAVEPATH + 'large_vessels.tif', large_vessels*255)
# imsave(SAVEPATH + 'intermediate_vessels.tif', intermediate_vessels*255)
# imsave(SAVEPATH + 'small_vessels.tif', small_vessels*255)

# 2 continuous. local vessel radius distribution: large, intermediate, micro (%)
large_vessels_length_pixels = np.sum(large_vessels)
large_vessels_length_um = large_vessels_length_pixels * VOXEL_SIZE # um
large_vessels_length_m = large_vessels_length_um / 1000000 # m
large_vessels_percentage = large_vessels_length_m/total_length_m*100
print('local large vessel length: ' + str(round(large_vessels_length_m, 2)) + ' m (' + str(round(large_vessels_percentage, 2)) + '%)')
print('local large vessel distribution: ' + str(round(large_vessels_percentage, 2)) + ' %')

intermediate_vessels_length_pixels = np.sum(intermediate_vessels)
intermediate_vessels_length_um = intermediate_vessels_length_pixels * VOXEL_SIZE # um
intermediate_vessels_length_m = intermediate_vessels_length_um / 1000000 # m
intermediate_vessels_percentage = intermediate_vessels_length_m/total_length_m*100
print('local intermediate vessel length: ' + str(round(intermediate_vessels_length_m, 2)) + ' m (' + str(round(intermediate_vessels_percentage, 2)) + '%)')
print('local intermediate vessel distribution: ' + str(round(intermediate_vessels_percentage, 2)) + ' %')

small_vessels_length_pixels = np.sum(small_vessels)
small_vessels_length_um = small_vessels_length_pixels * VOXEL_SIZE # um
small_vessels_length_m = small_vessels_length_um / 1000000 # m
small_vessels_percentage = small_vessels_length_m/total_length_m*100
print('local small vessel length: ' + str(round(small_vessels_length_m, 2)) + ' m (' + str(round(small_vessels_percentage, 2)) + '%)')
print('local small vessel distribution: ' + str(round(small_vessels_percentage, 2)) + ' %')