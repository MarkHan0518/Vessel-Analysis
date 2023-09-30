"""
This script calculates the volume of a binary mask and records it in a CSV file.
"""


import csv
import tifffile
import numpy as np


mask_name = ''
csv_file_path = 'mask_volume.csv'
binary_mask = tifffile.imread('' + mask_name + '.tif')
voxel_size = (2.68, 2.68, 2.68)

volume_um3 = np.sum(binary_mask) * np.prod(voxel_size)
volume_mm3 = volume_um3 / 1e6

print(f"Volume of the region: {volume_mm3} mm^3")

file_exists = False
try:
    with open(csv_file_path, 'r') as csvfile:
        file_exists = True
except FileNotFoundError:
    pass
headers = ["volume tag", "volume (mm3)"]
with open(csv_file_path, 'a', newline='') as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=headers)
    if not file_exists:
        csv_writer.writeheader()
    csv_writer.writerow({"volume tag": mask_name, "volume (mm3)": volume_mm3})
print(f'Data has been written to {csv_file_path}')