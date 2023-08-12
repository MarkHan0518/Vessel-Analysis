import numpy as np
from skimage import filters
import h5py as h5
import tifffile
import time
from tqdm import tqdm

save_path = '' # Filepath where the Multi_otsu.tif is saved and hyst.tif will be saved

otsu_3d = tifffile.imread(save_path + 'otsu_3d.tif')
count_3d = tifffile.imread(save_path + 'count_3d.tif')
d, w, h = otsu_3d.shape
# d = 50

start = time.time()

# Method1
data = (np.divide(otsu_3d, count_3d)).astype(np.uint8)
tifffile.imsave(save_path + 'divide.tif', data)

# Method2
# print('Start dividing...')
# curr = 1
# with tifffile.TiffWriter(save_path + 'divide.tif', bigtiff=True, append=True) as tif:
#     for i in tqdm(range(0, d)):
#         curr_otsu_3d = otsu_3d[i,:,:]
#         curr_count_3d = count_3d[i,:,:]
#         data = (np.divide(curr_otsu_3d, curr_count_3d)).astype(np.uint8)
#         tif.write(data)

end = time.time()
duration = end - start
print("Dividing time: " + str(round(duration, 2)) + ' s')
