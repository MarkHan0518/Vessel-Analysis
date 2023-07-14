"""
This module contains all the code needed for the vessel segmentation pipeline, including multi-otsu thresholding,
hysteresis thresholding, hole filling, and smoothing (in sequential order). The user needs to define filepaths and
parameters at the beginning of the main script.
"""


# # Set up env
# conda create -y -n vessel_seg -c conda-forge python=3.9
# conda activate vessel_seg
# pip install tifffile
# pip install h5py
# pip install scikit-image
# pip install matplotlib
# pip install tqdm
# pip install psutil


import psutil
import numpy as np
from tifffile import imsave
from skimage import filters
from math import ceil
import os
import h5py as h5
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from scipy.ndimage import median_filter
from scipy import ndimage as ndi


##### ---------- Functions ---------- #####
def i16_2_u16(stack):
    '''
    Parameters:
    - stack: a 3D numpy array of the data

    Function:
    Discard the signal counts that is below 0

    Return:
    A 3D numpy array of the processed data
    '''

    return np.uint16(np.clip(stack, 0, 65535))


def norm_uint8(stack, background_val, high_val):
    '''
    Parameters:
    - stack: a 3D numpy array of the data

    Function:
    Convert data from uint16 to uint8 with normalization

    Return:
    A 3D numpy array of the processed data
    '''

    stack_norm = np.clip(stack, background_val, high_val) - background_val
    stack_norm = stack_norm.astype(float) / (high_val-background_val) * 255
    stack_norm = stack_norm.astype(np.uint8)
    return stack_norm


def multi_otsu(data):
    '''
    Parameters:
    - data: a tuple contains a 3D numpy array of the data, a 3D numpy array of the downsampled data,
    and a tuple of x,y,z coordinates of the current region of interest

    Function:
    Preprocess the data by clipping with a high threshold (99.9% of the histogram of the downsampled dataset)
    and a low threshold (150); perform Multi-Otsu thresholding of downsampled dataset and apply it to the
    original dataset

    Return:
    A tuple contains a 3D numpy array of the Multi-Otsu processed data and a tuple of x,y,z coordinates of the
    current region of interest
    '''

    chunk, ds_chunk, position = data
    c_d, c_h, c_w = chunk.shape
    chunk_volume = c_d * c_h * c_w

    foreground_vals = np.sort(ds_chunk, axis=None)
    hist_clip = 0.999
    high_val = foreground_vals[int(np.round(len(foreground_vals)*hist_clip))]
    chunk = norm_uint8(i16_2_u16(chunk), 150, high_val)
    ds_chunk = norm_uint8(i16_2_u16(ds_chunk), 150, high_val)

    try:
        thresh = filters.threshold_multiotsu(ds_chunk, classes=5)
        print(thresh)
        otsu_3d = (np.digitize(chunk, thresh)).astype(np.uint8)

        if (otsu_3d == 1).sum() >= chunk_volume*0.99:
            result = np.zeros(chunk.shape, dtype = np.uint8)
        else:
            result = otsu_3d

    except ValueError:
        result = np.zeros(chunk.shape, dtype = np.uint8)

    return (result, position)


if __name__ == '__main__':
    ##### ---------- Define Filepaths ---------- #####
    DATADIR = ''  # Filepath of the HDF5 dataset. It must be an HDF5 file with at least 2 levels of downsampling (levels: 0, 1, 2)
    SAVEPATH = ''  # Filepath where the segmented masks will be saved
    #################################################


    ##### ---------- Define Parameters ---------- #####
    # Region of interest
    D_START = 0  # Start depth (0 if processing the entire dataset)
    D_END =  # End depth (use the maximum depth if processing the entire dataset)
    H_START = 0  # Start height (0 if processing the entire dataset)
    H_END =  # End height (use the maximum height if processing the entire dataset)
    W_START = 0  # Start width (0 if processing the entire dataset)
    W_END =  # End width (use the maximum width if processing the entire dataset)

    # Define chunks
    CHUNK_SIZE = 50  # Size of the convolving box. A small chunk size increases noise and computational time, while a large chunk size reduces the detection of microvessels.
    STEP_SIZE = CHUNK_SIZE
    DS_LEVEL = 2  # Level of the downsampled dataset to be used for regional thresholding (recommended value: 2)
    DS_FACTOR = 2 * DS_LEVEL
    CH = 't00000/s00/0/cells'
    CH_DS = 't00000/s00/' + str(DS_LEVEL) + '/cells'

    # Define the region for Multi-OTSU thresholding
    REGION_SIZE = 150  # Size of the region surrounding the convolving box used for regional thresholding (3 times the chunk size)
    FRONT_REGION = int(REGION_SIZE / 2 - int(STEP_SIZE / 2))
    BACK_REGION = REGION_SIZE - FRONT_REGION

    # Define hysteresis thresholding parameters
    HYST_LOW = 1  # Lower threshold for hysteresis thresholding (recommended value: 1)
    HYST_HIGH = 3  # Higher threshold for hysteresis thresholding (recommended value: 3)
    ###################################################


    ##### ---------- Preprocess ---------- #####
    with h5.File(DATADIR, 'r') as f:
        d_raw,h_raw,w_raw = f[CH].shape
        # d = 500
        # h = 500
        # w = 500
        d = d_raw
        h = h_raw
        w = w_raw
        print('Dataset located!')
        print('Region of interest shape:', d, h, w)

    if not os.path.exists(SAVEPATH):
        os.makedirs(SAVEPATH)

    num_pos_h = ceil(h/STEP_SIZE) 
    num_pos_w = ceil(w/STEP_SIZE)
    num_pos_d = ceil(d/STEP_SIZE)
    print('Num of steps in 3 dims:', num_pos_h, num_pos_w, num_pos_d)
    ############################################


    ##### ---------- Multi-otsu Thresholding ---------- #####
    current_step = 1
    total_step = (num_pos_h) * (num_pos_w) * (num_pos_d)
    otsu_3d = np.zeros((d,h,w), dtype = np.uint8)
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    data_list = []
    executor = ProcessPoolExecutor(max_workers=mp.cpu_count())
    num_processes = mp.cpu_count()
    print('Num of CPU available: ', num_processes)

    print('Preparing for multiprocessing!')
    slide_start = time.time()
    current_append = 1
    for pos_h in range(H_START, H_START+num_pos_h*STEP_SIZE, STEP_SIZE):
        for pos_w in range(W_START, W_START+num_pos_w*STEP_SIZE, STEP_SIZE):
            for pos_d in range(D_START, D_START+num_pos_d*STEP_SIZE, STEP_SIZE):
                print('Append: ' + str(current_append) + '/' + str(total_step))
                with h5.File(DATADIR, 'r') as f:
                    chunk = f[CH][pos_d:min(D_END, pos_d+CHUNK_SIZE), pos_h:min(H_END, pos_h+CHUNK_SIZE), pos_w:min(W_END, pos_w+CHUNK_SIZE)]
                    ds_chunk = f[CH_DS][max(0, pos_d-FRONT_REGION)//DS_FACTOR:min(d_raw, pos_d+BACK_REGION)//DS_FACTOR, max(0, pos_h-FRONT_REGION)//DS_FACTOR:min(h_raw, pos_h+BACK_REGION)//DS_FACTOR, max(0, pos_w-FRONT_REGION)//DS_FACTOR:min(w_raw, pos_w+BACK_REGION)//DS_FACTOR]
                data_list.append((chunk, ds_chunk, (pos_d-D_START, pos_h-H_START, pos_w-W_START)))
                current_append += 1
    future_to_position = {executor.submit(multi_otsu, data): data for data in data_list}

    print('Start multiprocessing!')
    current = 1
    for future in concurrent.futures.as_completed(future_to_position):
        result, position = future.result()
        i, j, k = position
        otsu_3d[i:min(d, i+CHUNK_SIZE), j:min(h, j+CHUNK_SIZE), k:min(w, k+CHUNK_SIZE)] = result
        print('Joined: ' + str(current) + '/' + str(total_step))
        current += 1
    
    slide_end = time.time()
    duration = slide_end - slide_start
    print('Overall duration of sliding: ' + str(duration) + ' s')
    imsave(SAVEPATH + 'Multi_otsu.tif', otsu_3d)
    #########################################################


    memory_info = psutil.virtual_memory()
    total_ram_used = memory_info.used
    print(f"Total RAM used for Multi-otsu Thresholding: {total_ram_used / 1024 ** 3} GB")


    ##### ---------- Hysteresis Thresholding ---------- #####
    # For a large dataset, the computer may run out of memory up to this point. Do it in a separate python task!!!
    hyst = filters.apply_hysteresis_threshold(otsu_3d, HYST_LOW, HYST_HIGH).astype(np.uint8)*255
    imsave(SAVEPATH + 'hyst.tif', hyst)
    #########################################################


    ##### ---------- Hole Filling ---------- #####
    # For a large dataset, the computer may run out of memory up to this point. Do it in a separate python task!!!
    hole_filling = ndi.binary_fill_holes(hyst).astype(np.uint8)
    # fill_hole = (hole_filling*255).astype(np.uint8)
    imsave(SAVEPATH + 'hole_filling.tif', hole_filling)
    ##############################################


    ##### ---------- Smoothing ---------- #####
    # For a large dataset, the computer may run out of memory up to this point. Do it in a separate python task!!!
    size = 3
    smoothing = median_filter(hole_filling, size=size)
    imsave(SAVEPATH + 'smoothing.tif', smoothing)
    ###########################################