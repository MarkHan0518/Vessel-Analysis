"""
This module contains the code for "multi-otsu thresholding" step needed for a vessel segmentation pipeline.
The user needs to define filepaths and parameters at the beginning of the main script.
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
import itk
from scipy.stats import norm


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

    chunk, position = data
    c_d, c_h, c_w = chunk.shape
    chunk_volume = c_d * c_h * c_w
    chunk = intensity_normalization(chunk, [0.5,3])
    chunk = edge_preserving_smoothing_3d(chunk)

    try:
        thresh = filters.threshold_multiotsu(chunk, classes=5)
        otsu_3d = (np.digitize(chunk, thresh)).astype(np.uint8)

        if (otsu_3d == 1).sum() >= chunk_volume*0.99:
            result = np.zeros(chunk.shape, dtype = np.uint8)
        else:
            result = otsu_3d

    except ValueError:
        result = np.zeros(chunk.shape, dtype = np.uint8)

    return (result, position)


def edge_preserving_smoothing_3d(
    struct_img: np.ndarray,
    numberOfIterations: int = 10,
    conductance: float = 1.2,
    timeStep: float = 0.0625,
    spacing: list = [1, 1, 1],
    ):
    """perform edge preserving smoothing on a 3D image

    Parameters:
    -------------
    struct_img: np.ndarray
        the image to be smoothed
    numberOfInterations: int
        how many smoothing iterations to perform. More iterations give more
        smoothing effect. Default is 10.
    timeStep: float
         the time step to be used for each iteration, important for numberical
         stability. Default is 0.0625 for 3D images. Do not suggest to change.
    spacing: List
        the spacing of voxels in three dimensions. Default is [1, 1, 1]

    Reference:
    -------------
    https://itk.org/Doxygen/html/classitk_1_1GradientAnisotropicDiffusionImageFilter.html
    https://github.com/AllenCell/aics-segmentation/blob/main/aicssegmentation/core/pre_processing_utils.py
    """

    itk_img = itk.GetImageFromArray(struct_img.astype(np.float32))

    # set spacing
    itk_img.SetSpacing(spacing)

    gradientAnisotropicDiffusionFilter = itk.GradientAnisotropicDiffusionImageFilter.New(itk_img)
    gradientAnisotropicDiffusionFilter.SetNumberOfIterations(numberOfIterations)
    gradientAnisotropicDiffusionFilter.SetTimeStep(timeStep)
    gradientAnisotropicDiffusionFilter.SetConductanceParameter(conductance)
    gradientAnisotropicDiffusionFilter.Update()

    itk_img_smooth = gradientAnisotropicDiffusionFilter.GetOutput()

    img_smooth_ag = itk.GetArrayFromImage(itk_img_smooth)

    return img_smooth_ag


def intensity_normalization(struct_img: np.ndarray, scaling_param: list):
    """Normalize the intensity of input image so that the value range is from 0 to 1.

    Parameters:
    ------------
    img: np.ndarray
        a 3d image
    scaling_param: List
        a list with only one value 0, i.e. [0]: Min-Max normlaizaiton,
            the max intensity of img will be mapped to 1 and min will
            be mapped to 0
        a list with a single positive integer v, e.g. [5000]: Min-Max normalization,
            but first any original intensity value > v will be considered as outlier
            and reset of min intensity of img. After the max will be mapped to 1
            and min will be mapped to 0
        a list with two float values [a, b], e.g. [1.5, 10.5]: Auto-contrast
            normalizaiton. First, mean and standard deviaion (std) of the original
            intensity in img are calculated. Next, the intensity is truncated into
            range [mean - a * std, mean + b * std], and then recaled to [0, 1]
        a list with four float values [a, b, c, d], e.g. [0.5, 15.5, 200, 4000]:
            Auto-contrast normalization. Similat to above case, but only intensity value
            between c and d will be used to calculated mean and std.
    
    Reference:
    -------------
    https://github.com/AllenCell/aics-segmentation/blob/main/aicssegmentation/core/pre_processing_utils.py
    """
    assert len(scaling_param) > 0

    if len(scaling_param) == 1:
        if scaling_param[0] < 1:
            print("intensity normalization: min-max normalization with NO absolute" + "intensity upper bound")
        else:
            print(f"intensity norm: min-max norm with upper bound {scaling_param[0]}")
            struct_img[struct_img > scaling_param[0]] = struct_img.min()
        strech_min = struct_img.min()
        strech_max = struct_img.max()
    elif len(scaling_param) == 2:
        m, s = norm.fit(struct_img.flat)
        strech_min = max(m - scaling_param[0] * s, struct_img.min())
        strech_max = min(m + scaling_param[1] * s, struct_img.max())
        struct_img[struct_img > strech_max] = strech_max
        struct_img[struct_img < strech_min] = strech_min
    elif len(scaling_param) == 4:
        img_valid = struct_img[np.logical_and(struct_img > scaling_param[2], struct_img < scaling_param[3])]
        assert (
            img_valid.size > 0
        ), f"Adjust intensity normalization parameters {scaling_param[2]} and {scaling_param[3]} to include the image with range {struct_img.min()}:{struct_img.max()}"  # noqa: E501
        m, s = norm.fit(img_valid.flat)
        strech_min = max(scaling_param[2] - scaling_param[0] * s, struct_img.min())
        strech_max = min(scaling_param[3] + scaling_param[1] * s, struct_img.max())
        struct_img[struct_img > strech_max] = strech_max
        struct_img[struct_img < strech_min] = strech_min
    assert (
        strech_min <= strech_max
    ), f"Please adjust intensity normalization parameters so that {strech_min}<={strech_max}"
    struct_img = (struct_img - strech_min + 1e-8) / (strech_max - strech_min + 1e-8)

    # print('intensity normalization completes')
    return struct_img


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
    CHUNK_SIZE = 100  # Size of the convolving box. A small chunk size increases noise and computational time, while a large chunk size reduces the detection of microvessels.
    STEP_SIZE = 50
    CH = 't00000/s00/0/cells'

    # Define hysteresis thresholding parameters
    HYST_LOW = 2  # Lower threshold for hysteresis thresholding (recommended value: 1)
    HYST_HIGH = 3  # Higher threshold for hysteresis thresholding (recommended value: 3)
    ###################################################


    ##### ---------- Preprocess ---------- #####
    with h5.File(DATADIR, 'r') as f:
        d_raw,h_raw,w_raw = f[CH].shape
        print(f[CH].shape)
        # d = 
        # h = 
        # w = 
        d = d_raw
        h = h_raw
        w = w_raw

        # chunk = f[CH][D_START:D_END, H_START:H_END, W_START:W_END]
        # chunk = intensity_normalization(chunk, [0.5,3])
        # chunk = edge_preserving_smoothing_3d(chunk)
        # imsave(SAVEPATH + 'original_data.tif', chunk)  
        # print('Done!')

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
    count_3d = np.zeros((d,h,w), dtype = np.uint8)
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
                data_list.append((chunk, (pos_d-D_START, pos_h-H_START, pos_w-W_START)))
                current_append += 1
    future_to_position = {executor.submit(multi_otsu, data): data for data in data_list}

    print('Start multiprocessing!')
    current = 1
    for future in concurrent.futures.as_completed(future_to_position):
        result, position = future.result()
        i, j, k = position
        print(position)
        print(result.shape)
        otsu_3d[i:min(d, i+CHUNK_SIZE), j:min(h, j+CHUNK_SIZE), k:min(w, k+CHUNK_SIZE)] += result
        count_3d[i:min(d, i+CHUNK_SIZE), j:min(h, j+CHUNK_SIZE), k:min(w, k+CHUNK_SIZE)] += np.ones(result.shape, dtype = np.uint8)
        print('Joined: ' + str(current) + '/' + str(total_step))
        current += 1
    
    slide_end = time.time()
    duration = slide_end - slide_start
    print('Overall duration of sliding: ' + str(duration) + ' s')
    imsave(SAVEPATH + 'otsu_3d.tif', otsu_3d)
    imsave(SAVEPATH + 'count_3d.tif', count_3d)
    #########################################################


    memory_info = psutil.virtual_memory()
    total_ram_used = memory_info.used
    print(f"Total RAM used for Multi-otsu Thresholding: {total_ram_used / 1024 ** 3} GB")