"""
This module contains the code for "multi-otsu thresholding" step needed for a vessel segmentation pipeline.
The user needs to define filepaths and parameters at the beginning of the main script.
"""


# # All the packages needed to set up the environment, copy them into the terminal
# conda create -y -n vessel_seg -c conda-forge python=3.9
# conda activate vessel_seg
# pip install tqdm
# pip install zarr
# pip install h5py
# pip install psutil
# pip install tifffile
# pip install matplotlib
# pip install scikit-image


import os
import itk
import zarr
import time
import psutil
import h5py as h5
import numpy as np
from tqdm import tqdm
from math import ceil
import concurrent.futures
from tifffile import imsave, imread
from scipy.stats import norm
import multiprocessing as mp
from scipy import ndimage as ndi
from skimage import filters, measure
from scipy.ndimage import median_filter
from concurrent.futures import ProcessPoolExecutor


##### ---------- Functions ---------- #####
def uint16_converter(stack: np.ndarray) -> np.ndarray:
    """Clip the intensity of an input 3D image to ensure values are in the range of 0 to 65535.

    Parameters:
    ----------
    stack: np.ndarray
        The input 3D image as a NumPy array.

    Returns:
    -------
    np.ndarray
        A new NumPy array with intensity values clipped to the range [0, 65535].
    """
    return np.uint16(np.clip(stack, 0, 65535))


def remove_noise(stack: np.ndarray, background_val: int) -> np.ndarray:
    """Remove noise from a 3D image by setting values below a specified threshold to 0.

    Parameters:
    ----------
    stack : np.ndarray
        The input 3D image as a NumPy array.
    background_val : int
        Intensity threshold below which noise values are set to 0.

    Returns:
    ----------
    np.ndarray
        The input NumPy array with noise values set to 0 below the specified threshold.
    """
    stack[stack <= background_val] = 0
    return stack


def edge_preserving_smoothing_3d(
    struct_img: np.ndarray,
    numberOfIterations: int = 10,
    conductance: float = 1.2,
    timeStep: float = 0.0625,
    spacing: list = [1, 1, 1],
    ) -> np.ndarray:
    """Perform edge-preserving smoothing on a 3D image using gradient anisotropic diffusion.

    This function applies edge-preserving smoothing to a 3D image using gradient anisotropic diffusion.
    The input image is smoothed while preserving sharp edges.

    Parameters:
    ----------
    struct_img : np.ndarray
        The 3D image to be smoothed.
    numberOfIterations : int, optional
        Number of smoothing iterations to perform. More iterations result in a stronger smoothing effect.
        Default is 10.
    conductance : float, optional
        Conductance parameter for diffusion. A higher value preserves edges more effectively.
        Default is 1.2.
    timeStep : float, optional
        Time step used for each iteration. Important for numerical stability. Default is 0.0625 for 3D images.
    spacing : list, optional
        Spacing of voxels in three dimensions. Default is [1, 1, 1].
        
    Returns:
    ----------
    np.ndarray
        A 3D numpy array representing the edge-preserving smoothed image.

    References:
    ----------
    - Gradient Anisotropic Diffusion Image Filter:
      https://itk.org/Doxygen/html/classitk_1_1GradientAnisotropicDiffusionImageFilter.html
    - AICS Segmentation Library (source of the code):
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


def intensity_normalization(struct_img: np.ndarray, scaling_param: list) -> np.ndarray:
    """Normalize the intensity of a 3D input image to the range [0, 1] using various methods.

    This function performs intensity normalization on a 3D image. Different methods of normalization
    are supported based on the given `scaling_param` list.

    Parameters:
    ----------
    struct_img : np.ndarray
        The 3D image to be intensity-normalized.
    scaling_param : list
        A list specifying the intensity normalization method:
        - [0]: Min-Max normalization, mapping the max intensity to 1 and min to 0.
        - [v]: Min-Max normalization, setting values above v to min before mapping.
        - [a, b]: Auto-contrast normalization. Truncates intensity to [mean - a * std, mean + b * std]
          and then scales to [0, 1].
        - [a, b, c, d]: Auto-contrast normalization within range [c, d].

    Returns:
    -------
    np.ndarray
        A 3D numpy array representing the intensity-normalized image.

    Reference:
    ----------
    AICS Segmentation Library (source of the code):
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
    return struct_img


def multi_otsu_processing(data: tuple) -> tuple:
    """Apply Multi-Otsu thresholding to a 3D dataset after preprocessing and filtering.

    This function takes a tuple containing a 3D numpy array of the data, a tuple of x, y, z coordinates of
    the current region of interest, and an integer value specifying a predefined background noise threshold.
    It preprocesses the data by performing the following steps:
    1. Remove noise by setting values below the specified noise threshold to 0.
    2. Apply intensity normalization to the chunk using a specific range.
    3. Perform edge-preserving 3D smoothing on the chunk.

    Multi-Otsu thresholding is then applied to the preprocessed chunk with a target of 5 classes. The resulting
    thresholded data is returned as a 3D numpy array. If the thresholding result has a dominant class occupying
    more than 99% of the chunk, the result is an array of zeros.

    Parameters:
    ----------
    data : tuple
        A tuple containing the following elements:
        - A 3D numpy array of the data.
        - A tuple of x, y, z coordinates of the current region of interest.
        - An integer representing a predefined background noise threshold.

    Returns:
    ----------
    tuple
        A tuple containing the following elements:
        - A 3D numpy array of the Multi-Otsu processed data.
        - A tuple of x, y, z coordinates of the current region of interest.
    """
    chunk, position, noise_thrsh = data
    c_d, c_h, c_w = chunk.shape
    chunk_volume = c_d * c_h * c_w
    chunk = intensity_normalization(remove_noise(uint16_converter(chunk), noise_thrsh), [0.5,3])
    chunk = edge_preserving_smoothing_3d(chunk)

    try:
        thresh = filters.threshold_multiotsu(chunk, classes=5)
        print("Thresholds of current chunk: " + str(thresh))
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
    DATADIR = ''  # Specify the path to the dataset.
    SAVEPATH = ''  # Specify the path where segmented masks will be saved.
    #################################################


    ##### ---------- Define Parameters ---------- #####
    # Define chunks
    BACKGROUND_NOISE_THRSH = 250  # Signal intensity of the no-data region
    CHUNK_SIZE = 100  # Size of the convolving box
    STEP_SIZE = 70  # Overlapping region equals CHUNK_SIZE - STEP_SIZE
    file_format = 'tiff' # or "hdf5"
    CH = 't00000/s00/0/cells'  # Path to the channel within the HDF5 file
    ###################################################


    ##### ---------- Preprocessing ---------- #####
    if file_format == 'tiff':
        data = imread(DATADIR, aszarr=True)
        data_zarr = zarr.open(data, mode='r')
        d_raw, h_raw, w_raw = data_zarr.shape
        print('Dataset located!')
        print('Raw dataset shape: ', data_zarr.shape)

    elif file_format == 'hdf5':
        with h5.File(DATADIR, 'r') as f:
            d_raw, h_raw, w_raw = f[CH].shape
            print('Dataset located!')
            print('Raw dataset shape: ', f[CH].shape)

    if not os.path.exists(SAVEPATH):
        os.makedirs(SAVEPATH)
    
    # Region of interest
    D_START = 0  # Start depth (0 if processing the entire dataset)
    D_END = d_raw  # End depth (use the maximum depth if processing the entire dataset)
    H_START = 0  # Start height (0 if processing the entire dataset)
    H_END = h_raw # End height (use the maximum height if processing the entire dataset)
    W_START = 0  # Start width (0 if processing the entire dataset)
    W_END = w_raw # End width (use the maximum width if processing the entire dataset)

    # Uncomment below for obtaining raw tiff of ROI
    # data_ROI = data_zarr[D_START:D_END, H_START:H_END, W_START:W_END]
    # imsave(os.path.join(SAVEPATH, 'original_data.tif'), data_ROI)
    # print('Raw Data Obtained!')

    d_size = D_END-D_START
    h_size = H_END-H_START
    w_size = W_END-W_START
    num_pos_d = ceil(d_size / STEP_SIZE)
    num_pos_h = ceil(h_size / STEP_SIZE)
    num_pos_w = ceil(w_size / STEP_SIZE)
    print('Num of steps in ROI:', num_pos_h, num_pos_w, num_pos_d)
    ############################################


    ##### ---------- Multi-otsu Thresholding ---------- #####
    current_step = 1
    total_step = num_pos_h * num_pos_w * num_pos_d
    otsu_3d = np.zeros((d_size, h_size, w_size), dtype=np.uint8)
    count_3d = np.zeros((d_size, h_size, w_size), dtype=np.uint8)
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    data_list = []
    executor = ProcessPoolExecutor(max_workers=mp.cpu_count())
    num_processes = mp.cpu_count()
    print('Num of CPU available: ', num_processes)

    print('Preparing for multiprocessing!')
    slide_start = time.time()
    current_append = 1
    for pos_h in range(H_START, H_START + num_pos_h * STEP_SIZE, STEP_SIZE):
        for pos_w in range(W_START, W_START + num_pos_w * STEP_SIZE, STEP_SIZE):
            for pos_d in range(D_START, D_START + num_pos_d * STEP_SIZE, STEP_SIZE):
                print('Append: ' + str(current_append) + '/' + str(total_step))
                if file_format == 'tiff':
                    chunk = data_zarr[pos_d:min(D_END, pos_d + CHUNK_SIZE), pos_h:min(H_END, pos_h + CHUNK_SIZE),
                                    pos_w:min(W_END, pos_w + CHUNK_SIZE)]
                elif file_format == 'hdf5':
                    with h5.File(DATADIR, 'r') as f:
                        chunk = f[CH][pos_d:min(D_END, pos_d + CHUNK_SIZE), pos_h:min(H_END, pos_h + CHUNK_SIZE),
                                    pos_w:min(W_END, pos_w + CHUNK_SIZE)]
                data_list.append((chunk, (pos_d - D_START, pos_h - H_START, pos_w - W_START), BACKGROUND_NOISE_THRSH))
                current_append += 1

    future_to_position = {executor.submit(multi_otsu_processing, data): data for data in data_list}

    print('Start multiprocessing!')
    current = 1
    for future in concurrent.futures.as_completed(future_to_position):
        result, position = future.result()
        i, j, k = position
        otsu_3d[i:min(d_size, i + CHUNK_SIZE), j:min(h_size, j + CHUNK_SIZE), k:min(w_size, k + CHUNK_SIZE)] += result
        count_3d[i:min(d_size, i + CHUNK_SIZE), j:min(h_size, j + CHUNK_SIZE), k:min(w_size, k + CHUNK_SIZE)] += np.ones(
            result.shape, dtype=np.uint8)
        print('Joined: ' + str(current) + '/' + str(total_step))
        current += 1

    slide_end = time.time()
    duration = slide_end - slide_start
    print('Overall duration of sliding: ' + str(duration) + ' s')
    imsave(os.path.join(SAVEPATH, 'otsu_3d.tif'), otsu_3d)
    imsave(os.path.join(SAVEPATH, 'count_3d.tif'), count_3d)
    #########################################################


    memory_info = psutil.virtual_memory()
    total_ram_used = memory_info.used
    print(f"Total RAM used for Multi-otsu Thresholding: {total_ram_used / 1024 ** 3} GB")