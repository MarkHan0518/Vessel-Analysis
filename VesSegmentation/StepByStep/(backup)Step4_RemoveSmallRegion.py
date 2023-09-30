import time
import numpy as np
from skimage import measure
from skimage import morphology
from tifffile import imsave, imread
import multiprocessing

def remove_small_region_chunk(chunk, small_region_size=40):
    labels = measure.label(chunk)
    properties = measure.regionprops(labels)
    num_removed = 0  # Initialize a counter for removed objects
    
    for i in range(len(properties)):
        if properties[i].area < small_region_size:
            temp = ~(labels == properties[i].label)
            temp = temp.astype(np.uint8)
            chunk = chunk * temp
            num_removed += 1  # Increment the counter
    
    return chunk, num_removed  # Return both the processed chunk and the count of removed objects

def remove_small_region(image, num_processes=4):
    chunk_size = image.shape[0] // num_processes
    chunks = [image[i:i+chunk_size] for i in range(0, image.shape[0], chunk_size)]
    num_objects_removed = 0  # Initialize a total count of removed objects
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(remove_small_region_chunk, chunks)
    
    processed_chunks, num_removed_list = zip(*results)  # Unpack the results
    num_objects_removed = sum(num_removed_list)  # Sum the counts of removed objects
    
    return np.vstack(processed_chunks), num_objects_removed  # Return both the processed image and the total count of removed objects

if __name__ == "__main__":
    ##### ---------- Define Parameters ---------- #####
    save_path = '/Users/qinghuahan/Desktop/LiuLabProjects/3D_Vessel_Segmentation/SegentationTest/largevessels/' # Filepath where the hole_filling.tif is saved and smoothing.tif will be saved
    ###################################################

    hole_filling = imread(save_path + 'hole_filling.tif')
    hole_filling = hole_filling.astype(bool)
    
    # Count objects before removal
    labels_before_removal = measure.label(hole_filling)
    num_objects_before_removal = len(measure.regionprops(labels_before_removal))
    print("Number of objects before removal: " + str(num_objects_before_removal))
    
    remove_start = time.time()
    remove_region, num_objects_removed = remove_small_region(hole_filling)
    remove_region = remove_region.astype(np.uint8)  # Convert the processed image to uint8
    remove_end = time.time()
    
    # Count objects after removal
    labels_after_removal = measure.label(remove_region)
    properties = measure.regionprops(labels_after_removal)
    num_objects_after_removal = len(measure.regionprops(labels_after_removal))
    # for props in properties:
    #     print(f"Label: {props.label}, Area: {props.area}")
    # print("Number of objects after removal: " + str(num_objects_after_removal))
    
    imsave(save_path + 'remove_region.tif', remove_region)
    remove_duration = remove_end - remove_start
    print("Remove small regions time: " + str(round(remove_duration)) + ' s')
    print("Total objects removed: " + str(num_objects_removed))