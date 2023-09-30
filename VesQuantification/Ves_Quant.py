"""
This script performs vessel analysis using a combination of techniques.
It extracts centerlines, radii, and performs graph-based analysis on vessel structures.
"""


import os
import json
import pickle
import functions
import numpy as np
import networkx as nx
from tqdm import tqdm 
import graph_generator
from tifffile import imsave, imread


# Set Parameters
save_path = '/Users/qinghuahan/Desktop/LiuLabProjects/3D_Vessel_Segmentation/SegentationTest/largevessels/'
sample_tag = 'Bx-xxx-xxx-segments_2'
pixel_dims = [2.68, 2.68, 2.68]


bin_image = imread(save_path + 'smoothing.tif')
# Initiate Parameters
info_dict = {
    'pruning count': 0,
    'filaments count': 0,
    'segments count': 0,
}


# Centerline Extraction
centerlines = functions.extract_centerlines(bin_image)
# imsave(save_path + 'centerlines.tif', centerlines)
# centerlines = imread(save_path + 'centerlines.tif')
print('Successfully extract centerlines!')


# Radius Extraction
radius_matrix = functions.extract_radius(bin_image, centerlines, pixel_dims)
# imsave(save_path + 'radius_matrix.tif', radius_matrix)
# radius_matrix = imread(save_path + 'radius_matrix.tif')
print('Successfully extract radius!')


# Networkx Graph Generation

# Create one if you do not have
nx_graph = graph_generator.get_networkx_graph_from_array(centerlines)
# with open(save_path + "nx_graph.graphml", 'wb') as file:
#     pickle.dump(nx_graph, file)
# with open(save_path + "nx_graph.graphml", 'rb') as file:
#     nx_graph = pickle.load(file)
print('Successfully generate graphs!')

node_degree_dict = dict(nx.degree(nx_graph))
# with open(save_path + 'node_degree_dict.pkl', 'wb') as file:
#     pickle.dump(node_degree_dict, file)
# with open(save_path + 'node_degree_dict.pkl', 'rb') as file:
#     node_degree_dict = pickle.load(file)
print('Successfully generate nodes dictionaries!')


# Networkx Graph Post-processing: Branch Pruning
nx_graph, info_dict['pruning count'] = functions.nx_graph_branch_prune(nx_graph, node_degree_dict, radius_matrix, pixel_dims, prunScale=1.5)
print("pruned {} branches".format(info_dict['pruning count']))
# with open(save_path + "pruned_nx_graph.graphml", 'wb') as file:
#     pickle.dump(nx_graph, file)
# with open(save_path + "pruned_nx_graph.graphml", 'rb') as file:
#     nx_graph = pickle.load(file)
print('Successfully prune graphs!')


# Define Filaments
filaments = list(functions.connected_component_subgraphs(nx_graph))
info_dict['filaments count'] = len(filaments)
print("found {} filaments".format(info_dict['filaments count']))

# with open(save_path + "nx_subgraphs.pkl", 'wb') as file:
#     pickle.dump(filaments, file)
# with open(save_path + "nx_subgraphs.pkl", 'rb') as file:
#     filaments = pickle.load(file)
print('Successfully define filaments!')


# Define Segments
filaments_dict = {}
segments_dict = {}
br_pts_dict = {}
end_pts_list = []
progress_bar = tqdm(enumerate(filaments), total=len(filaments), desc="Processing Filaments")
for ith_filament, ith_filament_subgraph in progress_bar:
    ith_filament_node_degree_dict = dict(nx.degree(ith_filament_subgraph))
    end_points = [k for (k, v) in ith_filament_node_degree_dict.items() if v == 1]
    if end_points:
        start = end_points[0]  # take random end point as beginning
        adjacency_dict = nx.to_dict_of_lists(ith_filament_subgraph)
    pred_dict = {}
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            for neighbor in adjacency_dict[vertex]:
                if neighbor not in visited:
                    pred_dict[neighbor] = vertex
                    stack.append(neighbor)
                elif neighbor in visited and not pred_dict[vertex] == neighbor:  # cycle found
                    if len(adjacency_dict[neighbor]) > 2:  # neighbor is branch point
                        old_pred = pred_dict[neighbor]
                        pred_dict[neighbor] = vertex   # change predecessor to get segment of cycle
                        segment = functions.get_segment(vertex, adjacency_dict, pred_dict)
                        pred_dict[neighbor] = old_pred  # change back to old predecessor
                        segments_dict[segment[0], segment[len(segment) - 1]] = functions.set_seg_stats(segment, radius_matrix, pixel_dims)
            if len(adjacency_dict[vertex]) == 1:    # end point found
                end_pts_list.append(vertex)
                if vertex != start:
                    segment = functions.get_segment(vertex, adjacency_dict, pred_dict)
                    segments_dict[segment[0], segment[len(segment) - 1]] = functions.set_seg_stats(segment, radius_matrix, pixel_dims)
            elif len(adjacency_dict[vertex]) > 2:   # branch point found
                br_pts_dict[vertex] = len(adjacency_dict[vertex])
                segment = functions.get_segment(vertex, adjacency_dict, pred_dict)
                segments_dict[segment[0], segment[len(segment) - 1]] = functions.set_seg_stats(segment, radius_matrix, pixel_dims)
    filaments_dict[ith_filament] = segments_dict

print("Total number of branch points: " + str(len(br_pts_dict)))
print("Total number of end points: " + str(len(end_pts_list)))
# with open(save_path + "segments_dict.pkl", 'wb') as file:
#     pickle.dump(filaments, file)
# with open(save_path + "segments_dict.pkl", 'rb') as file:
#     segments_dict = pickle.load(file)
print('Successfully define and analyze segments!')


# Reconstruct new centerlines
reconstruct_centerlines = functions.reconstruct_centerlines(centerlines, nx_graph)
# imsave(save_path + 'reconstruct_centerlines.tif', reconstruct_centerlines)
# centerlines = imread(save_path + 'reconstruct_centerlines.tif')
print('Successfully reconstruct new centerlines!')


# Save data as csv
functions.double_dict_to_csv(segments_dict, save_path + sample_tag + '.csv')
functions.record_branch_pts(save_path, sample_tag, len(br_pts_dict))