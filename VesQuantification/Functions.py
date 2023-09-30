"""
This module contains useful functions for feature extraction.
"""


import csv
import math
import numpy as np
import networkx as nx
from scipy import ndimage as ndi
from skimage.morphology import skeletonize_3d


def get_length(path: list, dimensions: list) -> float:
    """
    Find length of a path as distance between nodes in it

    Parameters
    ----------
    path : list
        List of nodes in the path.
    dimensions : list
        List with pixel dimensions in the desired unit (e.g., microns).
        3D: [z, y, x]   2D: [y, x]

    Returns
    ----------
    length : float
        Length of path.
    """
    length = 0
    for index, item in enumerate(path):
        if index + 1 != len(path):
            item2 = path[index + 1]
        vect = [j - i for i, j in zip(item, item2)]
        vect = [a * b for a, b in zip(vect, dimensions)]
        length += np.linalg.norm(vect)
    return length


def get_radius(radius_matrix: np.ndarray, segment: list) -> float:
    """
    Calculate the average radius of a segment.

    Parameters
    ----------
    radius_matrix : np.ndarray
        A matrix of radii associated with the vertices of the network graph.

    segment : list
        A list of nodes (vertices) that belong to the segment for which you want to calculate the average radius.

    Returns
    ----------
    float
        The average radius of the segment, calculated as the sum of radii divided by the number of vertices in the segment.
    """
    sum_radius = 0
    for vertex in segment:
        sum_radius += radius_matrix[vertex]
    return sum_radius / len(segment)


def get_volume_cylinder(radius: float, seg_length: float) -> float:
    """
    Calculate the volume of a cylinder.

    Parameters
    ----------
    radius : float
        The radius of the cylinder.

    seg_length : float
        The length (height) of the cylinder.

    Returns
    ----------
    float
        The volume of the cylinder, calculated as π * radius^2 * seg_length.
    """
    return math.pi * radius ** 2 * seg_length


def extract_centerlines(bin_image: np.ndarray) -> np.ndarray:
    """
    Compute the centerlines of a 3D binary image.

    Parameters
    ----------
    bin_image : np.ndarray
        3D binary mask.

    Returns
    ----------
    np.ndarray
        An ndarray representing the thinned image.
    """
    centerlines = skeletonize_3d(bin_image)
    centerlines.astype(dtype='uint8', copy=False)
    return centerlines


def extract_radius(bin_image: np.ndarray, centerlines: np.ndarray, pixel_dims: list) -> np.ndarray:
    """
    Exact Euclidean distance transform and convolve it with the centerlines.

    Parameters
    ----------
    bin_image : np.ndarray
        3D binary mask.
    centerlines : np.ndarray
        Skeleton of the 3D binary mask.
    pixel_dims : list
        List of pixel dimensions in the desired unit (e.g., microns).

    Returns
    ----------
    np.ndarray
        An ndarray representing the Euclidean distance, where each pixel value represents the distance.

    Important
    ----------
    The Euclidean distance must be multiplied by the voxel size to obtain the actual radius of the vessel.
    """
    dist_transf = ndi.distance_transform_edt(bin_image, sampling=pixel_dims)
    radius_matrix = dist_transf * centerlines
    return radius_matrix


def nx_graph_branch_prune(nx_graph: nx.Graph, node_degree_dict: dict, radius_matrix: np.ndarray, pixel_dims: list, prunScale: float = 1.5) -> tuple:
    """
    Branches are removed when |ep - bp|^2 <= s * |f - bp|^2
    where ep = end point, bp = branch point, s = scaling factor, f = closest boundary point

    Parameters
    ----------
    nx_graph : nx.Graph
        NetworkX graph representing the vessel structure.
    node_degree_dict : dict
        Dictionary mapping nodes to their degrees.
    radius_matrix : np.ndarray
        An ndarray containing radius information.
    pixel_dims : list
        List with pixel dimensions in the desired unit (e.g., microns).
    prunScale : float, optional
        Scaling factor for pruning, by default 1.5.

    Returns
    ----------
    tuple
        A tuple containing:
        - The pruned NetworkX graph.
        - The number of branches removed.

    References
    ----------
    - VesselExpress (source of the code):
      https://github.com/RUB-Bioinf/VesselExpress/blob/master/VesselExpress/modules/graph.py#L250
    - Original concept:
      Montero, M. L., & Lang, J. (2012). Skeleton pruning by contour approximation and the integer medial axis transform.
      Computers & Graphics, 36(5), 477-487. https://www.sciencedirect.com/science/article/pii/S0097849312000684
    """
    endPtsList = [k for (k, v) in node_degree_dict.items() if v == 1]
    branchesToRemove = []
    for endPt in endPtsList:
        visited, stack = set(), [endPt]
        branch = []
        while stack:
            vertex = stack.pop()
            vertex = tuple(vertex)
            if vertex not in visited:
                visited.add(vertex)
                branch.append(vertex)
                neighbors = [n for n in nx_graph.neighbors(vertex)]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        stack.append(neighbor)
                if len(neighbors) > 2:
                    if get_length(branch, pixel_dims) <= radius_matrix[vertex] * prunScale:
                        branchesToRemove.append(branch)
                    break
    for branch in branchesToRemove:
        for node in branch[:-1]:
            nx_graph.remove_node(node)

    return nx_graph, len(branchesToRemove)


def connected_component_subgraphs(G):
    """
    Generate connected component subgraphs.

    Parameters
    ----------
    G : NetworkX graph
        The graph in which to find connected components.

    Yields
    ----------
    subgraph : NetworkX graph
        A subgraph containing a connected component of the input graph.

    """
    for c in nx.connected_components(G):
        yield G.subgraph(c)


def get_segment(vertex, adjacency_dict, pred_dict):
    """
    Find the segment of a branch or end node by iterating over its predecessors.

    Parameters
    ----------
    vertex : list of node coordinates
        Coordinates of the starting node.
        3D: [z, y, x]
        2D: [y, x]

    adjacency_dict : dict
        A dictionary representing the adjacency of nodes in the graph.

    pred_dict : dict
        A dictionary representing predecessors of nodes in the graph.

    Returns
    ----------
    segment_list : list of nodes in the segment
        A list of nodes that form the segment.

    References
    ----------
    - VesselExpress (source of the code):
      https://github.com/RUB-Bioinf/VesselExpress/blob/master/VesselExpress/modules/filament.py#L189
    """
    segment_list = [vertex]
    while True:
        vertex = pred_dict.get(vertex)
        if vertex is None:  # may happen due to postprocessing removing predecessors of old branching points
            return None
        segment_list.insert(0, vertex)
        if len(adjacency_dict[vertex]) == 1 or len(adjacency_dict[vertex]) > 2:
            break
    return segment_list


def set_seg_stats(segment, radius_matrix, pixel_dims) -> dict:
    """
    Calculate statistics for a segment and return them in a dictionary.

    Parameters
    ----------
    segment : list
        List of nodes in the segment.

    radius_matrix : ndarray
        A matrix containing radius information for each pixel.

    pixel_dims : list
        List with pixel dimensions in the desired unit (e.g., microns).
        3D: [z, y, x]
        2D: [y, x]

    Returns
    ----------
    segments_info : dict
        A dictionary containing the following segment statistics:
        - 'diameter (um)': Diameter of the segment in microns.
        - 'tortuosity': Tortuosity of the segment.
        - 'length (um)': Length of the segment in microns.
        - 'volume (um3)': Volume of the segment in cubic microns.
    """
    segments_info = {}

    # calculate segment mean radius
    seg_radius = get_radius(radius_matrix, segment)

    # calculate segment displacement
    vect = [j - i for i, j in zip(segment[0], segment[len(segment) - 1])]
    vect = [a * b for a, b in zip(vect, pixel_dims)]  # multiply pixel length with pixel dimension
    seg_displacement = np.linalg.norm(vect)

    # calculate segment length
    seg_length = get_length(segment, pixel_dims)

    # calculate segment volume
    seg_volume = get_volume_cylinder(seg_radius, seg_length)

    # fill dictionary for csv file containing all segment statistics
    segments_info['diameter (um)'] = seg_radius * 2
    segments_info['tortuosity'] = seg_length / seg_displacement
    segments_info['length (um)'] = seg_length
    segments_info['volume (um3)'] = seg_volume

    return segments_info


def reconstruct_centerlines(centerlines, nx_graph) -> np.ndarray:
    """
    Reconstruct centerlines from a networkx graph.

    Parameters
    ----------
    centerlines : np.ndarray
        Thinned binary image representing centerlines.

    nx_graph : networkx.Graph
        Networkx graph representing the vessel structure.

    Returns
    ----------
    reconstructed_centerlines : np.ndarray
        Reconstructed centerlines as a binary image.
    """
    reconstructed_centerlines = np.zeros(centerlines.shape, dtype='uint8')
    for ind in list(nx_graph.nodes):
        reconstructed_centerlines[ind] = 1
    return reconstructed_centerlines


def double_dict_to_csv(double_dict, csv_file_path) -> None:
    """
    Write data from a nested dictionary to a CSV file.

    Parameters
    ----------
    double_dict : dict
        A nested dictionary where the outer keys represent the primary data labels.

    csv_file_path : str
        The path to the CSV file where the data will be written.

    Returns
    -------
    None
    """
    column_headers = ['Key'] + list(next(iter(double_dict.values())).keys())
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(column_headers)
        for key, nested_dict in double_dict.items():
            row_data = [key] + [nested_dict[column] for column in column_headers[1:]]
            csv_writer.writerow(row_data)
    print(f'Data has been written to {csv_file_path}')


def record_branch_pts(save_path: str, sample_tag: str, br_pts_num: int) -> None:
    """
    Record the number of branch points to a CSV file.

    Parameters
    ----------
    save_path : str
        The directory path where the CSV file will be saved.

    sample_tag : str
        The tag or identifier for the sample.

    br_pts_num : int
        The total number of branch points to record.

    Returns
    -------
    None
    """
    csv_file_path = save_path + 'branch_point.csv'
    # Check if the file already exists or needs to be created
    file_exists = False
    try:
        with open(csv_file_path, 'r') as csvfile:
            file_exists = True
    except FileNotFoundError:
        pass
    headers = ["sample tag", "total # branch points"]
    with open(csv_file_path, 'a', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=headers)
        if not file_exists:
            csv_writer.writeheader()
        csv_writer.writerow({"sample tag": sample_tag, "total # branch points": br_pts_num})
    print(f'Data has been written to {csv_file_path}')