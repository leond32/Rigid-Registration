# A noise applied to all points. Points within the segmentation are only linearly transformed, 
# while other points are linearly and non-linearly transormed
# -> Creates a deformation field

import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter

def median_smoothing(segmentation_map, kernel_size=3):
    """
    Apply a median filter to the segmentation map so that each pixel/voxel becomes the median
    of the values around it.
    
    Parameters:
        segmentation_map (np.array): The input binary segmentation map of any dimensionality.
        kernel_size (int): The size of the median filter kernel. Must be an odd number.
    
    Returns:
        np.array: The smoothed segmentation map.
    """
    # Apply the median filter
    smoothed_map = median_filter(segmentation_map, size=kernel_size, mode='constant', cval=0.0)
    
    return smoothed_map


def smooth_transitions(image, sigma=1):
    """
    Apply a Gaussian filter to create smooth transitions for non-zero values in a 2D numpy array.
    
    Args:
    - image (numpy.ndarray): The input 2D image array.
    - sigma (float): The standard deviation of the Gaussian kernel. Higher values result in smoother transitions.
    
    Returns:
    - numpy.ndarray: The smoothed image.
    """
    smoothed_image = gaussian_filter(image, sigma=sigma)
    return smoothed_image


def rotation_matrix_from_angles(angles):
    dimensions = len(angles)
    assert dimensions > 1, "Dimensions must be greater than 1"
    
    R = np.eye(dimensions)  # Start with an identity matrix of appropriate dimensions
    
    for i in range(dimensions - 1):
        angle = angles[i]
        rot_matrix_i = np.eye(dimensions)
        rot_matrix_i[i:i+2, i:i+2] = [[np.cos(angle), -np.sin(angle)],
                                      [np.sin(angle), np.cos(angle)]]
        R = np.dot(R, rot_matrix_i)
    
    return R

def linear_transformation_map(segmentation_map: np.ndarray, centroids: np.ndarray,
                              use_random_seed: bool = False, random_seed: int = 0, 
                              max_translation_ratio: float = 10.0, 
                              max_rotation_angle: float = 45.0, use_img = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly Transforms the Objects inside the Segmentation Map.
    
    Parameters:
    segmentation_map (np.ndarray): The segmentation map as a numpy array of shape [x, y, z].
    centroids (np.ndarray): Array of centroid coordinates for the objects in the segmentation map.
    use_random_seed (bool, optional): Use a random seed for reproducibility. Default is False.
    random_seed (int, optional): Random seed value if use_random_seed is True. Default is 0.
    max_translation_ratio (float, optional): Maximum translation ratio as a fraction of the maximum dimension size. 
                                            Default is 10.0.
    max_rotation_angle (float, optional): Maximum rotation angle in degrees. Default is 45.0.
    
    Returns:
    tuple: 
        new_segmentation_map (np.ndarray): The new segmentation map for training.
        deformation_field (np.ndarray): The deformation field of the points.
    """

    if use_random_seed:
        np.random.seed(random_seed)

    # Get the dimension 
    dim = segmentation_map.ndim

    # Get the maximum translation of a pixel
    max_translation = np.array(segmentation_map.shape).T / max_translation_ratio

    # Create a T and R matrix for each of the segmented regions
    unique_values = np.unique(segmentation_map)
    transformation_dict = {}

    for value in unique_values:
        if value != 0:  # Assuming 0 is the background and doesn't need transformation
            # Create translation vector T
            translation = (np.random.rand(dim) - 0.5) * 2  # Values between -1 and 1, dim variable
            T = translation * max_translation  # Scale it, shape [dim]

            # Get a rotation matrix
            rotation_angles = (np.random.rand(dim) - 0.5) * 2 * np.pi * max_rotation_angle / 360  # Angle for each dimension
            R = rotation_matrix_from_angles(rotation_angles)

            # Store in dictionary
            transformation_dict[value] = {'T': T, 'R': R}

    # Create the new segmentation map
    new_segmentation_map = np.zeros(segmentation_map.shape)
    deformation_field = np.zeros((*segmentation_map.shape, dim))

    for value in unique_values:
        if value != 0:  # Skip background
            # Get the indices of the current cluster
            cluster_indices = np.argwhere(segmentation_map == value)
            
            # Apply the transformation to all indices in the cluster
            transformed_indices = cluster_indices - centroids[value]
            transformed_indices = np.dot(transformed_indices, transformation_dict[value]['R'].T)
            transformed_indices = transformed_indices + centroids[value] + transformation_dict[value]['T']
            transformed_indices = np.rint(transformed_indices).astype(int)

            # Check if transformed indices are within bounds
            valid_mask = np.all((transformed_indices >= 0) & (transformed_indices < np.array(new_segmentation_map.shape)), axis=1)
            transformed_indices = transformed_indices[valid_mask]
            cluster_indices = cluster_indices[valid_mask]
            
            new_segmentation_map[tuple(transformed_indices.T)] = value
            
            # Ensure proper assignment by iterating over valid transformed indices
            for original_idx, transformed_idx in zip(cluster_indices, transformed_indices):
                deformation_field[tuple(original_idx)] = transformed_idx - original_idx

    return new_segmentation_map, deformation_field