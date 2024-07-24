import numpy as np
from scipy.ndimage import label
import scipy.ndimage as ndi
from skimage import io
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter

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

def label_connected_components(segmentation_map:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    """
    Label connected components in the segmentation map and set the values
    of the pixels that are part of this component to a certain value.
    Also calculates the centroid of each labeled component.

    Parameters:
        segmentation_map (np.array): The input grayscale segmentation map of any dimensionality.

    Returns:
        np.array: The labeled segmentation map with unique values for each component.
        dict: A dictionary where keys are component labels and values are centroids of the components.
    """
    # Create a binary map where background (0) is False and objects (>0) are True
    binary_map = segmentation_map > 0

    # Label connected components
    labeled_map, num_features = label(binary_map)

    # Create a new map to store the uniquely labeled components
    unique_labeled_map = np.zeros_like(segmentation_map)

    # Dictionary to store centroids
    centroids = {}

    # Assign unique labels to each connected component and calculate centroids
    current_label = 1
    for i in range(1, num_features + 1):
        component_mask = (labeled_map == i)
        unique_labeled_map[component_mask] = current_label
        
        # Calculate the centroid of the component
        indices = np.argwhere(component_mask)
        centroid = indices.mean(axis=0)
        centroids[current_label] = centroid

        current_label += 1

    return unique_labeled_map, centroids

def color_pixels_based_on_distance(clusters, max_distance):
    """
    Color pixels based on their distance to the nearest cluster within a maximum distance,
    ensuring clear delineation between clusters.
    
    Parameters:
    - clusters: NumPy array (2D or 3D) where each element is a unique cluster number or 0.
    - max_distance: Maximum distance to a cluster within which pixels will be colored.

    Returns:
    - result_map: NumPy array (2D or 3D) with the same shape as input, where each pixel is colored
                  with the ID of the nearest cluster within max_distance.
    """
    
    # Initialize distance_map and result_map
    distance_map = np.full(clusters.shape, np.inf)
    result_map = np.zeros(clusters.shape, dtype=int)

    cluster_ids = np.unique(clusters)
    cluster_ids = cluster_ids[cluster_ids != 0]  # Remove background cluster (0)
    
    # Dictionary to hold distance transforms for each cluster
    distance_transforms = {}
    
    # Compute distance transforms for each cluster separately
    for cluster_id in cluster_ids:
        cluster_mask = clusters == cluster_id
        distance_transforms[cluster_id] = ndi.distance_transform_edt(~cluster_mask)
    
    # Stack all distance transforms into a single array
    all_distances = np.stack([distance_transforms[cluster_id] for cluster_id in cluster_ids], axis=-1)
    
    # Find the minimum distance and the corresponding cluster
    min_distances = np.min(all_distances, axis=-1)
    closest_clusters = np.argmin(all_distances, axis=-1) + 1  # +1 to map to actual cluster ids
    
    # Handle equidistant points by setting them to 0
    sorted_distances = np.sort(all_distances, axis=-1)
    equidistant_mask = np.isclose(sorted_distances[..., 0], sorted_distances[..., 1])
    
    # Apply max_distance threshold and set equidistant points to 0
    result_map = np.where((min_distances <= max_distance) & ~equidistant_mask, closest_clusters, 0)
    
    return result_map

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

def scale_perlin_by_distance(perlin_2d,mask):
    distance = distance_transform_edt(mask != 0)
    scaled_perlin = perlin_2d*mask[:,:,np.newaxis]*distance[:,:,np.newaxis]*0.001
    return scaled_perlin

def create_linear_deformation_field(segmentation_map: np.ndarray,distance: int):
    """
    This function applies all the steps necessary to create the final deformation field. 
    The returned deformation field is for the whole image. The mask indicates where the segmented objects are, 
    and the segmentation_deformation_field is the deformation field to just change the segmentation map.
    The distance parameter determines how far away from the segmented parts the pixels should be treated as rigid bodies.

    Parameters:
    segmentation_map (np.ndarray): The input segmentation map where different objects are labeled.
    distance (int): The distance from the segmented parts within which pixels should be treated as rigid bodies.

    Returns:
    tuple: 
        final_deformation_field (np.ndarray): The deformation field applied to the whole image.
        mask (np.ndarray): A binary mask indicating the locations of the segmented objects.
        segmentation_deformation_field (np.ndarray): The deformation field applied to the segmented parts.
    """

    #first label all the components in the segmentation map
    segmentation_map,centroids = label_connected_components(segmentation_map)
    #create a segmentation map with pixels colored that are around the rigid objects for later smooth transition from linear to perlin
    segmentation_map_all = color_pixels_based_on_distance(segmentation_map,distance)
    #now linearly deform these objects
    _,deformation_field = linear_transformation_map(segmentation_map_all, centroids,use_random_seed=True)
    mask = (segmentation_map != 0).astype(int)
    segmentation_deformation_field = deformation_field*mask[:,:,np.newaxis]
    #smooth the field
    deformation_field = smooth_transitions(deformation_field,6)
    distance_map = distance_transform_edt(mask == 0)
    td_distance = np.stack((distance_map,distance_map),axis=-1)
    #make the deformation "weaker", the farther you are away from true rigid bodies
    final_deformation_field =  np.exp(-td_distance*5/distance)*deformation_field 
    mask = (mask == 0)
    return final_deformation_field,mask,segmentation_deformation_field