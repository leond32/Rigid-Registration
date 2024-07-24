import numpy as np
from scipy.ndimage import label
import scipy.ndimage as ndi
from skimage import io

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