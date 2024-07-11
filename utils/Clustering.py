import numpy as np
from scipy.ndimage import label

def label_connected_components(segmentation_map):
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