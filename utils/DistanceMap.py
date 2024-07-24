from scipy.ndimage import distance_transform_edt
import numpy as np

def cluster_proximity_and_distance(segmentation_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binary_map = (segmentation_map == 0).astype(np.uint8)
    distance_map = distance_transform_edt(binary_map)
    proximity_map = np.zeros_like(segmentation_map)
    unique_values = np.unique(segmentation_map)
    unique_values = unique_values[unique_values != 0]

    for value in unique_values:
        mask = (segmentation_map == value)
        distance_to_cluster = distance_transform_edt(~mask)
        proximity_map[(proximity_map == 0) & (distance_map > 0)] = value * (distance_to_cluster < distance_map[(proximity_map == 0) & (distance_map > 0)])
        distance_map = np.minimum(distance_map, distance_to_cluster)

    return proximity_map, distance_map
