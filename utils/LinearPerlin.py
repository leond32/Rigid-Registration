# A noise applied to all points. Points within the segmentation are only linearly transformed, 
# while other points are linearly and non-linearly transormed
# -> Creates a deformation field

import numpy as np

def LinearTransformationMap(segmentation_map, use_random_seed = False, random_seed = 0):
    """ Linearly Transforms the Objects inside the Segmentation Map. 
        - Inputs - 
        segmentation_map : the segmentation map as a numpy array of shape [x,y,z]

        - Optional Inputs -
        use_random_seed : use a random seed for reproducability, default 0
        random_seed : choose other random seed than 0

        - Outputs -
        new_segmentation_map : the new segmentation map for training
        deformation_field : the deformagtion field of the points
    """

    if use_random_seed:
        np.random.seed(random_seed)
    
    # Random rotation angles (in radians)
    theta_x = np.deg2rad(np.random.uniform(0, 360))
    theta_y = np.deg2rad(np.random.uniform(0, 360))
    theta_z = np.deg2rad(np.random.uniform(0, 360))

    # Random translation vector
    translation = np.random.uniform(-10, 10, size=3)

    # Random rotation pivot point within the map dimensions
    pivot = np.random.uniform(0, np.array(segmentation_map.shape))

    # Rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

    R_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    R_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    R = R_z @ R_y @ R_x

    # Homogeneous transformation matrices
    T_to_origin = np.eye(4)
    T_to_origin[:3, 3] = -pivot

    T_back = np.eye(4)
    T_back[:3, 3] = pivot

    T_final = np.eye(4)
    T_final[:3, 3] = translation

    # Combined transformation matrix
    M = T_final @ T_back @ R @ T_to_origin

    # Generate the coordinates for the entire 3D space
    coords = np.array(np.meshgrid(
        np.arange(segmentation_map.shape[0]),
        np.arange(segmentation_map.shape[1]),
        np.arange(segmentation_map.shape[2])
    )).T.reshape(-1, 3)

    # Select only the coordinates where segmentation_map is 1
    rigid_coords = coords[segmentation_map.ravel() == 1]

    # Transform rigid points
    rigid_coords_homogeneous = np.hstack((rigid_coords, np.ones((rigid_coords.shape[0], 1))))
    transformed_coords_homogeneous = (M @ rigid_coords_homogeneous.T).T
    transformed_coords = transformed_coords_homogeneous[:, :3]

    # Create a new segmentation map with transformed coordinates
    new_segmentation_map = np.zeros_like(segmentation_map)
    deformation_field = np.zeros_like(coords, dtype=float)

    for original, transformed in zip(rigid_coords, transformed_coords):
        x, y, z = transformed.astype(int)
        if 0 <= x < segmentation_map.shape[0] and 0 <= y < segmentation_map.shape[1] and 0 <= z < segmentation_map.shape[2]:
            new_segmentation_map[x, y, z] = 1
            deformation_field[np.all(coords == original, axis=1)] = transformed - original

    deformation_field = deformation_field.reshape(segmentation_map.shape + (3,))

    return new_segmentation_map, deformation_field