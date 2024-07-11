# A noise applied to all points. Points within the segmentation are only linearly transformed, 
# while other points are linearly and non-linearly transormed
# -> Creates a deformation field

import numpy as np

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

def LinearTransformationMap2D(segmentation_map, use_random_seed=False, random_seed=0):
    """ Linearly Transforms the Objects inside the Segmentation Map. 
        - Inputs - 
        segmentation_map : the segmentation map as a numpy array of shape [x,y]

        - Optional Inputs -
        use_random_seed : use a random seed for reproducibility, default False
        random_seed : choose other random seed than 0, default 0

        - Outputs -
        new_segmentation_map : the new segmentation map for training
        deformation_field : the deformation field of the points
    """
    
    if use_random_seed:
        np.random.seed(random_seed)
    
    # Random rotation angle (in radians)
    theta = np.deg2rad(np.random.uniform(0, 360))

    # Random translation vector
    translation = np.random.uniform(-10, 10, size=2)

    # Random rotation pivot point within the map dimensions
    pivot = np.random.uniform(0, np.array(segmentation_map.shape))

    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Homogeneous transformation matrices
    T_to_origin = np.eye(3)
    T_to_origin[:2, 2] = -pivot

    T_back = np.eye(3)
    T_back[:2, 2] = pivot

    T_final = np.eye(3)
    T_final[:2, 2] = translation

    # Combined transformation matrix
    M = T_final @ T_back @ R @ T_to_origin

    # Generate the coordinates for the entire 2D space
    coords = np.array(np.meshgrid(
        np.arange(segmentation_map.shape[0]),
        np.arange(segmentation_map.shape[1])
    )).T.reshape(-1, 2)

    # Select only the coordinates where segmentation_map is 1
    rigid_coords = coords[segmentation_map.ravel() == 1]

    # Transform rigid points
    rigid_coords_homogeneous = np.hstack((rigid_coords, np.ones((rigid_coords.shape[0], 1))))
    transformed_coords_homogeneous = (M @ rigid_coords_homogeneous.T).T
    transformed_coords = transformed_coords_homogeneous[:, :2]

    # Create a new segmentation map with transformed coordinates
    new_segmentation_map = np.zeros_like(segmentation_map)
    deformation_field = np.zeros_like(coords, dtype=float)

    for original, transformed in zip(rigid_coords, transformed_coords):
        x, y = transformed.astype(int)
        if 0 <= x < segmentation_map.shape[0] and 0 <= y < segmentation_map.shape[1]:
            new_segmentation_map[x, y] = 1
            deformation_field[np.all(coords == original, axis=1)] = transformed - original

    deformation_field = deformation_field.reshape(segmentation_map.shape + (2,))

    return new_segmentation_map, deformation_field