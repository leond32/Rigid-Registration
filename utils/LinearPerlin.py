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

def linear_transformation_map(segmentation_map, use_random_seed=False, random_seed=0, max_translation_ratio=10):
    """ Linearly Transforms the Objects inside the Segmentation Map. 
        - Inputs - 
        segmentation_map : the segmentation map as a numpy array of shape [x,y]

        - Optional Inputs -
        use_random_seed : use a random seed for reproducibility, default False
        random_seed : choose other random seed than 0, default 0
        max_translation_ratio : what fraction of the max dimension size should the translation be at max

        - Outputs -
        new_segmentation_map : the new segmentation map for training
        deformation_field : the deformation field of the points
    """
    
    if use_random_seed:
        np.random.seed(random_seed)
        
    #get the dimension 
    dim = segmentation_map.ndim

    # get the maximum translation of a pixel
    max_translation = np.array(segmentation_map.shape).T/max_translation_ratio

    # calculate the translation 
    translation = (np.random.rand(dim)-0.5)*2 #values between -1 and 1, dim variable
    T = translation * max_translation #scale it, shape [dim]

    # get a rotation matrix
    rotation_angles = np.random.rand(dim) * 2 * np.pi #angle for each dimension
    R = rotation_matrix_from_angles(rotation_angles)

    new_segmentation_map = np.zeros(segmentation_map.shape)

    for idx in np.ndindex(segmentation_map.shape):

        #if the segmentation map has the correct index, apply the trafo
        if segmentation_map[idx] == 1.0:
            new_idx = np.array(idx) + T
            new_idx = new_idx.dot(R)
            new_idx = new_idx + np.array(segmentation_map.shape)/2
            new_idx = np.floor(new_idx).astype(int)
            is_within_bounds = all(0 <= i < d for i, d in zip(new_idx, new_segmentation_map.shape))
            if is_within_bounds:
                new_segmentation_map[tuple(new_idx)] = 1

    return new_segmentation_map