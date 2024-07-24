import numpy as np
import nibabel as nib
import scipy.ndimage

def load_nii_as_numpy(path_to_nifty, target_spacing=0.31):
    """
    Load a NIfTI file and adjust its voxel spacing to be uniform.

    Parameters:
    path_to_nifty (str): Path to the NIfTI file.
    target_spacing (float): Desired voxel spacing in mm for all dimensions.

    Returns:
    np.ndarray: The resampled array with uniform voxel spacing.
    """
    # Load the NIfTI file
    img = nib.load(path_to_nifty)

    # Get the current voxel spacing from the header
    header = img.header
    current_spacing = header['pixdim'][1:4]  # Spacing along x, y, z dimensions

    # Convert the image data to a numpy array
    segmentation_map = np.array(img.dataobj)

    # Calculate the zoom factors for each dimension
    zoom_factors = current_spacing / target_spacing

    # Resample the array using scipy.ndimage.zoom
    resampled_array = scipy.ndimage.zoom(segmentation_map, zoom_factors, order=1)

    return resampled_array