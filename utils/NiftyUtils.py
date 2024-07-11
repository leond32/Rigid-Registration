import numpy as np
import nibabel as nib


def load_nii(path_to_nifty):

    img = nib.load(path_to_nifty)

    segmentation_map = np.array(img.dataobj)
    
    return segmentation_map