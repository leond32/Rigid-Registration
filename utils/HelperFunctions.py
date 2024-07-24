from scipy.ndimage import distance_transform_edt
from PIL import Image
import numpy as np


def load_png_as_grayscale(image_path):
    """
    Load a PNG image as a 2D numpy array in grayscale.

    - Inputs -
    image_path : path to the PNG image file

    - Outputs -
    grayscale_array : the image as a 2D numpy array in grayscale
    """
    # Load the image
    image = Image.open(image_path)
    
    # Convert the image to grayscale
    grayscale_image = image.convert("L")
    
    # Convert the grayscale image to a numpy array
    grayscale_array = np.array(grayscale_image,dtype=np.int32)
    
    return grayscale_array

def deform_loop(image, deformation_field):
    new_image = np.zeros_like(image)
    rows,cols = image.shape
    for x in range(rows):
        for y in range(cols):
                new_x = (x + deformation_field[x,y,0]).astype(int)
                new_y = (y + deformation_field[x,y,1]).astype(int)
                if 0 <= new_x < rows and 0 <= new_y < cols:
                    new_image[new_x,new_y] = image[x,y]

    return new_image

