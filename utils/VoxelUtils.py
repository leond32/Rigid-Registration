import numpy as np
from pyvox.models import Vox
from pyvox.parser import VoxParser
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_voxel_as_numpy(file_path, padding=0):
    # Load the .vox file
    vox = VoxParser(file_path).parse()
    
    # Extract voxel data
    voxel_data = vox.models[0].voxels
    
    # Determine the dimensions of the voxel grid
    if not voxel_data:
        raise ValueError("No voxels found in the provided .vox file.")
    
    # Calculate the size of the grid by finding the max coordinates
    max_x = max(voxel.x for voxel in voxel_data)
    max_y = max(voxel.y for voxel in voxel_data)
    max_z = max(voxel.z for voxel in voxel_data)
    
    # Include padding
    padded_width = max_x + 1 + 2 * padding
    padded_height = max_y + 1 + 2 * padding
    padded_depth = max_z + 1 + 2 * padding
    
    # Create a NumPy array with padding
    voxel_array = np.zeros((padded_width, padded_height, padded_depth), dtype=np.uint8)
    
    # Populate the NumPy array with voxel data
    for voxel in voxel_data:
        x, y, z = voxel.x, voxel.y, voxel.z
        voxel_array[x + padding, y + padding, z + padding] = 1

    return voxel_array



def create_voxel_mesh(x, y, z, size=1, color='red'):
    # Define the vertices of a cube
    vertices = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ])

    # Scale the vertices
    vertices = vertices * size

    # Define the faces of the cube
    faces = np.array([
        [0, 1, 3, 2], # front face
        [4, 5, 7, 6], # back face
        [0, 1, 5, 4], # bottom face
        [2, 3, 7, 6], # top face
        [0, 2, 6, 4], # left face
        [1, 3, 7, 5], # right face
    ])

    # Create the 3D mesh
    mesh = go.Mesh3d(
        x=vertices[:, 0] + x,
        y=vertices[:, 1] + y,
        z=vertices[:, 2] + z,
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=0.5
    )

    return mesh

def visualize_voxel_array(voxel_array, voxel_size=1):
    # Get the coordinates of the voxels that are set to 1
    filled_voxels = np.argwhere(voxel_array == 1)

    # Create a plotly figure
    fig = go.Figure()

    # Add a cube for each voxel
    for (x, y, z) in filled_voxels:
        mesh = create_voxel_mesh(x, y, z, size=voxel_size, color='red')
        fig.add_trace(mesh)

    # Set the aspect ratio to be equal
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[0, voxel_array.shape[0]]),
            yaxis=dict(nticks=10, range=[0, voxel_array.shape[1]]),
            zaxis=dict(nticks=10, range=[0, voxel_array.shape[2]]),
            aspectmode='cube'
        ),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10)
    )

    # Show the plot
    fig.show()

def visualize_3d_array(data):
    """
    Visualizes a 3D NumPy array using a scatter plot with a color scale.
    
    Parameters:
    data (numpy.ndarray): A 3D NumPy array with shape (depth, height, width).
                          Each value in the array represents the intensity (grayscale).
    """
    depth, height, width = data.shape
    
    # Create coordinate grids
    z, y, x = np.indices((depth, height, width))

    # Flatten the arrays for scatter plot
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    c = data.flatten()  # Intensity values for color scale

    # Only consider points with intensity > 0
    mask = c > 0
    x = x[mask]
    y = y[mask]
    z = z[mask]
    c = c[mask]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Use scatter to create a 3D scatter plot with color scale based on intensity values
    scatter = ax.scatter(x, y, z, c=c, cmap='viridis')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Add a color bar to show the intensity scale
    colorbar = plt.colorbar(scatter, ax=ax, label='Intensity')

    plt.show()