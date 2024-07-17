import numpy as np
import plotly.graph_objects as go
from skimage import measure

def plot_3d(image):
    # Create a 3D mesh from the 3D numpy array
    verts, faces, _, _ = measure.marching_cubes(image, level=0.5)

    # Create a Plotly mesh plot
    x, y, z = zip(*verts)
    I, J, K = zip(*faces)

    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K,
                                    color='lightpink', opacity=0.50)])

    fig.update_layout(scene=dict(aspectmode='data'),
                      title='3D Visualization of 3D Image')
    fig.show()

# Example usage
# Assuming `image_3d` is a 3D numpy array
image_3d = np.random.rand(50, 50, 50) > 0.5  # Example 3D binary image
plot_3d(image_3d)
