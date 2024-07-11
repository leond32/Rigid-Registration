import numpy as np
from scipy.ndimage import distance_transform_edt
import plotly.graph_objects as go
from ipywidgets import fixed,interactive

def create_distance_map(segmentation_array):
    #create a distance map
    distance_map = distance_transform_edt(segmentation_array == 0)
    return distance_map

def plot_slice(distance_map, z):
    fig = go.Figure(data=go.Heatmap(
        z=distance_map[:, :, z],
        colorscale='Viridis'
    ))
    fig.update_layout(
        title=f'Layer {z}',
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        xaxis=dict(scaleanchor="y", scaleratio=1),  # Ensure square pixels
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    fig.show()

def visualize_distance_map(distance_map):
    # Create an interactive slider to scroll through the layers
    interactive_plot = interactive(plot_slice, distance_map=fixed(distance_map), z=(0, distance_map.shape[2] - 1))
    return interactive_plot