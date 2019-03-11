import os
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.stats import gaussian_kde

import plotly
import plotly.graph_objs as go

# https://stackoverflow.com/questions/26989131/add-cylinder-to-plot
def plot_3D_cylinder(ax, radius, height, elevation=0, resolution=100, color='gray', x_center=0., y_center=0.):
    x = np.linspace(x_center-radius, x_center+radius, resolution)
    z = np.linspace(elevation, elevation+height, resolution)
    X, Z = np.meshgrid(x, z)

    Y = np.sqrt(radius**2 - (X - x_center)**2) + y_center # Pythagorean theorem

    ax.plot_surface(X, Y, Z, linewidth=0, color=color)
    ax.plot_surface(X, (2*y_center-Y), Z, linewidth=0, color=color)

    floor = Circle((x_center, y_center), radius, color=color)
    ax.add_patch(floor)
    art3d.pathpatch_2d_to_3d(floor, z=elevation, zdir="z")

    ceiling = Circle((x_center, y_center), radius, color=color)
    ax.add_patch(ceiling)
    art3d.pathpatch_2d_to_3d(ceiling, z=elevation+height, zdir="z")


def cc(color, alpha=0.3):
    return mcolors.to_rgba(color, alpha=alpha)


def generate_3d_fetch_stack_heatmap_from_npy_records(working_dir, file_prefix, delete_records=False):

    file_names = [file_name for file_name in os.listdir(working_dir)
                  if file_name.endswith(".npy") and file_name.startswith(file_prefix)]

    location_records = np.concatenate(
        [np.load(os.path.join(working_dir, file_name)) for file_name in file_names],
        axis=0)

    max_heatmap_samples = 5000

    location_records = location_records[np.random.choice(len(location_records),
                                                                         min(max_heatmap_samples,
                                                                             len(location_records)),
                                                                         replace=False)]


    location_records = location_records.swapaxes(0, 1)
    scatter_plot_colors = gaussian_kde(location_records)(location_records)

    x, y, z = location_records

    initial_gripper_pos = go.Scatter3d(
    x=[1.3419],
    y=[0.7291],
    z=[0.5347],
    mode='markers',
    name="Initial Gripper Position",
    marker=dict(
        symbol="diamond-open",
        size=6,
        color="#FFFFFF",
        opacity=1
    ))

    scatter = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    name="Gripper Positions",
    mode='markers',
    marker=dict(
        size=2,
        color=scatter_plot_colors,
        colorscale='Jet',
        opacity=0.6
    ))

    table_x = [1.03, 1.03, 1.52, 1.52, 1.03, 1.03, 1.52, 1.52]
    table_y = [0.3957, 1.112, 1.112, 0.3957, 0.3957, 1.112, 1.112, 0.3957]
    table_z = [0, 0, 0, 0, 0.395, 0.395, 0.395, 0.395]

    table = go.Mesh3d(x=table_x, y=table_y, z=table_z, name="Table",
                      i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                      j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                      k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                      opacity=0.6,
                      color='#9900cc',)

    robot_x = [0.75, 0.75, 0.25, 0.65, 0.65, 0.65]
    robot_y = [0.5, 1, 0.75, 0.75, 0.5, 1]
    robot_z = [0, 0, 0, 0.9, 0, 0]

    robot = go.Mesh3d(x=robot_x, y=robot_y, z=robot_z, name="Robot",
                      alphahull=1,
                      opacity=1,
                      color='#DDDDDD')

    heatmap_save_path = os.path.join(working_dir, "{}_heatmap.html".format(file_prefix))

    saved_file_link = plotly.offline.plot({
    "data": [initial_gripper_pos, robot, table, scatter],
    "layout": go.Layout(
        title=file_prefix,
        paper_bgcolor="#111111",
        showlegend=True,
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgb(200, 200, 230)",
                gridcolor="rgb(255, 255, 255)",
                showbackground=False,
                zerolinecolor="rgb(255, 255, 255)",
                visible=False,
                showgrid=False,
                showline=False,
                zeroline=False,),
            yaxis=dict(
                backgroundcolor="rgb(230, 200,230)",
                gridcolor="rgb(255, 255, 255)",
                showbackground=False,
                zerolinecolor="rgb(255, 255, 255)",
                visible=False,
                showgrid=False,
                showline=False,
                zeroline=False,),
            zaxis=dict(
                backgroundcolor="rgb(230, 230,200)",
                gridcolor="rgb(255, 255, 255)",
                showbackground=False,
                zerolinecolor="rgb(255, 255, 255)",
                visible=False,
                showgrid=False,
                showline=False,
                zeroline=False,))
    )}, auto_open=False, filename=heatmap_save_path)

    if delete_records:
        for file_name in file_names:
            try:
                os.remove(os.path.join(working_dir, file_name))
            except OSError:
                pass

    return saved_file_link
