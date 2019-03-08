import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import colors as mcolors
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.stats import gaussian_kde
import matplotlib.animation as animation

import plotly
import plotly.plotly as py
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
        color="#FFFFFF",                # set color to an array/list of desired values
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
        color=scatter_plot_colors,                # set color to an array/list of desired values
        colorscale='Jet',   # choose a colorscale
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



    # dpi = 96
    #
    # fig = plt.figure(figsize=(600 / dpi, 600 / dpi), dpi=dpi)
    # ax = fig.add_subplot(111, projection='3d')
    #
    # def init():
    #     # Table
    #
    #     # Top
    #     x = [1.52, 1.03, 1.03, 1.52]
    #     y = [0.3957, 0.3957, 1.112, 1.112]
    #     z = [0.395, 0.395, 0.395, 0.395]
    #     verts = [list(zip(x, y, z))]
    #     ax.add_collection3d(Poly3DCollection(verts, facecolors=cc('orange')))
    #
    #     # Bottom
    #     x = [1.52, 1.03, 1.03, 1.52]
    #     y = [0.3957, 0.3957, 1.112, 1.112]
    #     z = [0, 0, 0, 0]
    #     verts = [list(zip(x, y, z))]
    #     ax.add_collection3d(Poly3DCollection(verts, facecolors=cc('orange')))
    #
    #     # Front
    #     x = [1.52, 1.52, 1.52, 1.52]
    #     y = [0.3957, 0.3957, 1.112, 1.112]
    #     z = [0, 0.395, 0.395, 0]
    #     verts = [list(zip(x, y, z))]
    #     ax.add_collection3d(Poly3DCollection(verts, facecolors=cc('orange')))
    #
    #     # Back
    #     x = [1.03, 1.03, 1.03, 1.03]
    #     y = [0.3957, 0.3957, 1.112, 1.112]
    #     z = [0, 0.395, 0.395, 0]
    #     verts = [list(zip(x, y, z))]
    #     ax.add_collection3d(Poly3DCollection(verts, facecolors=cc('orange')))
    #
    #     # Left
    #     x = [1.52, 1.03, 1.03, 1.52]
    #     y = [0.3957, 0.3957, 0.3957, 0.3957]
    #     z = [0, 0, 0.395, 0.395]
    #     verts = [list(zip(x, y, z))]
    #     ax.add_collection3d(Poly3DCollection(verts, facecolors=cc('orange')))
    #
    #     # Right
    #     x = [1.52, 1.03, 1.03, 1.52]
    #     y = [1.112, 1.112, 1.112, 1.112]
    #     z = [0, 0, 0.395, 0.395]
    #     verts = [list(zip(x, y, z))]
    #     ax.add_collection3d(Poly3DCollection(verts, facecolors=cc('orange')))
    #
    #     # Floor
    #     x = [1.65, -0.05, -0.05, 1.65]
    #     y = [1.45, 1.45, 0, 0]
    #     z = [0, 0, 0, 0]
    #     verts = [list(zip(x, y, z))]
    #     ax.add_collection3d(Poly3DCollection(verts, facecolors=cc('dimgray', alpha=0.6)))
    #
    #     plot_3D_cylinder(ax=ax, radius=0.25, height=0.8, elevation=0, resolution=6, color=cc('gray'), x_center=0.5, y_center=0.753)
    #
    #     # idx = scatter_plot_colors.argsort()
    #
    #
    #     ax.scatter(*location_records, s=3, c=scatter_plot_colors, marker='o', alpha=0.2, cmap=plt.cm.jet)
    #
    #     plt.axis('off')
    #     ax.grid(False)
    #     ax.set_xlim(0.20, 1.65)
    #     ax.set_ylim(0, 1.45)
    #     ax.set_zlim(-0.4, 1.05)
    #     ax.dist = 7
    #
    #     return ()
    #
    # def animate(i):
    #     ax.view_init(elev=15., azim=(i*2)-50)
    #     return ()
    #
    #
    # # ax.auto_scale_xyz()
    #
    # anim = animation.FuncAnimation(fig, animate, init_func=init,
    #                                frames=100, interval=1, blit=True)
    #
    # # WriterClass = animation.writers['ffmpeg']
    # # writer = animation.FFMpegFileWriter(fps=24, metadata=dict(artist='bww'), bitrate=900)
    # writer = animation.ImageMagickFileWriter(fps=10, metadata=dict(artist='bww'), bitrate=500)
    #
    # heatmap_save_path = os.path.join(working_dir, "{}_heatmap.gif".format(file_prefix))
    # anim.save(heatmap_save_path, writer=writer)
    #
    # plt.close()
    #
    # if delete_records:
    #     for file_name in file_names:
    #         try:
    #             os.remove(os.path.join(working_dir, file_name))
    #         except OSError:
    #             pass
    #
    # return heatmap_save_path


def generate_animated_3d_fetch_stack_heatmap_from_npy_records(working_dir, file_prefixes, delete_records=False):

    scatter_objects = []

    for file_prefix in file_prefixes:
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

        scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        name="Gripper Positions",
        mode='markers',
        marker=dict(
            size=2,
            color=scatter_plot_colors,                # set color to an array/list of desired values
            colorscale='Jet',   # choose a colorscale
            opacity=0.6
        ))

        scatter_objects.append(scatter)

    initial_gripper_pos = go.Scatter3d(
        x=[1.3419],
        y=[0.7291],
        z=[0.5347],
        mode='markers',
        name="Initial Gripper Position",
        marker=dict(
            symbol="diamond-open",
            size=6,
            color="#FFFFFF",  # set color to an array/list of desired values
            opacity=1
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

    heatmap_save_path = os.path.join(working_dir, "animated_heatmap.html")

    saved_file_link = plotly.offline.plot({
    "data": [initial_gripper_pos, robot, table],
    "layout": go.Layout(
        title=file_prefixes[0],
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
                zeroline=False,)),
        updatemenus=[{'type': 'buttons',
                                      'buttons': [{'label': 'Play',
                                                   'method': 'animate',
                                                   'args': [None, {'frame': {'duration': 500, 'redraw': False},
                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}]}]}]),
    "frames": [{'data': [scatter_objects[i]],
                "layout": go.Layout(title=file_prefixes[i])
                } for i in range(1, len(scatter_objects))]

    }, auto_open=False, filename=heatmap_save_path)

    if delete_records:
        for file_name in file_names:
            try:
                os.remove(os.path.join(working_dir, file_name))
            except OSError:
                pass

    return saved_file_link


if __name__ == '__main__':
    # print(generate_3d_fetch_stack_heatmap_from_npy_records(working_dir='basic_functionality_test_31-12-2018_16:51:11/heatmaps',
    #                                                  file_prefix='epoch1', delete_records=False))

    print(generate_animated_3d_fetch_stack_heatmap_from_npy_records(
        working_dir='/tmp/sparse_test_intrinsic_no_sub_goals_8_workers_SEED_31_31-12-2018_18:29:33/heatmaps/',
        file_prefixes=['epoch{}'.format(i) for i in range(1, 10)],
        delete_records=False))
