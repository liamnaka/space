import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


# Sphere plotting logic from: https://stackoverflow.com/a/44627342
def plot_heatmap_on_sphere(heatmap, save_filename):

    # Creating the theta and phi values.z
    ntheta, nphi = heatmap.shape

    theta = np.linspace(0, np.pi*1, ntheta+1)
    phi = np.linspace(0, np.pi*2, nphi+1)

    # Creating the coordinate grid for the unit sphere.
    X = np.outer(np.sin(theta), np.cos(phi))
    Y = np.outer(np.sin(theta), np.sin(phi))
    Z = np.outer(np.cos(theta), np.ones(nphi+1))

    # Creating a 2D array to be color-mapped on the unit sphere.
    # {X, Y, Z}.shape → (ntheta+1, nphi+1) but c.shape → (ntheta, nphi)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    # Creating the colormap thingies.
    cm = mpl.cm.inferno
    sm = mpl.cm.ScalarMappable(cmap=cm)
    sm.set_array([])

    # Creating the plot.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm(heatmap), alpha=0.5)
    ax.set_ylabel('Azimuth')
    ax.set_zlabel('Elevation')

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Get rid of the panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    plt.colorbar(sm)

    # Save the plot.
    plt.savefig(save_filename, bbox_inches='tight')
