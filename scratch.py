# Create cardioid function
import math
from numba import cuda
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
from skimage import measure
import plotUtils

import util as *


def main():
    # Set up mesh
    n = 100
    R = 3.0
    m = 3.0
    dx = 2 * m / (n - 1)
    dy = 2 * m / (n - 1)
    dz = 2 * m / (n - 1)
    x = np.linspace(-m, m, n)
    y = np.linspace(-m, m, n)
    z = np.linspace(-m, m, n)
    X, Y, Z = np.meshgrid(x, y, z)

    # CUDA
    # heart = f_heart3D(X, Y, Z)
    # sphere = f_sphere3D(X, Y, Z, R)

    # without cuda
    heart = heart(X, Y, Z)
    sphere = sphere(X, Y, Z, R)

    surf = morph3D(sphere, heart, 0.90)

    # Extract a 2D surface mesh from a 3D volume (F=0)
    verts, faces = measure.marching_cubes_classic(surf, 0.0, spacing=(0.1, 0.1, 0.1))

    # Create a 3D figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                    cmap='Spectral', lw=1)

    # Change the angle of view and title
    ax.view_init(15, -15)

    # ax.set_title(u"Made with ❤ (and Python)", fontsize=15) # if you have Python 3
    ax.set_title("Made with <3 (and Python)", fontsize=15)

    # Show me some love ^^
    plt.show()
    '''
    plotUtils.arraycontourplot3d(surf, x, y, z, dx, dy, dz, vars=['x', 'y', 'z'], titlestring="morph3D", filename="morph3D")
    '''
    filename = 'shit'

    print('Object exported to ' + filename + '.ply')
    exportPLY(filename, verts, faces)


if __name__ == '__main__':
    main()
