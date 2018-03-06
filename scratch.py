# Create cardioid function
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
from skimage import measure

TPBX = 8
TPBY = 8
TPBZ = 8
@cuda.jit(device=True)
def f_heart(x,y,z):
    return 320 * ((-x**2 * z**3 -9*y**2 * z**3/80) + \
                  (x**2 + 9*y**2/4 + z**2-1)**3)

@cuda.jit
def f_heart_kernel(d_f, d_x, d_y, d_z):
    i, j, k = cuda.grid(3)
    nx, ny, nz = d_f.shape
    if i < nx and j < ny and k < nz:
        d_f[i,j,k] = f_heart(d_x[i], d_y[j], d_z[k])

def f_heart3D(X, Y, Z):
    nx = X.size
    ny = Y.size
    nz = Z.size

    d_x = cuda.to_device(X)
    d_y = cuda.to_device(Y)
    d_z = cuda.to_device(Z)
    d_f = cuda.device_array((nx,ny,nz), dtype=np.float32)

    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY,
                (nz + TPBZ - 1) // TPBZ
                )
    blockDims = (TPBX, TPBY, TPBZ)
    f_heart_kernel[gridDims, blockDims](d_f, d_x, d_y, d_z)
    return d_f.copy_to_host()


@cuda.jit(device=True)
def f_sphere(x,y,z,R):
    return 1 * (x**2 + y**2 + z**2 - R**2)

@cuda.jit
def f_sphere_kernel(d_f, d_x, d_y, d_z, R):
    i, j, k = cuda.grid(3)
    nx, ny, nz = d_f.shape
    if i < nx and j < ny and k < nz:
        d_f[i,j,k] = f_heart(d_x[i], d_y[j], d_z[k], R)

def f_sphere3D(X, Y, Z, R):
    nx = X.size
    ny = Y.size
    nz = Z.size

    d_x = cuda.to_device(X)
    d_y = cuda.to_device(Y)
    d_z = cuda.to_device(Z)
    d_f = cuda.device_array((nx,ny,nz), dtype=np.float32)

    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY,
                (nz + TPBZ - 1) // TPBZ
                )
    blockDims = (TPBX, TPBY, TPBZ)
    f_sphere_kernel[gridDims, blockDims](d_f, d_x, d_y, d_z, R)
    return d_f.copy_to_host()

def morph(shape_start, shape_end, t):
    n = len(shape_start)
    rst = np.zeros((n,n,n), dtype=np.float32)
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                rst[ix,iy,iz] = shape_start[ix,iy,iz] * (1-t) + t * shape_end[ix,iy,iz]
    return rst

@cuda.jit
def morph_kernel(d_f, d_x, d_y, t):
    i, j, k = cuda.grid(3)
    nx, ny, nz = d_f.shape
    if i < nx and j < ny and k < nz:
        d_f[i,j,k] = d_x[i,j,k] * (1-t) + t * d_y[i,j,k]

def morph3D(shape_start, shape_end, t):
    nx, ny, nz = shape_start.shape

    d_x = cuda.to_device(shape_start)
    d_y = cuda.to_device(shape_end)
    d_f = cuda.device_array((nx,ny,nz), dtype=np.float32)

    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY,
                (nz + TPBZ - 1) // TPBZ
                )
    blockDims = (TPBX, TPBY, TPBZ)
    f_sphere_kernel[gridDims, blockDims](d_f, d_x, d_y, t)
    return d_f.copy_to_host()

def main():
    # Set up mesh
    n = 100 
    R = 3.0
    x = np.linspace(-3,3,n)
    y = np.linspace(-3,3,n)
    z = np.linspace(-3,3,n)
    X, Y, Z =  np.meshgrid(x, y, z)
    heart = f_heart3D(X,Y,Z)
    sphere = f_sphere3D(X,Y,Z,R)

    surf = morph3D(sphere, heart, 0.90)
    # Extract a 2D surface mesh from a 3D volume (F=0)
    verts, faces = measure.marching_cubes_classic(surf, 0.0, spacing=(0.1, 0.1, 0.1))

    # Create a 3D figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                    cmap='Spectral', lw=1)

    # Change the angle of view and title
    ax.view_init(15, -15)

    # ax.set_title(u"Made with ❤ (and Python)", fontsize=15) # if you have Python 3
    ax.set_title("Made with <3 (and Python)", fontsize=15)

    # Show me some love ^^
    plt.show()