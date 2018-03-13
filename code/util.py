# Create cardioid function
import math
from numba import cuda
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
from skimage import measure
import plotUtils

TPBX = 8
TPBY = 8
TPBZ = 8


def heart(x, y, z):
    return 320 * ((-x**2 * z**3 - 9 * y**2 * z**3 / 80) +
                  (x**2 + 9 * y**2 / 4 + z**2 - 1)**3)


@cuda.jit(device=True)
def f_heart(x, y, z):
    return 320 * ((-x**2 * z**3 - 9 * y**2 * z**3 / 80) +
                  (x**2 + 9 * y**2 / 4 + z**2 - 1)**3)


@cuda.jit
def f_heart_kernel(d_f, d_x, d_y, d_z):
    i, j, k = cuda.grid(3)
    nx, ny, nz = d_f.shape
    if i < nx and j < ny and k < nz:
        d_f[i, j, k] = f_heart(d_x[i, j, k], d_y[i, j, k], d_z[i, j, k])


def f_heart3D(X, Y, Z):
    nx, ny, nz = X.shape

    d_x = cuda.to_device(X)
    d_y = cuda.to_device(Y)
    d_z = cuda.to_device(Z)
    d_f = cuda.device_array((nx, ny, nz), dtype=np.float32)

    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY,
                (nz + TPBZ - 1) // TPBZ
                )
    blockDims = (TPBX, TPBY, TPBZ)
    f_heart_kernel[gridDims, blockDims](d_f, d_x, d_y, d_z)
    return d_f.copy_to_host()


@cuda.jit(device=True)
def f_sphere(x, y, z, R):
    return 1 * (x**2 + y**2 + z**2 - R**2)


def sphere(x, y, z, R):
    return 1 * (x**2 + y**2 + z**2 - R**2)


@cuda.jit
def f_sphere_kernel(d_f, d_x, d_y, d_z, R):
    i, j, k = cuda.grid(3)
    nx, ny, nz = d_f.shape
    if i < nx and j < ny and k < nz:
        d_f[i, j, k] = f_sphere(d_x[i, j, k], d_y[i, j, k], d_z[i, j, k], R)


def f_sphere3D(X, Y, Z, R):
    nx, ny, nz = X.shape

    d_x = cuda.to_device(X)
    d_y = cuda.to_device(Y)
    d_z = cuda.to_device(Z)
    d_f = cuda.device_array((nx, ny, nz), dtype=np.float32)

    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY,
                (nz + TPBZ - 1) // TPBZ
                )
    blockDims = (TPBX, TPBY, TPBZ)
    f_sphere_kernel[gridDims, blockDims](d_f, d_x, d_y, d_z, R)
    return d_f.copy_to_host()


def morph(shape_start, shape_end, t):
    n = len(shape_start)
    rst = np.zeros((n, n, n), dtype=np.float32)
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                rst[ix, iy, iz] = shape_start[ix, iy, iz] * (1 - t) + t * shape_end[ix, iy, iz]
    return rst


@cuda.jit
def morph_kernel(d_f, d_x, d_y, t):
    i, j, k = cuda.grid(3)
    nx, ny, nz = d_f.shape
    if i < nx and j < ny and k < nz:
        d_f[i, j, k] = d_x[i, j, k] * (1 - t) + t * d_y[i, j, k]


def morph3D(shape_start, shape_end, t):
    nx, ny, nz = shape_start.shape

    d_x = cuda.to_device(shape_start)
    d_y = cuda.to_device(shape_end)
    d_f = cuda.device_array((nx, ny, nz), dtype=np.float32)

    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY,
                (nz + TPBZ - 1) // TPBZ
                )
    blockDims = (TPBX, TPBY, TPBZ)
    morph_kernel[gridDims, blockDims](d_f, d_x, d_y, t)
    return d_f.copy_to_host()


def exportPLY(filename, verts2, faces):
    plyf = open(filename + '.ply', 'w')
    plyf.write("ply\n")
    plyf.write("format ascii 1.0\n")
    plyf.write("comment ism.py generated\n")
    plyf.write("element vertex " + str(verts2.size // 3) + '\n')
    plyf.write("property float x\n")
    plyf.write("property float y\n")
    plyf.write("property float z\n")
    plyf.write("element face " + str(faces.size // 3) + '\n')
    plyf.write("property list uchar int vertex_indices\n")
    plyf.write("end_header\n")
    for i in range(0, verts2.size // 3):
        plyf.write(str(verts2[i][0]) + ' ' + str(verts2[i][1]) + ' ' + str(verts2[i][2]) + '\n')

    for i in range(0, faces.size // 3):
        plyf.write('3 ' + str(faces[i][0]) + ' ' + str(faces[i][1]) + ' ' + str(faces[i][2]) + '\n')
    plyf.close()
    # end of PLY file write
