'''
March 8, 2018
Project of Voxel Model ME 599
Tianqi Li, Junlin Li

This file implements CUDA to generate .ply files of two 2d functions f1(x,y), f2(x,y)
and create a new 3d geomentry by
F(x,y,z) = z/h*f1(x,y) + (1-z/h) f2(x,y)

in Z-dimension, h is the size of the newly generated geomentry.
'''

import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
from skimage import measure
from numba import cuda
from plotUtils import *
from numba import cuda
import time

TPBX = 8
TPBY = 8
TPBZ = 8

# Create cardioid function
def f_heart(x,y,z):
    F = 320 * ((-x**2 * z**3 -9*y**2 * z**3/80) +
               (x**2 + 9*y**2/4 + z**2-1)**3)
    return F

def f_cross_(x, y):

    # F = np.zeros((n,n), dtype=np.float32)

    if (abs(abs(x) - 40.0) < 1e-16):
        if (abs(y) <= 9.0):
            F = 0.0
        else:
            F = 40.0

    elif (9.0 < abs(x) < 40.0):
        if (abs(9.0 - abs(y)) < 1e-16):
            F = 0.0
        else:
            F = 40.0

    elif (abs(abs(x) - 9.0) < 1e-16):
        if (9.0 <= abs(y) <= 40.0):
            F = 0.0
        else:
            F = 40.0

    elif (abs(x) < 9.0):
        if (abs(abs(y) - 40.0) < 1e-16):
            F = 0.0
        else:
            F = 40.0
    return F

def circle(x,y, R):
    return x**2 + y**2 - R**2

def superellipsoid(x, y, z):
    return x**2/2 + y**2  - 1

def rate(z, h):
    return z/h


def compose(x,y,z,h):

    # This function concrete f1 and f2 in z dimension, with rate(z,h) as the parameter
    # w in
    # F = w*f1 + (1-w)*f2
    R = 2
    if z >= 0 and z <= h:
        return rate(z,h)*circle(x,y, h) + (1 - rate(z,h))*superellipsoid(x, y, z)
    else:
        return 100

@cuda.jit(device=True)
def compose_par(x,y,z,h):

    # This function concrete f1 and f2 in z dimension, with rate(z,h) as the parameter
    # w in
    # F = w*f1 + (1-w)*f2

    rate = z/h
    if z >= 0 and z <= h:
        return rate*(x**2 + y**2 - 4**2) + (1 - rate)*(x**2/2 + y**2  - 1)
    else:
        return 100

@cuda.jit(device=True)
def circle_par(x,y,R):
    return x**2 + y**2 - R**2

# parallel computation of function
def parallel_compose(x, y, z, h):
    nx, ny, nz= x.shape[0], y.shape[0], z.shape[0]
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_z = cuda.to_device(z)
    d_f = cuda.device_array((nx,ny,nz), dtype=np.float32)

    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY,
                (nz + TPBZ - 1) // TPBZ)
    blockDims = (TPBX, TPBY, TPBZ)
    compose_kernel[gridDims, blockDims](d_x, d_y, d_z, d_f, h)
    return d_f.copy_to_host()

@cuda.jit
def compose_kernel(d_x, d_y, d_z, d_f, h):
    i, j, k= cuda.grid(3)
    nx, ny, nz = d_f.shape
    if i < nx and j < ny and k < nz:
        d_f[i,j,k] = compose_par(d_x[i], d_y[j], d_z[k], h)

def compose_serial(x, y, z, h):
    nx, ny, nz= x.shape[0], y.shape[0], z.shape[0]
    F = np.zeros([nx, ny, nz])

    # creat function based on 3d array, h stands for the height of the aimming 3d
    # model

    for k in range(nx):
        for j in range(ny):
            for i in range(nz):
                F[k, j, i] = compose(x[i], y[j], z[k], h)
    return F

def main():
    # Set up mesh

    m = 5  # min and max coordinate values
    num = 201  # number of points along each axis

    # Map onto a 3D grid
    xvals = np.linspace(-m, m, num)
    yvals = np.linspace(-m, m, num)
    zvals = np.linspace(-m, m, num)

    # main function

    h = m
    # serial computation
    a = time.time()
    # F_serial = compose_serial(xvals, yvals, zvals, h)
    b = time.time()
    F_parallel = parallel_compose(xvals, yvals, zvals, h)
    c = time.time()

    # include option arg filename='fname' to export in PLY format
    dx = 2 * m / (num - 1)
    dy = 2 * m / (num - 1)
    dz = 2 * m / (num - 1)
    # print("serial time: %f, parallel time: %f"%(b-a, c-b))
    # arraycontourplot3d(F_serial, xvals, yvals, zvals, dx, dy, dz, levels=[-1000, 0],
    #                    titlestring='Testing arraycontourplot3d', filename='box_elp_ser')
    arraycontourplot3d(F_parallel, xvals, yvals, zvals, dx, dy, dz, levels=[-1000, 0],
                       titlestring='Testing arraycontourplot3d', filename='box_elp_par')




if __name__ == '__main__':
    main()
