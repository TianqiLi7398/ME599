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
def f_heart(x, y, z):
    F = 320 * ((-x **2 * z ** 3 - 9 * y ** 2 * z ** 3 / 80) +
               (x **2 + 9 * y ** 2 / 4 + z ** 2 - 1) ** 3)
    return F

def f_cross(x, y):

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

def circle(x, y, R):
    return x ** 2 + y ** 2 - R ** 2

@cuda.jit(device=True)
def circle_par(x, y, R):
    return x ** 2 + y ** 2 - R ** 2

def superellipsoid(x, y, z):
    return x ** 2 / 2 + y ** 2  - 1

def rate(z, h):
    return z / h

def ellipse_x(x, y):
    return (x ** 2 + y ** 2 / 0.25 - 1)

def ellipse_y(x, y):
    return (x ** 2 / 0.25 + y ** 2 - 1)

@cuda.jit(device=True)
def ellipse_x_par(x, y):
    return (x ** 2 + y ** 2 / 0.25 - 1)

@cuda.jit(device=True)
def ellipse_y_par(x, y):
    return (x ** 2 / 0.25 + y ** 2 - 1)

def compose(x, y, t):

    # This function concrete f1 and f2 in z dimension, with rate(z,h) as the parameter
    # w in [0, 1]
    # F = w*f1 + (1-w)*f2
    if t >= 0 and t <= 1:
        return (1 - t) * ellipse_x(x, y) + t * ellipse_y(x, y)
    else:
        return 100

@cuda.jit(device=True)
def compose_par(x, y, t):

    # This function concrete f1 and f2 in z dimension, with rate(z,h) as the parameter
    # w in [0, 1]
    # F = w*f1 + (1-w)*f2
    if t >= 0 and t <= 1:
        return (1 - t) * ellipse_x_par(x, y) + t * ellipse_y_par(x, y)
    else:
        return 100

@cuda.jit
def compose_kernel(d_x, d_y, d_t, d_f):
    i, j, k= cuda.grid(3)
    nx, ny, nt = d_f.shape
    if i < nx and j < ny and k < nt:
        d_f[i,j,k] = compose_par(d_x[i], d_y[j], d_t[k])

# parallel computation of function
def parallel_compose(x, y, t):
    nx, ny, nt= x.shape[0], y.shape[0], t.shape[0]
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_t = cuda.to_device(t)
    d_f = cuda.device_array((nx,ny,nt), dtype=np.float32)

    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY,
                (nt + TPBZ - 1) // TPBZ)
    blockDims = (TPBX, TPBY, TPBZ)
    compose_kernel[gridDims, blockDims](d_x, d_y, d_t, d_f)
    return d_f.copy_to_host()

def compose_serial(x, y, t):
    nx, ny, nt= x.shape[0], y.shape[0], t.shape[0]
    F = np.zeros([nx, ny, nt])

    # creat function based on 3d array, h stands for the height of the aimming 3d
    # model

    for i in range(nx):
        for j in range(ny):
            for k in range(nt):
                F[i, j, k] = compose(x[i], y[j], t[k])
    return F

def main():
    # Set up mesh

    nums = np.linspace(11, 201, 20)
    # time_p, time_s = np.linspace(11, 201, 20), np.linspace(11, 201, 20)
    time_p, time_s = [],[]



    for num in nums:
        m = 1  # min and max coordinate values
        print(num)
        # num = 201  # number of points along each axis

        # Map onto a 3D grid
        xvals = np.linspace(-m, m, num, dtype=np.float32 )
        yvals = np.linspace(-m, m, num,  dtype=np.float32)
        tvals = np.linspace(0, m, num,  dtype=np.float32) # weight

        # main function

        # serial computation
        a = time.time()
        F_serial = compose_serial(xvals, yvals, tvals)
        b = time.time()
        F_parallel = parallel_compose(xvals, yvals, tvals)
        c = time.time()
        time_s.append(b-a)
        time_p.append(c-b)

    line_up, = plt.plot(nums[2:], time_s[2:],'r', label='Serial Computation')
    line_down, = plt.plot(nums[2:], time_p[2:],'b', label='Parallel Computation')
    plt.legend(handles=[line_up, line_down])
    # plt.legend([line_up, line_down], ['Line Up', 'Line Down'])

    # plt.plot(nums, time_s,'r', label='Serial Computation')
    # plt.plot(nums, time_p,'b', label='Parallel Computation')
    plt.ylabel('Computing time/s', fontsize=12)
    plt.xlabel('mesh points', fontsize=12)
    plt.show()

    # include option arg filename='fname' to export in PLY format
    # dx = 2 * m / (num - 1)
    # dy = 2 * m / (num - 1)
    # dt = m / (num - 1)
    # arraycontourplot3d(F_parallel, xvals, yvals, tvals, dx, dy, dt, levels=[-1000, 0],
    #                    titlestring='Testing 2Dmorph arraycontourplot3d', filename='2Dmorph_box_elp_par')




if __name__ == '__main__':
    main()
