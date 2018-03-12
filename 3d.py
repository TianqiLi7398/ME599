'''
March 8, 2018
Project of Voxel Model ME 599
Tianqi Li, Junlin Li

This file implements CUDA to generate .ply files of two 3d functions f1(x,y,z), f2(x,y,z)
and create a new 3d geomentry by
F(x,y,z) = t*f1(x,y) + (1-t) f2(x,y)

t is the size of the newly generated geomentry.
'''

import numpy as np
import time
from numba import cuda
import seaborn
seaborn.set()
from plotUtils import *

# create test functions for superellipse and superellipsoid
TPBX = 8
TPBY = 8
TPBZ = 8

def superellipse(x, y, n):
    return x**n + y**n - 1


def superellipsoid(x, y, z, n):
    return x**n + y**n + z**n - 3


def heart(x, y, z):
    return 320 * ((-x**2 * z**3 - 9 * y**2 * z**3 / 80) +
                  (x**2 + 9 * y**2 / 4 + z**2 - 1)**3)


def paraboloid_elip(x, y, z):
    return z**2 / (0.5 ** 2) + x**2 / 1 + y**2 / (0.5 ** 2) -1


def Hyperboloid(x, y, z):
    return z - x**2 / 1 + y ** 2 / 2


def box(x, y, z):
    return x**10 + y ** 10 + z**10 - 1


def first(x, y, z):
    return x**3 + y**2 - z**2


def gravity(x, y, z):
    G = 6.674e-11
    M = 1989100e24
    m = 5.9736e24
    Ear = 92.96  # unit million miles
    sun = -G * M * m / np.sqrt(x**2 + y**2 + z**2)
    earth = - G * M * m / np.sqrt((x - 92.96)**2 + y**2 + z**2)

    c = -G * M * m / np.sqrt(50**2 + 0**2 + 0**2) - G * M * m / \
        np.sqrt((50 - 92.96)**2 + 0**2 + 0**2)
    return sun + earth - c


def compose_ser(x, y, z, tvals):
    nx, ny, nz= x.shape
    F = np.zeros((nx, ny, nz), dtype=np.float32)

    F_total = []

    for t in tvals:
        print(t)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    F[i,j,k] = (1 - t) * paraboloid_elip(x[i,j,k], y[i,j,k], z[i,j,k])  + t * box(x[i,j,k], y[i,j,k], z[i,j,k])
        F_total.append(F)
    return F_total

# parallel cube
@cuda.jit(device=True)
def f_cube(x,y,z,L):
    return x ** 10 + y ** 10 + z ** 10 - L

@cuda.jit
def f_cube_kernel(d_f, d_x, d_y, d_z, L):
    i, j, k = cuda.grid(3)
    nx, ny, nz = d_f.shape
    if i < nx and j < ny and k < nz:
        d_f[i,j,k] = f_cube(d_x[i,j,k], d_y[i,j,k], d_z[i,j,k], L)

def f_cube3D(X, Y, Z, L):
    nx, ny, nz = X.shape

    d_x = cuda.to_device(X)
    d_y = cuda.to_device(Y)
    d_z = cuda.to_device(Z)
    d_f = cuda.device_array((nx,ny,nz), dtype=np.float32)

    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY,
                (nz + TPBZ - 1) // TPBZ
                )
    blockDims = (TPBX, TPBY, TPBZ)
    f_cube_kernel[gridDims, blockDims](d_f, d_x, d_y, d_z, L)
    return d_f.copy_to_host()

# parallel ellipsoid
@cuda.jit(device=True)
def f_ellipsoid(x,y,z,L):
    # return 1 * (x**2 + y**2 + z**2 - R**2)
    return z**2 / (0.5 ** 2) + x**2 / 1 + y**2 / (0.5 ** 2) - L

@cuda.jit
def f_ellipsoid_kernel(d_f, d_x, d_y, d_z, R):
    i, j, k = cuda.grid(3)
    nx, ny, nz = d_f.shape
    if i < nx and j < ny and k < nz:
        d_f[i,j,k] = f_sphere(d_x[i,j,k], d_y[i,j,k], d_z[i,j,k], R)

def f_ellipsoid3D(X, Y, Z, R):
    nx, ny, nz = X.shape

    d_x = cuda.to_device(X)
    d_y = cuda.to_device(Y)
    d_z = cuda.to_device(Z)
    d_f = cuda.device_array((nx,ny,nz), dtype=np.float32)

    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY,
                (nz + TPBZ - 1) // TPBZ
                )
    blockDims = (TPBX, TPBY, TPBZ)
    f_ellipsoid_kernel[gridDims, blockDims](d_f, d_x, d_y, d_z, R)
    return d_f.copy_to_host()

# parallel morph
@cuda.jit
def morph_kernel(d_f, d_x, d_y, d_z, t, C):
    i, j, k = cuda.grid(3)
    nx, ny, nz = d_f.shape
    if i < nx and j < ny and k < nz:
        d_f[i,j,k] = f_cube(d_x[i,j,k], d_y[i,j,k], d_z[i,j,k], C) * (t) + (1-t) * f_ellipsoid(d_x[i,j,k], d_y[i,j,k], d_z[i,j,k], C)

def morph3D(x, y, z, t, C):
    nx, ny, nz = x.shape

    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_z = cuda.to_device(z)
    d_f = cuda.device_array((nx,ny,nz), dtype=np.float32)

    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY,
                (nz + TPBZ - 1) // TPBZ
                )
    blockDims = (TPBX, TPBY, TPBZ)
    morph_kernel[gridDims, blockDims](d_f, d_x, d_y, d_z, t, C)
    return d_f.copy_to_host()

def main():
    m = 10  # min and max coordinate values
    num = 201  # number of points along each axis
    # Map onto a 2D grid
    xvals = np.linspace(-m, m, num)
    yvals = np.linspace(-m, m, num)
    zvals = np.linspace(-m, m, num)
    tvals = np.linspace(0,1,11)
    X, Y, Z =  np.meshgrid(xvals, yvals, zvals)

    # F_ser = compose_ser(xvals, yvals, zvals, tvals)

    L = 1.0
    dx = 2 * m / (num - 1)
    dy = 2 * m / (num - 1)
    dz = 2 * m / (num - 1)

    # serial calculation

    a = time.time()

    F_ser = compose_ser(X, Y, Z, tvals)

    b = time.time()

    F_par = []

    for t in tvals:
        surf = morph3D(X, Y, Z, t, L)
        F_par.append(surf)
    c = time.time()

    print("serial computation: %d, parallel computation: %d" %(b-a, c-b))



    for i in range(len(tvals)):
        arraycontourplot3d(F_ser[i], xvals, yvals, zvals, dx, dy, dz, levels=[-1000, 0],
                   titlestring='Testing arraycontourplot3d', filename='box_elp_ser%d' % i)
        arraycontourplot3d(F_par[i], xvals, yvals, zvals, dx, dy, dz, levels=[-1000, 0],
                   titlestring='Testing arraycontourplot3d', filename='box_elp_par%d' % i)


    #
    # for it in tvals:
    #     surf = morph3D(X, Y, Z, it, L)
    #     i = it*10
    #     arraycontourplot3d(surf, xvals, yvals, zvals, dx, dy, dz, levels=[-1000, 0],
    #                titlestring='Testing arraycontourplot3d', filename='box_elp_par%d' % i)

    # i = 1

    # for i in range(len(tvals)):
    #     arraycontourplot3d(F_ser[i], xvals, yvals, zvals, dx, dy, dz, levels=[-1000, 0],
    #                titlestring='Testing arraycontourplot3d', filename='box_elp%d' % i)

    '''

    # create 2d array and fill with function evaluated on 2d grid
    f2 = np.zeros([num,num])
    for j in range(num):
            for i in range(num):
                f2[j,i] = superellipse(xvals[i],yvals[j],6)

    # create 3d plot of the array
    # plot3d(f2,xvals, yvals,titlestring='Testing plot3d')

    # create contourplot based on 2d array
    # arraycontourplot(f2,xvals, yvals, levels=[-1000,0],
        # titlestring='Testing arraycontourplot', filled=True)

    # map onto 3d grid

    zvals = np.linspace(-m,m,num)
    f1 = np.zeros([num,num,num])
    for k in range(num):
        for j in range(num):
            for i in range(num):
                f1[k,j,i] = superellipsoid(xvals[i],yvals[j],zvals[k],2)
    # create contourplot based on 3d array
    # include option arg filename='fname' to export in PLY format
    dx = 2*m/(num-1)
    dy = 2*m/(num-1)
    dz = 2*m/(num-1)
    arraycontourplot3d(f1,xvals, yvals, zvals,dx,dy,dz, levels=[-1000,0],
        titlestring='Testing arraycontourplot3d')


    zvals = np.linspace(-m,m,num)
    f2 = np.zeros([num,num,num])
    for k in range(num):
        for j in range(num):
            for i in range(num):
                f2[k,j,i] = first(xvals[i],yvals[j],zvals[k])
    # create contourplot based on 3d array
    # include option arg filename='fname' to export in PLY format
    dx = 2*m/(num-1)
    dy = 2*m/(num-1)
    dz = 2*m/(num-1)
    arraycontourplot3d(f2,xvals, yvals, zvals,dx,dy,dz, levels=[-1000,0],
        titlestring='Testing arraycontourplot3d')


    zvals = np.linspace(-m,m,num)
    f3 = np.zeros([num,num,num])
    for k in range(num):
        for j in range(num):
            for i in range(num):
                f3[k,j,i] = compose(xvals[i],yvals[j],zvals[k],0.5)
    # create contourplot based on 3d array
    # include option arg filename='fname' to export in PLY format
    dx = 2*m/(num-1)
    dy = 2*m/(num-1)
    dz = 2*m/(num-1)
    arraycontourplot3d(f3,xvals, yvals, zvals,dx,dy,dz, levels=[-1000,0],
        titlestring='Testing arraycontourplot3d')


    # create contourplot based on 3d array
    # include option arg filename='fname' to export in PLY format
    dx = 2*m/(num-1)
    dy = 2*m/(num-1)
    dz = 2*m/(num-1)
    arraycontourplot3d(f2,xvals, yvals, zvals,dx,dy,dz, levels=[-1000,0],
        titlestring='Testing arraycontourplot3d')
'''


if __name__ == '__main__':
    main()
