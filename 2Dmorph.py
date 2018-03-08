import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
from skimage import measure
from numba import cuda

TPBX = 8
TPBY = 8
TPBZ = 8

# Obtain value to at every point in mesh
# vol = f_heart(X,Y,Z)
# a cube
def f_cross(x, y):
    n = len(x)
    F = np.zeros((n,n), dtype=np.float32)
    for ix in range(n):
        for iy in range(n):
            if (abs(abs(x[ix,iy]) - 40.0) < 1e-16):
                if (abs(y[ix,iy]) <= 9.0):
                    F[ix, iy] = 0.0
                else:
                    F[ix, iy] = 40.0
            
            elif (9.0 < abs(x[ix,iy]) < 40.0):
                if (abs(9.0 - abs(y[ix, iy])) < 1e-16):
                    F[ix, iy] = 0.0
                else:
                    F[ix, iy] = 40.0

            elif (abs(abs(x[ix,iy]) - 9.0) < 1e-16):
                if (9.0 <= abs(y[ix,iy]) <= 40.0):
                    F[ix, iy] = 0.0
                else:
                    F[ix, iy] = 40.0
            
            elif (abs(x[ix,iy]) < 9.0):
                if (abs(abs(y[ix, iy]) - 40.0) < 1e-16):
                    F[ix, iy] = 0.0
                else:
                    F[ix, iy] = 40.0
            else:
                F[ix, iy] = 40.0
    return F

@cuda.jit
def cross_kernel(d_f, d_x, d_y):
    i, j = cuda.grid(2)
    nx, ny = d_f.shape
    if i < nx and j < ny:
        if (abs(abs(d_x[i,j]) - 40.0) < 1e-16):
                if (abs(d_y[i,j]) <= 9.0):
                    d_f[i,j] = 0.0
                else:
                    d_f[i,j] = 40.0
            
            elif (9.0 < abs(d_x[i,j]) < 40.0):
                if (abs(9.0 - abs(d_y[i,j])) < 1e-16):
                    d_f[i,j] = 0.0
                else:
                    d_f[i,j] = 40.0

            elif (abs(abs(d_x[i,j]) - 9.0) < 1e-16):
                if (9.0 <= abs(d_y[i,j]) <= 40.0):
                    d_f[i,j] = 0.0
                else:
                    d_f[i,j] = 40.0
            
            elif (abs(d_x[i,j]) < 9.0):
                if (abs(abs(d_y[i,j]) - 40.0) < 1e-16):
                    d_f[i,j] = 0.0
                else:
                    d_f[i,j] = 40.0
            else:
                d_f[i,j] = 40.0

def f_cross_p(X, Y):
    nx, ny= X.shape

    d_x = cuda.to_device(X)
    d_y = cuda.to_device(Y)
    d_f = cuda.device_array((nx,ny), dtype=np.float32)

    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY
                )
    blockDims = (TPBX, TPBY)
    cross_kernel[gridDims, blockDims](d_f, d_x, d_y)
    return d_f.copy_to_host()

def arraycontourplot(fvals, xvals, yvals, levels=[-1000,0], vars=['x','y'], 
    titlestring='', filled=False):
    fig = plt.figure()
    X,Y = np.meshgrid(xvals,yvals)
    if filled==True:
        cp = plt.contourf(X, Y, fvals, levels) #, linestyles='dashed')
    else:
        cp = plt.contour(X, Y, fvals, levels) #, linestyles='dashed')
    # plt.clabel(cp, inline=True, fontsize=10)
    plt.title(titlestring)
    plt.xlabel(vars[0])
    plt.ylabel(vars[1])
    plt.axis('square')
    plt.show()
    return cp

def f_cycle(x,y,R):
    n = len(x)
    F = np.zeros((n,n), dtype=np.float32)
    for ix in range(n):
        for iy in range(n):
            F[ix,iy] = x[ix,iy]**2 + y[ix,iy]**2 - R**2
    return F

@cuda.jit(device=True)
def cycle_f(x,y,R):
    return x**2 + y**2 - R**2

@cuda.jit
def cycle_kernel(d_f, d_x, d_y, R):
    i, j = cuda.grid(2)
    nx, ny = d_f.shape
    if i < nx and j < ny:
        d_f[i,j] = cycle_f(d_x[i,j], d_y[i,j], R)

def f_cycle_p(X, Y, R):
    nx, ny= X.shape

    d_x = cuda.to_device(X)
    d_y = cuda.to_device(Y)
    d_f = cuda.device_array((nx,ny), dtype=np.float32)

    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY
                )
    blockDims = (TPBX, TPBY)
    cycle_kernel[gridDims, blockDims](d_f, d_x, d_y, R)
    return d_f.copy_to_host()

def morph(shape_start, shape_end, t, final_shape):
    n = len(shape_start)
    for iz in range(len(t)):
        it = t[iz]
        for ix in range(n):
            for iy in range(n):
                    final_shape[ix,iy,iz] = shape_start[ix,iy] * (1-it) + it * shape_end[ix,iy]
    return final_shape

@cuda.jit(device=True)
def morph_f(x,y,t):
    return x * (1 - t) + y * t

@cuda.jit
def morph_kernel(d_f, d_x, d_y, T):
    i, j, t = cuda.grid(3)
    nx, ny, nt = d_f.shape
    if i < nx and j < ny and t < nt:
        d_f[i,j,t] = morph_f(d_x[i,j], d_y[i,j], T[t])

def f_morph_p(X, Y, T):
    nx, ny= X.shape
    nt = len(T)
    d_x = cuda.to_device(X)
    d_y = cuda.to_device(Y)
    d_f = cuda.device_array((nx,ny,nt), dtype=np.float32)

    gridDims = ((nx + TPBX - 1) // TPBX,
                (ny + TPBY - 1) // TPBY,
                (nt + TPBZ - 1) // TPBZ
                )
    blockDims = (TPBX, TPBY, TPBZ)
    morph_kernel[gridDims, blockDims](d_f, d_x, d_y, T)
    return d_f.copy_to_host()

def main():
# Set up mesh
    n = 821
    m = 41

    x = np.linspace(-m,m,n)
    y = np.linspace(m,-m,n)
    X, Y=  np.meshgrid(x, y)
    cross = f_cross_p(X, Y)
    # arraycontourplot(cross, x, y, levels=[-1,1])
    R = 41.0
    x_cycle = np.linspace(-m,m,n)
    y_cycle = np.linspace(m,-m,n)
    X_cycle, Y_cycle = np.meshgrid(x_cycle, y_cycle)
    cycle = f_cycle_p(X_cycle, Y_cycle, R)
    nt = 100
    t = np.linspace(0,10,nt)
    morph_shape = f_morph_p(cross, cycle, t)
    surf = morph_shape

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

if __name__ == '__main__':
    main()
