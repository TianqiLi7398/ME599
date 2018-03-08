import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
from skimage import measure
from numba import cuda
from plotUtils import *

# Create cardioid function
def f_heart(x,y,z):
    F = 320 * ((-x**2 * z**3 -9*y**2 * z**3/80) +
               (x**2 + 9*y**2/4 + z**2-1)**3)
    return F



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
    return x**2 - y**2 - R**2

def compose(x,y,z):
    return 0.1*z*circle(x,y, 20) + 0.1*(10 - z)*f_cross_(x,y)

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

def f_square(x,y,R):
    n = len(x)
    F = np.zeros((n,n), dtype=np.float32)
    for ix in range(n):
        for iy in range(n):
            F[ix,iy] = x[ix,iy]**12 + y[ix,iy]**12 - R**2
    return F
# surf = vol

def morph(shape_start, shape_end, t, final_shape):
    n = len(shape_start)
    for iz in range(100):
        it = t[iz]
        for ix in range(n):
            for iy in range(n):
                    final_shape[ix,iy,iz] = shape_start[ix,iy] * (1-it) + it * shape_end[ix,iy]
    return final_shape

def main():
# Set up mesh
    m = 40  # min and max coordinate values
    num = 1000  # number of points along each axis
    # Map onto a 2D grid
    xvals = np.linspace(-m, m, num)
    yvals = np.linspace(-m, m, num)
    zvals = np.linspace(-m, m, num)

    f4 = np.zeros([num, num, num])


    for k in range(num):
        for j in range(num):
            for i in range(num):
                f4[k, j, i] = compose(xvals[i], yvals[j], zvals[k])
    # create contourplot based on 3d array
    # include option arg filename='fname' to export in PLY format
    dx = 2 * m / (num - 1)
    dy = 2 * m / (num - 1)
    dz = 2 * m / (num - 1)
    arraycontourplot3d(f4, xvals, yvals, zvals, dx, dy, dz, levels=[-1000, 0],
                       titlestring='Testing arraycontourplot3d', filename='box_elp')




if __name__ == '__main__':
    main()
