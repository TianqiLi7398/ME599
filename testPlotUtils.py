'''
File to test plotting utilities from plotUtils.py
Test functions correspond to superlipse and superellipsoid.
Execution produces (in order):
1) 3d mesh plot
2) 2d contour plot
3) 3d contour plot (and surface tesselation)
'''
import numpy as np
import seaborn
seaborn.set()
from plotUtils import *

# create test functions for superellipse and superellipsoid


def superellipse(x, y, n):
    return x**n + y**n - 1


def superellipsoid(x, y, z, n):
    return x**n + y**n + z**n - 3


def heart(x, y, z):
    return 320 * ((-x**2 * z**3 - 9 * y**2 * z**3 / 80) +
                  (x**2 + 9 * y**2 / 4 + z**2 - 1)**3)


def paraboloid_elip(x, y, z):
    return z - x**2 / 1 - y**2 / 2


def Hyperboloid(x, y, z):
    return z - x**2 / 1 + y ** 2 / 2


def box(x, y, z):
    return x**10 + y ** 10 + z**10 - 1**10


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


def compose(x, y, z, t):
    return t * box(x, y, z) + (1 - t) * paraboloid_elip(x, y, z)


def main():
    m = 3  # min and max coordinate values
    num = 110  # number of points along each axis
    # Map onto a 2D grid
    xvals = np.linspace(-m, m, num)
    yvals = np.linspace(-m, m, num)
    zvals = np.linspace(-m, m, num)

    f4 = np.zeros([num, num, num])

    for t in range(10):
        for k in range(num):
            for j in range(num):
                for i in range(num):
                    f4[k, j, i] = compose(xvals[i], yvals[j], zvals[k], t * 0.1)
        # create contourplot based on 3d array
        # include option arg filename='fname' to export in PLY format
        dx = 2 * m / (num - 1)
        dy = 2 * m / (num - 1)
        dz = 2 * m / (num - 1)
        arraycontourplot3d(f4, xvals, yvals, zvals, dx, dy, dz, levels=[-1000, 0],
                           titlestring='Testing arraycontourplot3d', filename='box_elp%d' % t)

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
