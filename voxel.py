from stl import mesh
import numpy as np
import math


def Get_triangles(your_mesh):
    # this function returns all triangle meshes with form of
    # [...[A,B,C,n]...]
    # A,B,C is three vertices of triangle, n is the norm of the plane
    num = len(your_mesh.v0)
    Triangles = []
    for i in range(num):
        Triangles.append(np.vstack((your_mesh.vectors[i], your_mesh.units[i])))
    X_min = min(your_mesh.v0[:, 0].min(), your_mesh.v1[:, 0].min(), your_mesh.v2[:, 0].min())
    Y_min = min(your_mesh.v0[:, 1].min(), your_mesh.v1[:, 1].min(), your_mesh.v2[:, 1].min())
    Z_min = min(your_mesh.v0[:, 2].min(), your_mesh.v1[:, 2].min(), your_mesh.v2[:, 2].min())
    X_max = max(your_mesh.v0[:, 0].max(), your_mesh.v1[:, 0].max(), your_mesh.v2[:, 0].max())
    Y_max = max(your_mesh.v0[:, 1].max(), your_mesh.v1[:, 1].max(), your_mesh.v2[:, 1].max())
    Z_max = max(your_mesh.v0[:, 2].max(), your_mesh.v1[:, 2].max(), your_mesh.v2[:, 2].max())

    return Triangles, [[X_min, Y_min, Z_min], [X_max, Y_max, Z_max]]


def getBlock(triangle):
    X_min = triangle[:, 0][0:-1].min()
    Y_min = triangle[:, 1][0:-1].min()
    Z_min = triangle[:, 2][0:-1].min()
    X_max = triangle[:, 0][0:-1].max()
    Y_max = triangle[:, 1][0:-1].max()
    Z_max = triangle[:, 2][0:-1].max()

    return [[X_min, Y_min, Z_min], [X_max, Y_max, Z_max]]


def cut(XYZ, edge):
    [[X_min, Y_min, Z_min], [X_max, Y_max, Z_max]] = XYZ
    XYZ_MIN = [X_min // edge, Y_min // edge, Z_min // edge]
    range_x = X_max // edge - X_min // edge
    range_y = Y_max // edge - Y_min // edge
    range_z = Z_max // edge - Z_min // edge
    return XYZ_MIN, [range_x, range_y, range_z]


def getVertical(line_fun, point):
    # this function will get the vertical point of one point to a line in 3D model
    # line_fun is a array of (m,n,p,x0,y0) which is the line function
    # (x-x0)/m = (y-y0)/n = z/p
    # point is the coordinate of point, (x,y,z)
    [m, n, p, point_line] = line_fun
    x1, y1, z1 = point
    # (AB-xn)n = 0
    AB = [point_line[0] - x1, point_line[1] - y1, point_line[2] - z1]
    x = np.dot(AB, [m, n, p]) / (m**2 + n**2 + p**2)

    vertical = np.array(AB) - np.array([x * m, x * n, x * p])
    return vertical


def getInterLine(square, triangle):
    # this function gets the intersection line function of the square plane and triangle plane
    # and will return the function
    # line_fun is a array of (m,n,p,x0,y0) which is the line function
    # (x-x0)/m = (y-y0)/n = z/p
    A1, B1, C1, n1 = triangle
    A2, B2, C2, D2, n2 = square
    m = np.cross(n1, n2)
    if np.linalg.norm(m) == 0:
        return None
    # normalize
    m = m / np.linalg.norm(m)
    # get two planes function
    # Ax + By + Cz = d
    d1 = np.dot(A1, n1)
    d2 = np.dot(A2, n2)
    # find cube square's direction, cuz one dimension is hidden in Cartesian coordinate system
    n2_sq = np.multiply(n2, n2)
    dimension = np.where(np.array(n2_sq) == 1.0)[0][0]
    # print dimension

    if dimension == 0:
        if n1[2] != 0.0:
            point = [d2, 1, (d1 - n1[0] * d2 - n1[1]) / n1[2]]
        else:
            point = [d2, (d1 - n1[0] * d2) / n1[1], 0]
    elif dimension == 1:
        if n1[2] != 0.0:
            point = [1, d2, (d1 - n1[1] * d2 - n1[0]) / n1[2]]
        else:
            point = [(d1 - n1[1] * d2) / n1[0], d2, 0]
    elif dimension == 2:
        if n1[1] != 0:
            point = [1, (d1 - n1[2] * d2 - n1[0]) / n1[1], d2]
        else:
            point = [(d1 - n1[2] * d2) / n1[0], 0, d2]
    else:
        print("error in getInterLine")

    return [m[0], m[1], m[2], point]


def getSXSquare(x, y, z, edge):
    #     this function generates the function of sX square surfaces of a cube, with it's min point
    #     at (x,y,z) and edge
    front = [(x, y, z), (x + edge, y, z), (x, y, z + edge), (x + edge, y, z + edge), (0, -1, 0)]
    back = [(x, y + edge, z), (x + edge, y + edge, z), (x, y + edge,
                                                        z + edge), (x + edge, y + edge, z + edge), (0, 1, 0)]

    left = [(x, y, z), (x, y + edge, z), (x, y, z + edge), (x, y + edge, z + edge), (-1, 0, 0)]
    right = [(x + edge, y, z), (x + edge, y + edge, z), (x + edge, y,
                                                         z + edge), (x + edge, y + edge, z + edge), (1, 0, 0)]

    down = [(x, y, z), (x + edge, y, z), (x, y + edge, z), (x + edge, y + edge, z), (0, 0, -1)]
    up = [(x, y, z + edge), (x + edge, y, z + edge), (x, y + edge,
                                                      z + edge), (x + edge, y + edge, z + edge), (0, 0, 1)]

    return [front, back, left, right, down, up]


your_mesh = mesh.Mesh.from_file('models/sphere.stl')
edge = 0.1
triangles, XYZ_ALL = Get_triangles(your_mesh)
# print(XYZ_ALL)
XYZ_ALL_MIN, [range_x_all, range_y_all, range_z_all] = cut(XYZ_ALL, edge)
cube = np.zeros((int(range_x_all) + 1, int(range_y_all) + 1, int(range_z_all) + 1))
# print(XYZ_ALL_MIN, [range_x_all, range_y_all, range_z_all])
a = 0

for triangle in triangles:
    a += 1
    # print(a)
    XYZ = getBlock(triangle)
    XYZ_MIN, [range_x, range_y, range_z] = cut(XYZ, edge)  # let X,Y,Z suit for block size

    # print(XYZ_MIN, [range_x, range_y, range_z])
    if (range_x == 0.0 and range_y == 0.0) or (range_x == 0.0 and range_z == 0.0) or (range_y == 0.0 and range_z == 0.0) or (range_x == 0.0 and range_y == 0.0 and range_z == 0.0):
        for x in range(int(range_x) + 1):
            for y in range(int(range_y) + 1):
                for z in range(int(range_z) + 1):
                    # print(1, [int(XYZ_MIN[0]) + x, int(XYZ_MIN[1]) + y, int(XYZ_MIN[2]) + z])
                    cube[int(XYZ_MIN[0] - XYZ_ALL_MIN[0]) + x, int(XYZ_MIN[1] -
                                                                   XYZ_ALL_MIN[1]) + y, int(XYZ_MIN[2] - XYZ_ALL_MIN[2]) + z] = 1
    else:
        for x in range(int(range_x) + 1):
            for y in range(int(range_y) + 1):
                for z in range(int(range_z) + 1):
                    # generate all surfaces of the cube
                    Squares = getSXSquare(x * edge, y * edge, z * edge, edge)
                    for square in Squares:
                        line_fun = getInterLine(square, triangle)
                        if line_fun == None:
                            # the two plane is parallel
                            break
                        vertical_line_triangle = []
                        for point_tr in triangle[0: -1]:
                            vertical_line_triangle.append(getVertical(line_fun, point_tr))
                        if np.dot(vertical_line_triangle[0], vertical_line_triangle[1]) <= 0 or np.dot(vertical_line_triangle[2], vertical_line_triangle[1]) <= 0:
                            vertical_line_square = []
                            for point_sq in square[0: -1]:
                                vertical_line_square.append(getVertical(line_fun, point_sq))
                            if np.dot(vertical_line_square[0], vertical_line_square[1]) <= 0 or \
                                    np.dot(vertical_line_square[2], vertical_line_square[1]) <= 0 or \
                                    np.dot(vertical_line_square[3], vertical_line_square[2]) <= 0 or \
                                    np.dot(vertical_line_square[3], vertical_line_square[1]) <= 0 or \
                                    np.dot(vertical_line_square[0], vertical_line_square[3]) <= 0 or \
                                    np.dot(vertical_line_square[0], vertical_line_square[2]) <= 0:
                                cube[int(XYZ_MIN[0] - XYZ_ALL_MIN[0]) + x, int(XYZ_MIN[1] -
                                                                               XYZ_ALL_MIN[1]) + y, int(XYZ_MIN[2] - XYZ_ALL_MIN[2]) + z] = 1
# print cube.sum()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


N = edge
nx, ny, nz = cube.shape
whole_cube = np.asarray([], dtype=mesh.Mesh.dtype)
test1 = None
sample1 = None
for ix in range(nx):
    for iy in range(ny):
        for iz in range(nz):
            if cube[ix, iy, iz]:
                X = N * ix
                Y = N * iy
                Z = N * iz
                data = np.zeros(12, dtype=mesh.Mesh.dtype)
                # top
                data['vectors'][0] = np.array([[X, Y, Z + N],
                                               [X + N, Y, Z + N],
                                               [X, Y + N, Z + N]])
                data['vectors'][1] = np.array([[X + N, Y + N, Z + N],
                                               [X + N, Y, Z + N],
                                               [X, Y + N, Z + N]])
                # bottom
                data['vectors'][2] = np.array([[X, Y, Z],
                                               [X + N, Y, Z],
                                               [X, Y + N, Z]])
                data['vectors'][3] = np.array([[X + N, Y + N, Z],
                                               [X + N, Y, Z],
                                               [X, Y + N, Z]])

                # right
                data['vectors'][4] = np.array([[X + N, Y, Z],
                                               [X + N, Y, Z + N],
                                               [X + N, Y + N, Z + N]])
                data['vectors'][5] = np.array([[X + N, Y + N, Z],
                                               [X + N, Y, Z],
                                               [X + N, Y + N, Z + N]])

                # right
                data['vectors'][6] = np.array([[X, Y, Z],
                                               [X, Y, Z + N],
                                               [X, Y + N, Z]])
                data['vectors'][7] = np.array([[X, Y, Z + N],
                                               [X, Y + N, Z + N],
                                               [X, Y + N, Z]])

                # left
                data['vectors'][8] = np.array([[X, Y, Z],
                                               [X + N, Y, Z],
                                               [X, Y, Z + N]])
                data['vectors'][9] = np.array([[X + N, Y, Z],
                                               [X, Y, Z + N],
                                               [X + N, Y, Z + N]])

                # left
                data['vectors'][10] = np.array([[X + N, Y + N, Z + N],
                                                [X + N, Y + N, Z],
                                                [X, Y + N, Z + N]])
                data['vectors'][11] = np.array([[X, Y + N, Z],
                                                [X + N, Y + N, Z],
                                                [X, Y + N, Z + N]])

                cube_back = mesh.Mesh(data.copy())
                # cube_front = mesh.Mesh(data.copy())

                # cube_back.rotate([0.5, 0.0, 0.0], math.radians(90))
                # cube_back.rotate([0.0, 0.5, 0.0], math.radians(90))
                # cube_back.rotate([0.5, 0.0, 0.0], math.radians(90))
                if test1 is None:
                    # print(X, Y, Z)
                    sample1 = cube_back
                    test1 = np.concatenate([cube_back.data.copy()
                                            # cube_front.data.copy(),
                                            ])
                whole_cube = np.concatenate([whole_cube,
                                             cube_back.data.copy()
                                             # cube_front.data.copy(),
                                             ])
                # x_s = np.linspace(, 2, N, dtype=np.float32)
                # y_s = np.linspace(-2, 2, N, dtype=np.float32)
                # dis = prob3_a(x_s, y_s, N)
                #
                # fig = plt.figure(4)
                # ax = fig.add_subplot(111, projection='3d')
                # ax.set_title('Problem 3a: Distance grid u for set of points: (-1,0),(1,0),(0,1)')
                # ax.set_xlabel('x')
                # ax.set_ylabel('y')
                # x, y = np.meshgrid(x_s, y_s)
                # ax.plot_wireframe(x, y, dis, color='red')
                # plt.show()
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
# final_cube = mesh.Mesh(test1)
final_cube = mesh.Mesh(whole_cube)
fig = plt.figure()
axes = mplot3d.Axes3D(fig)

axes.add_collection3d(mplot3d.art3d.Poly3DCollection(final_cube.vectors))
# scale = sample1.points.flatten(-1)
scale = final_cube.points.flatten(-1)
axes.auto_scale_xyz(scale, scale, scale)
# scale =
plt.show()
