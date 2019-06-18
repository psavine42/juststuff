import matplotlib
import matplotlib.patches
from sympy.geometry import Point, Polygon, Line
import numpy as np
import math
from functools import reduce

# class Point:
#    pass
"""
DCL -

vert = Polygon.vertices[1]
edge = Polygon.edges[1]

# everyone can move
# if geometry has new intersections, 
# this must be resolved in the move. 
vert.translate(x=1,y=0,z=0)
edge.translate(x=1,y=0,z=0)

# remove an edge. 
edge.remove()

# add a point, recomputing half edges. 
edge.split(point)

# 

"""


def _line_low(x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    yi = 1
    if dy < 0:
        yi, dy = -1, -dy
    d = 2 * dy - dx
    y = y0
    marked = [[x0, y]]
    for x in range(x0, x1+1):
        marked.append([x, y])
        if d > 0:
            y += yi
            d -= 2 * dx
        d += 2 * dy
    return marked


def _line_high(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    d = 2*dx - dy
    x = x0
    marked = []
    for y in range(y0, y1+1):
        marked.append([x, y])
        if d > 0:
            x += xi
            d -= 2*dy
        d += 2*dx
    return marked


def discretize_segment(p1, p2):
    """from some book lol """
    (x0, y0), (x1, y1) = p1, p2
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            return _line_low(x1, y1, x0, y0)
        return _line_low(x0, y0, x1, y1)
    else:
        if y0 > y1:
            return _line_high(x1, y1, x0, y0)
        return _line_high(x0, y0, x1, y1)


def generate_hatches(size, num):
    """
    generate num planes of size with
    Horizantal, vertical, diag, dot_freq

    todo - good nuff for now, need more bitmaks
    """
    H = 4
    nx, ny = 1 + size[1] // 3, 1 + size[2] // 3

    bmaps = np.zeros((6, 3, 3))

    bmaps[0, 2, :] = 1  # 1) vert
    bmaps[0, 0, :] = 1

    bmaps[1, :, 0] = 1  # 2) horz
    bmaps[1, :, 2] = 1

    bmaps[2, :, 1] = 1  # 3) h and vertical
    bmaps[2, 1, :] = 1

    bmaps[3, :, :] = np.eye(3)  # 4) diag 1
    bmaps[3, 2, 0] = 1
    bmaps[3, 0, 2] = 1

    bmaps[4, :, 0] = 1  # 5) L-shape
    bmaps[4, 0, :] = 1

    bmaps[5, 0, 0] = 1      # 6) box
    bmaps[5, 1:, 1:] = 1

    tiles = np.tile(bmaps, (nx, ny))[:num, :size[1], :size[2]]
    return tiles


def expand_line(mat):
    pass



def ccw_sort(p):
    p = np.array(p)
    mean = np.mean(p, axis=0)
    d = p-mean
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


def perpendicular_vector(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])



def mag(vec):
    return math.sqrt(reduce(lambda x, y: x + y**2, vec))


def unit(vec):
    _mag = mag(vec)
    return [x/_mag for x in vec]


def rot_mat2d(theta):
    return np.asarray([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])


def are_collinear(p1, p2, p3):
    arr = np.asarray([[p1[0] - p2[0], p2[0] - p3[0]],
                      [p1[1] - p2[1], p2[1] - p3[1]]])
    return np.isclose(np.linalg.det(arr), 0.)


class Vertex:
    def __init__(self, point):
        self._point = point
        self._in_edges = []
        self._out_edges = []

    def move(self, to=None):
        pass

    def add_edge(self, edge):
        """ store the """


class HalfEdge:
    def __init__(self, edge, face):
        self._edge = edge
        self._face = face

    def points(self):
        frm, to = self._edge._from, self._edge._to
        return

    def dual(self):
        return self._edge.other(self)

    def __next__(self):
        pass


class Edge:
    def __init__(self, vfrom, vto):
        self._half_edges = []
        self._origin = None
        self._cw_origin = None
        self._ccw_origin = None

        self._target = None
        self._cw_target = None
        self._ccw_target = None
        # add edge
        self._left_face = None
        self._right_face = None

    def other(self, half_edge):
        pass

    def lhs(self):
        pass

    def rhs(self):
        pass

    def add_vertex(self, pt):
        """ """

        pass

    def translate(self, x=0, y=0, z=0):
        self._origin.translate(x=x, y=y, z=z)
        self._target.translate(x=x, y=y, z=z)


class Wire:
    """ a list of half edges """
    def __init__(self):
        pass

    def vertices(self):
        pass

    def edges(self):
        pass


class Face(Polygon):
    def __init__(self, *points):
        self._half_edges = []

    def split(self, line):
        pass

    def edges(self):
        return [x.edge for x in self._half_edges]

    def points(self):
        """ """
        pts = []
        for edge in self._half_edges:
            pts.append(edge.end)
        return pts

    def wires(self):
        return


class Domain(Polygon):
    def __init__(self, *args):
        Polygon.__init__(self, *args)

    def embed_graph(self, g):
        pass




