import cvxpy as cvx
from cvxpy import Variable
import cvxpy.lin_ops
import cvxpy.utilities
from .logical import *
import math
import scipy.spatial
import itertools
import numpy as np
import src.cvopt.constraining as cnstr
from .mesh.mesh2d import *


def rectangle(w, h):
    return list(itertools.product(range(0, w), range(0, h)))


def union_shapes(sh1, sh2):
    s1 = set([tuple([x, y]) for x, y in sh1])
    for shp in sh2:
        s1 = s1.union(set([tuple([x, y]) for x, y in shp]))
    return list(s1)


def diff_shapes(sh1, *sh2):
    s1 = set([tuple([x, y]) for x, y in sh1])
    for shp in sh2:
        s1 = s1.difference(set([tuple([x, y]) for x, y in shp]))
    return list(s1)


def verts_to_edges(coords):
    base = [(coords[i], coords[i+1]) for i in range(len(coords)-1)]
    return base + [(coords[-1], coords[0])]


class _VarGraphBase(object):
    def __init__(self):
        self._faces = []
        self._verts = []
        self._edges = []
        self._half_edges = []
        self.X = None

    @property
    def value(self):
        if isinstance(self.X, Variable):
            return self.X.value

    def explain(self, **kwargs):
        return

    def adj_vars(self, face=True, edge=False, vert=False, half_edge=False):
        if face is True:
            return [x.X for x in self._faces]
        elif edge is True:
            return [x.X for x in self._edges]
        elif vert is True:
            return [x.X for x in self._verts]
        elif half_edge is True:
            return [x.X for x in self._half_edges]

    def connect(self, ent):
        if isinstance(ent, HalfEdge):
            self._half_edges.append(ent)
        elif isinstance(ent, Vertex):
            self._verts.append(ent)
        elif isinstance(ent, Face):
            self._faces.append(ent)
        elif isinstance(ent, Edge):
            self._edges.append(ent)


# ------------------------------------------------------------------
class Vertex(Mesh2d.Vertex, _VarGraphBase):
    def __init__(self, *args, n_colors=0, **kwargs):
        _VarGraphBase.__init__(self)
        Mesh2d.Vertex.__init__(self, *args, **kwargs)
        self.X = Variable(n_colors, boolean=True, name='vert.{}'.format(self.index))

    @property
    def constraints(self):
        return []


# ------------------------------------------------------------------
class HalfEdge(Mesh2d.HalfEdge, _VarGraphBase):
    """


    """
    def __init__(self, *args, n_colors=0, **kwargs):
        _VarGraphBase.__init__(self)
        Mesh2d.HalfEdge.__init__(self, *args, **kwargs)
        self.X = Variable(n_colors, boolean=True, name='hedg.{}'.format(self.index))
        self._map = dict()
        self._hard_map = dict()

    @property
    def map(self):
        return self._map

    @property
    def inv_map(self):
        """
        return edge map as
        sparse dict of dict of list
        {color:{plc: [ X_i ] ... })
        """
        res = {}
        for plc_i, action_is in self._map.items():
            for action_i, color_ix in enumerate(action_is):
                if color_ix == 0:
                    continue
                if color_ix not in res:
                    res[color_ix] = {}
                if plc_i not in res[color_ix]:
                    res[color_ix][plc_i] = []
                res[color_ix][plc_i].append(action_i)
        return res

    def register_placement(self, p):
        """
        if the X_ti causes a color for this edge, record _map[plcmnt][X_i] = signed-color

        self._map : placement_i is True, what color will self become ?

        *tensor size = [ num_tile, num_action, num_color ]

        check
        """
        if p.index in self._map:
            print('placement {} already registered to HE {}'.format(p.index, self.index))
            return
        self._map[p.index] = [0] * len(p.maps)
        colors = p.template.half_edge_meta
        template_colors = []
        for i in range(len(p.template.boundary.ext_half_edges)):
            if i in colors:
                template_colors.append(colors[i].get('color', None))
            else:
                template_colors.append(None)

        own_edge_index = self.edges(index=True)
        for i, mapping in enumerate(p.maps):
            if own_edge_index not in mapping.edges:
                continue
            # boundary transformed by X_i
            bnd = [self._P.index_of_vertex(x) for x in mapping.transformed.boundary.vertices]
            bnd = r2.verts_to_edges(bnd)
            bnd = [self._P.index_of_half_edge(x) for x in bnd]
            if self.index not in bnd:
                continue
            ix_template_boundary = bnd.index(self.index)
            if template_colors[ix_template_boundary] is None:
                continue
            self._map[p.index][i] = template_colors[ix_template_boundary]

    @property
    def constraints(self):
        """ for actions

        Hard constraints - either X_0,i => c == X_1,j
        or the other half edge does not exist
        """
        return []


# ------------------------------------------------------------------
class Edge(Mesh2d.Edge, _VarGraphBase):
    def __init__(self, *args, n_colors=0, is_usable=1, **kwargs):
        _VarGraphBase.__init__(self)
        Mesh2d.Edge.__init__(self, *args, **kwargs)
        self._usable = max([0, min(1, is_usable)])
        self.X = Variable(shape=n_colors,
                          boolean=True,
                          name='edge.{}'.format(self.index))
        self._map = ddict(dict)
        self._placements = []
        self._check_acks = []

    def register_placement(self, p):
        """
        if the X_i implies a color for this edge's half edges, maximize
        the amount by which colors 'line up', aka:

            he1 @ X_rel[i] = he2 @ X_rel[j]  for all i, j in X_rel

            X_rel is subset of X_i s.t. self is in placement_i's boundary edges

        """
        self._placements.append(p)

    @property
    def J(self):
        return self.X

    @property
    def constraints(self):
        """
        required constraints:

        available options:

        if X_i, get he_color_j, he_color_k s.t. k = -j

        if color_j == 0 or color_k == 0 -> J_i => 0
        if color_j != color_k           -> J_i => 0
        if color_j == color_k           -> J_i => 1

        """
        # 0 or 1, goes to 0 if edge is restricted
        C = [] # cvx.sum(self.J) <= self._usable

        # if J_i = 1, this implies that one of the configurations
        # such that both half edges have matching colors
        #
        if len(self._half_edges) != 2 or self._usable == 0:
            C += [self.J == 0]
            return C

        he1, he2 = self._half_edges
        col_plc_x1, col_plc_x2 = he1.inv_map, he2.inv_map

        def tuplify_acks(he_acks, color_ix):
            """ get tuples of
                (placement_index, color_index, action_ix, sign)
            """
            res = []
            sign = 1 if color_ix > 0 else -1
            for plc_ix, act_ixs in he_acks.get(color_ix, {}).items():
                for act_ix in act_ixs:
                    res.append([plc_ix, np.abs(color_ix), act_ix, sign])
            return res

        for i in range(self.J.shape[0]):
            color_index = i + 1
            # gather actions which result in one of the half edges
            # having color at index i
            all_ac = []
            all_ac += tuplify_acks(col_plc_x1, +color_index)
            all_ac += tuplify_acks(col_plc_x1, -color_index)
            all_ac += tuplify_acks(col_plc_x2, +color_index)
            all_ac += tuplify_acks(col_plc_x2, -color_index)

            # sum 1p = 1 and sum(2n) = 1
            #  OR
            # sum 2p >= 1 and sum(1n) = 1
            and_indicators = []
            for ti, tj in itertools.combinations(all_ac, 2):
                if ti[0] != tj[0] and ti[2] != tj[2] and ti[3] != tj[3]:
                    self._check_acks.append([ti, tj])
                    p1 = self._placements[ti[0]].X[ti[2]]
                    p2 = self._placements[tj[0]].X[tj[2]]
                    # create indicator - todo is this necessary
                    ind_ij = Variable(boolean=True)
                    C += and_constraint([p1, p2], ind_ij)
                    and_indicators.append(ind_ij)
            if and_indicators:
                C += or_constraint(and_indicators, self.J[i])
        return C

    @property
    def objective_max(self):
        """
        maximize Joint
        """
        # f1, f2 = self.adj_vars(half_edge=True)
        # return cvx.max(self.X - cvx.abs(f1 - f2))
        return 0.5 * cvx.sum(self.J)


class Face(Mesh2d.Face, _VarGraphBase):
    def __init__(self, *args, n_colors=0, is_usable=1, is_sink=0, **kwargs):
        _VarGraphBase.__init__(self)
        Mesh2d.Face.__init__(self, *args, **kwargs)
        if n_colors == 0:
            self.X = Variable(boolean=True, name='face.{}'.format(self.index))
        else:
            self.X = Variable(n_colors, boolean=True, name='face.{}'.format(self.index))
        self._usable = max([0, min(1, is_usable)])
        self._is_sink = max([0, min(1, is_sink)])

    @property
    def constraints(self):
        # todo - constraint for all possible placements resulting in this config
        # this face being this color
        return [
            cvx.sum(self.X) <= self._usable,    # face can have at most one tile
            0 <= self.X,                        # all_pos
        ]

    @property
    def objective_max(self):
        return self.area * self.X   # maximize placed area



# ------------------------------------------------------------------
class Box(object):
    """ A box in a floor packing problem. """
    ASPECT_RATIO = 2.0

    def __init__(self, min_area,
                 max_area=None,
                 min_dim=None,
                 max_dim=None,
                 name=None,
                 aspect=2.0
                 ):
        # CONSTRAINTS
        self.name = name
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.max_area = max_area
        self.min_area = min_area
        self.aspect = aspect

        # VARS
        self.height = Variable(pos=True, name='{}.h'.format(name))
        self.width = Variable(pos=True, name='{}.w'.format(name))
        self.x = Variable(pos=True, name='{}.x'.format(name))
        self.y = Variable(pos=True, name='{}.y'.format(name))

    def __str__(self):
        return '{}: x: {}, y: {}, w: {} h: {}'.format(
           *[self.name] + [round(x, 2) for x in [self.x.value, self.y.value,
                                                 self.width.value, self.height.value]]
        )

    def detail(self):
        ds = str(self)
        ds += '\n\tarea: {}'.format(self.area.value)
        ds += '\n\tperimeter: {}'.format(self.perimeter.value)
        return ds

    def __iter__(self):
        yield self

    def __len__(self):
        return 1

    @property
    def position(self):
        return np.round(self.x.value, 2), np.round(self.y.value, 2)

    @property
    def size(self):
        return np.round(self.width.value, 2), np.round(self.height.value, 2)

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y

    @property
    def top(self):
        return self.y + self.height

    @property
    def perimeter(self):
        return 2 * self.width + self.height

    @property
    def area(self):
        return cvx.geo_mean(cvx.vstack([self.width, self.height]))

    # ------------------------------------------------------------------
    def own_constraints(self):
        constraints = []
        if self.aspect:
            constraints.append((1 / self.aspect) * self.height <= self.width)
            constraints.append(self.width <= self.aspect * self.height)

        if self.min_area:
            #
            constraints.append(self.area >= math.sqrt(self.min_area))

        if self.min_dim:
            # height and width must be atleast
            constraints.append(self.height >= self.min_dim)
            constraints.append(self.width >= self.min_dim)

        if self.max_dim:
            # height and width must less than
            constraints.append(self.height <= self.max_dim)
            constraints.append(self.width <= self.max_dim)

        return constraints


class Group:
    """ N boxes """
    def __init__(self, name=None,
                 min_area=100,
                 min_dim=None,
                 max_dim=None,
                 aspect=None):
        # Box.__init__(self, **kwargs)
        self.boxes = []
        self.name = name
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.min_area = min_area
        self.aspect = aspect
        #
        self.b1 = Box(None, name='{}-0'.format(name), min_dim=3, aspect=10)
        self.b2 = Box(None, name='{}-1'.format(name), min_dim=3, aspect=10)

    def __iter__(self):
        yield self.b1
        yield self.b2

    def __len__(self):
        return 2

    @property
    def x(self):
        return self.left

    @property
    def y(self):
        return self.bottom

    @property
    def bottom(self):
        return cvx.minimum(self.b1.y, self.b2.y)

    @property
    def top(self):
        return cvx.maximum(self.b1.y, self.b2.y)

    @property
    def left(self):
        return cvx.minimum(self.b1.x, self.b2.x)

    @property
    def right(self):
        return cvx.maximum(self.b1.x, self.b2.x)

    @property
    def height(self):
        return self.top - self.bottom

    @property
    def width(self):
        return self.right - self.left

    @property
    def area(self):
        return self.b1.area + self.b2.area

    def own_constraints(self):
        """ b1, b2 are adjacent """
        constraints = []
        constraints += cnstr.must_be_adjacent(self.b1, self.b2)
        if self.min_dim:
            # height and width must be atleast
            constraints.append(self.height >= self.min_dim)
            constraints.append(self.width >= self.min_dim)

        if self.min_area:
            constraints.append(self.area >= self.min_area)

        # todo need workaround for grouping aspect ratios - breaks dcp
        # if self.aspect:
        #    constraints.append((1 / self.aspect) * self.height <= self.width)
        #    constraints.append(self.width <= self.aspect * self.height)

        constraints += self.b1.own_constraints()
        constraints += self.b2.own_constraints()
        return constraints


class Polygon6(Box):
    """ N boxes """
    def __init__(self, **kwargs):
        Box.__init__(self, None, **kwargs)
        #
        self.b2 = Box(None, name='{}-1'.format(self.name))

    def __len__(self):
        return 2

    def within(self, other):
        return

    def expr_intersects(self, other, eps=1e3):
        or_vars = Variable(shape=4, boolean=True,
                           name='overlap_or({},{})'.format(self.name, other.name))
        or_vars2 = Variable(shape=4, boolean=True,
                            name='overlap_or({},{})'.format(self.name, other.name))
        j_vars = Variable(2, boolean=True)
        constraints = [
            # overlaps
            # 0 , 0 -> True
            # 0 , 1 -> False
            # 1 , 0 -> False
            # 1 , 1 -> False
            # Does not overlap main
            self.right <= other.x + eps * or_vars[0],
            other.right <= self.x + eps * or_vars[1],
            self.top <= other.y + eps * or_vars[2],
            other.top <= self.y + eps * or_vars[3],
            sum(or_vars) <= 3,

            # --- or ---
            # overlaps main and overlaps cutout
            self.b2.right <= other.x + eps * or_vars2 [0],
            other.right <= self.b2.x + eps * or_vars2 [1],
            self.b2.top <= other.y + eps * or_vars2[2],
            other.top <= self.b2.y + eps * or_vars2[3],
            sum(or_vars2) >= 2,

            # if above(1) is met sum largest will
            cvx.sum_largest(or_vars, 3) * j_vars

        ]
        return constraints

    @property
    def area(self):
        return self.area - self.b2.area

    def own_constraints(self):
        """ b1, b2 """
        rr = cvx.vstack([
            self.x == self.b2.x,
            self.x == self.b2.x,
            self.x == self.b2.x,
            self.x == self.b2.x,
        ])

        constraints = []
        return constraints


