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
from cvxpy.utilities.performance_utils import compute_once
import networkx as nx
from collections import defaultdict as ddict
from functools import reduce
import operator
from src.cvopt.utils import translate


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


def sort_cw(coords):
    center = tuple(map(operator.truediv,
                       reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    return sorted(coords, key=lambda coord: (-135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)


def verts_to_edges(coords):
    base = [(coords[i], coords[i+1]) for i in range(len(coords)-1)]
    return base + [(coords[-1], coords[0])]


# ----------------------------------------------------------
class Mesh2d(object):
    """
    *** IMMUTABLE ***

    assumes no dangling edgese
    """
    class _MeshObj(object):
        def __init__(self, parent, index):
            self._P = parent
            self._index = index

        @property
        def G(self):
            return self._P

        @property
        def index(self):
            return self._index

        def _info(self):
            return ''

        def adjacent_faces(self):
            return

        def adjacent_edges(self):
            return

        def adjacent_half_edges(self):
            return

        def adjacent_vertices(self):
            return

        @property
        def geom(self):
            return []

        @property
        def np(self):
            return np.asarray(self.geom)

        def __str__(self):
            return '{}:{} - {}'.format(self.__class__.__name__, self._index, self._info())

        def __repr__(self):
            return self.__str__()

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return self.index == other.index
            return False

    # -----------------------------------------------------------
    class Vertex(_MeshObj):
        def __init__(self, parent, index, v,  **kwargs):
            Mesh2d._MeshObj.__init__(self, parent, index)
            self.v = v

        def faces(self):
            return

        @property
        def half_edges_from(self):
            return

    class Edge(_MeshObj):
        def __init__(self, parent, index,  uv, **kwargs):
            Mesh2d._MeshObj.__init__(self, parent, index)
            self.u = uv

        @property
        def geom(self):
            return self.u

        def faces(self):
            return

        @property
        def unit_vec(self):
            uv = self.np[1] - self.np[0]
            return self.np / (self.np **2).sum()**0.5

    class HalfEdge(_MeshObj):
        def __init__(self, parent, index, i, j,
                     cw=None, ccw=None, face=None,
                     **kwargs):
            """

            face: index
            """
            Mesh2d._MeshObj.__init__(self, parent, index)
            self.u = i
            self.v = j
            self._face = face

            self._cw, self._ccw = cw, ccw

        def _info(self):
            return '{} {}'.format(self.u, self.v)

        @property
        def length(self):
            return

        def face(self, index=False, verts=False):
            """ """
            if index:
                return self._P.index_of_face(self._face)
            elif verts:
                return self._face
            return self._P.get_face(self._face)

        def edges(self, index=True):
            u, v = tuple(sorted([self.u, self.v]))
            if index is True:
                return self._P.half_edges_to_edges()[(self.u, self.v)]
            return u, v

        def vertices(self, index=False):
            if index is True:
                ui1 = self._P.index_of_vertex(self.u)
                ui2 = self._P.index_of_vertex(self.v)
                return ui1, ui2

    class Face(_MeshObj):
        def __init__(self, parent, index, verts, **kwargs):
            Mesh2d._MeshObj.__init__(self, parent, index)
            self.verts = verts

        @property
        def area(self):
            pts = np.asarray(self.verts)
            lines = np.hstack([pts, np.roll(pts, -1, axis=0)])
            area = np.abs(np.sum([x1 * y2 - x2 * y1 for x1, y1, x2, y2 in lines]))
            return 0.5 * area

        def adjacent_faces(self):
            return

        @property
        def bottom_left(self):
            return self.verts[0]

        def edges(self, index=True):
            edges = self._P.faces_to_edges()[self.index]
            if index is True:
                return [self._P.index_of_edge(e) for e in edges]

    """
    --------------------------------------------------------------
    Immutable 
    
    Storage:
        vertices   - 
        half-edges - cw edges that touch a valid face
        edges      - lexigraphicaly ordered list of edges
        faces      - 
        
    12 functions to switch between representations (4 x 4 - trace())
        
    """
    def __init__(self, g=None, w=None, h=None):
        self.G = None
        if g is not None and isinstance(g, nx.Graph):
            G = g
        else:
            fh, fw = None, None
            if w and h:
                fh, fw = h, w
            elif w is None and h is not None:
                fh, fw = h, h
            elif w is not None and h is None:
                fh, fw = w, w
            G = nx.grid_2d_graph(fw, fh)

        res, self.G = nx.check_planarity(G)
        if res is False:
            raise Exception('not planar')

    # misc ---------------------------------------------------------
    @compute_once
    def interior_half_edge_index(self):
        """
        immutable catalogue of INTERIOR half edges of size interior edges

        dict of {0: { hald_edge_index1 : hald_edge_index2 ... , }
                1:  { hald_edge_index2 : hald_edge_index1 ...   }}

        """
        adj = {0: {}, 1: {}}
        for e_ix, half_edges in self.edges_to_half_edges().items():
            if len(half_edges) == 1:
                continue
            ix_he1, ix_he2 = [self.index_of_half_edge(x) for x in half_edges]
            adj[0][ix_he1] = ix_he2
            adj[1][ix_he2] = ix_he1
        return adj

    # boundary --------------------------------------------------
    @compute_once
    def boundary_vertices(self):
        """ CCW list of vertecies on boundary
            IN OTHER WORDS - REPRESENTING WHERE
        """
        pnts = np.asarray(self.vertices(), dtype=int)
        hull = scipy.spatial.ConvexHull(pnts)
        bnds = [(x, y) for x, y in zip(pnts[hull.vertices, 0], pnts[hull.vertices, 1])]
        od = {i: [bnds[i]] for i in range(-1, len(bnds)-1)}
        for x3, y3 in pnts:
            for i in range(-1, len(bnds)-1):
                (x1, y1), (x2, y2) = bnds[i], bnds[i+1]
                if (x3, y3) in [bnds[i], bnds[i+1]]:
                    break
                a = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                if a == 0:
                    od[i].append((x3, y3))
        extras = []
        for k, v in od.items():
            if len(v) == 1:
                pass
            elif v[0] < v[1]:
                v.sort()
            else:
                v.sort(reverse=True)
            extras += v
        min_ix = extras.index(min(extras))
        fl = extras[min_ix:] + extras[0:min_ix]
        return fl

    def _boundary(self):
        return Boundary(self)

    @property
    def boundary(self):
        return self._boundary()

    # faces ------------------------------------------------------
    @compute_once   # CW
    def faces_to_vertices(self):
        """
        OK
        dict of {face_index: [node_location ...] ...}

        Note - PlanarEmbedding considers the exterior plane as a face - needs
        to be filtered out - if the half edge belongs to t
        """
        # bnd = set(self.boundary_ext_half_edges())
        faces = set()
        for u, v in self.half_edges():
            stop = False
            face_verts = sort_cw(self.G.traverse_face(u, v))
            nvers = len(face_verts)
            if nvers > 4:
                continue
            for i in range(2, nvers):
                if (face_verts[i], face_verts[0]) in self.G.edges:
                    face_verts = face_verts[0:i+1]
                    break
            if stop is True:
                continue
            face_verts = tuple(sort_cw(face_verts))
            if face_verts not in faces:
                faces.add(face_verts)
        faces = sorted(list(faces))
        return {i: verts for i, verts in enumerate(faces)}

    @compute_once   # CW
    def faces_to_half_edges(self):
        """
        dict of {face_index: [ (vertex_start, vertex_end), ...] ...}
        """
        f2e = {}
        for k, v in self.faces_to_vertices().items():
            ixs = list(range(len(v))) + [0]
            f2e[k] = [(v[ixs[i]], v[ixs[i+1]]) for i in range(len(v))]
        return f2e

    @compute_once  # CW
    def faces_to_edges(self):
        """
        dict of {face_index: [ (vertex_start, vertex_end), ...] ...}
        """
        f2e = {}
        for k, v in self.faces_to_half_edges().items():
            f2e[k] = [tuple(sorted([p1, p2])) for p1, p2 in v]
        return f2e

    # edges ------------------------------------------------------
    @compute_once
    def _edges(self):
        """ half edges are stored so eliminate duplicates

        list: [ (vertex_start, vertex_end), ...]

        sorted lexigraphically
        """
        es = set([tuple(sorted([u, v])) for u, v in self.G.edges])
        return sorted(list(es))

    @compute_once
    def edges_to_half_edges(self):
        """
        dict of { edge_index : { half_edge_coord ... } ... }
        """
        e2he = ddict(set)
        for he, e_ix in self.half_edges_to_edges().items():
            e2he[e_ix].add(he)
        return e2he

    @compute_once
    def edges_to_faces(self):
        """
         dict of { edge_index : {face_index_1, face_index_2 } ... }
        """
        e2f = ddict(set)
        for face_ix, edges in self.faces_to_edges().items():
            for e in edges:
                e2f[self.index_of_edge(e)].add(face_ix)
        return e2f

    @property
    def edges(self):
        return self._edges()

    # vertices ------------------------------------------------------
    @compute_once
    def vertices(self):
        return list(self.G.nodes)

    @compute_once
    def vertices_to_index(self):
        return {n:i for i, n in enumerate(list(self.G.nodes))}

    @compute_once
    def vertices_to_half_edges(self):
        return

    @compute_once
    def vertices_to_faces(self):
        """
        dict of { node : {face_ix ...} ...}
        """
        v2f = ddict(set)
        for f, verts in self.faces_to_vertices().items():
            for v in verts:
                v2f[v].add(f)
        return v2f

    # half edges ------------------------------------------------------
    # @compute_once
    def half_edges_data(self):
        """
        list of admissible half edges from networkx with data
        """
        return [(u, v, self.G[u][v]) for u, v in self.half_edges()]

    @compute_once
    def half_edges(self):
        """
        list of admissible half edges from networkx
        """
        bnd = set(self.boundary.ext_half_edges)
        return [(u, v) for u, v in self.G.edges if (u, v) not in bnd]

    @compute_once
    def half_edges_to_faces(self):
        """
        dict of { (vertex_start, vertex_end) : face_index ...}
        """
        he2f = {}
        for f, hes in self.faces_to_half_edges().items():
            for he in hes:
                he2f[he] = f
        return he2f

    @compute_once   # Lexographic
    def half_edges_to_edges(self):
        """
        dict: { half_edge_coord : edge_index }

        """
        _edges = self._edges()
        he2e = dict()
        for u, v in self.half_edges():
            he2e[(u, v)] = _edges.index(tuple(sorted([u, v])))
        return he2e

    # indexing -----------------------------------------------
    def index_of_face(self, geom):
        return self.faces_to_vertices().index(geom)

    def index_of_half_edge(self, geom):
        return self.half_edges().index(geom)

    def index_of_edge(self, geom):
        return self._edges().index(geom)

    def index_of_vertex(self, geom):
        return self.vertices().index(geom)

    # OO helpers ---------------------------------------------
    def get_half_edge(self, ix, **kwargs):
        u, v, data = self.half_edges_data()[ix]
        kwargs['face'] = self.half_edges_to_faces()[(u, v)]
        return HalfEdge(self, ix, u, v, **kwargs)

    def get_edge(self, ix, **kwargs):
        uv = self.edges[ix]
        return Edge(self, ix, uv, **kwargs )

    def get_vertex(self, ix, **kwargs):
        xy = self.vertices()[ix]

        return Vertex(self, ix, xy, **kwargs)

    def get_face(self, ix, **kwargs):
        verts = self.faces_to_vertices()[ix]
        edges = self.faces_to_edges()[ix]
        return Face(self, ix, verts, edges=edges, **kwargs)

    # embedding utils -------------------------------------
    def __contains__(self, item):
        if isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[0], (float, int)):
                # node
                return item in self.G
            elif isinstance(item[0], (list, tuple)):
                # half-edge
                return item in self.half_edges()
        elif isinstance(item, Mesh2d.Face):
            return
        elif isinstance(item, Mesh2d.Vertex):
            return item.v in self.vertices()
        elif isinstance(item, Mesh2d.Edge):
            return (item.u, item.v) in self._edges()
        elif isinstance(item, Mesh2d.HalfEdge):
            return (item.u, item.v) in self.half_edges()
        return False

    @compute_once
    def area(self):
        return len(self.faces_to_vertices())


class Boundary(object):
    def __init__(self, parent):
        self._parent = parent

    @compute_once
    def _vertices(self):
        """ CCW list of vertecies on boundary
            IN OTHER WORDS - REPRESENTING WHERE
        """
        pnts = np.asarray(self._parent.vertices(), dtype=int)
        hull = scipy.spatial.ConvexHull(pnts)
        bnds = [(x, y) for x, y in zip(pnts[hull.vertices, 0], pnts[hull.vertices, 1])]
        od = {i: [bnds[i]] for i in range(-1, len(bnds)-1)}
        for x3, y3 in pnts:
            for i in range(-1, len(bnds)-1):
                (x1, y1), (x2, y2) = bnds[i], bnds[i+1]
                if (x3, y3) in [bnds[i], bnds[i+1]]:
                    break
                a = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                if a == 0:
                    od[i].append((x3, y3))
        extras = []
        for k, v in od.items():
            if len(v) == 1:
                pass
            elif v[0] < v[1]:
                v.sort()
            else:
                v.sort(reverse=True)
            extras += v
        minv = min(extras)
        min_ix = extras.index(minv)
        fl = extras[min_ix:] + extras[0:min_ix]
        return fl

    @property
    def edges(self):
        bnd = self._vertices()  # ccw
        return [tuple(sorted([bnd[i], bnd[i+1]])) for i in range(-1, len(bnd)-1)]

    @property   # CCW
    def ext_half_edges(self):
        """
        not-valid half edges that are on the boundary
        list of [ half_edge_coord, ... ]
        """
        bnd_ccw = self._vertices()  # ccw
        return [(bnd_ccw[i], bnd_ccw[i + 1]) for i in range(len(bnd_ccw) - 1)] + [(bnd_ccw[-1], bnd_ccw[0])]

    @property   # CW
    def int_half_edges(self):
        """
        VALID half edges that are on the boundary
        list of [ half_edge_coord, ... ]
        """
        bnd_cw = list(reversed(self._vertices()))  # cw
        return [(bnd_cw[-1], bnd_cw[0])] + [(bnd_cw[i], bnd_cw[i + 1]) for i in range(len(bnd_cw)-1)]

    @property
    def vertices(self):
        return self._vertices()

    def __getitem__(self, item):
        return self._vertices()[item]


class _View(object):
    def __init__(self, parent):
        self._p = parent
        self._base_set = {}

    def to_half_edges(self):
        pass

    def to_faces(self, k=None, v=None):
        pass

    def to_vertices(self, k=None, v=None):
        pass

    def to_edges(self, k=None, v=None):
        pass

    def items(self):
        return

    def __getitem__(self, item):
        return


class EdgeView(_View):
    def __init__(self, parent):
        _View.__init__(self, parent)



# ------------------------------------------------------
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
        self._map[p.index] = [0] * len(p.placement_edges)
        # indexed to template.inner_boundary_half_edge (CW)
        template_colors = p.template.colors(half_edge=True)
        own_edge_index = self.edges(index=True)

        for i, edge_index_list in enumerate(p.placement_edges):
            if own_edge_index not in edge_index_list:
                continue
            # boundary transformed by X_i
            bnd = p.boundary_xformed(i, half_edges=True)

            if self.index not in bnd:
                continue
            ix_template_boundary = bnd.index(self.index)
            self._map[p.index][i] = template_colors[ix_template_boundary]

    @property
    def constraints(self):
        """ for actions

        Hard constraints - either X_0,i => c == X_1,j
        or the other half edge does not exist
        """
        return []


class Edge(Mesh2d.Edge, _VarGraphBase):
    def __init__(self, *args, n_colors=0, is_usable=1, **kwargs):
        _VarGraphBase.__init__(self)
        Mesh2d.Edge.__init__(self, *args, **kwargs)
        self._usable = max([0, min(1, is_usable)])
        self.X = Variable(shape=n_colors, boolean=True, name='edge.{}'.format(self.index))
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

        if X_i, get he_color_j, he_color_k
        if color_j == 0 or color_k == 0 -> J_i => 0
        if color_j != color_k           -> J_i => 0
        if color_j == color_k           -> J_i => 1

        """
        # 0 or 1, goes to 0 if edge is restricted
        C = [] # cvx.sum(self.J) <= self._usable

        # if J_i = 1, this implies that one of the configurations
        # such that both half edges have matching colors
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

    def sink_constraints(self, sinks):
        return

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


