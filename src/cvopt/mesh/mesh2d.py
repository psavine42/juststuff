from collections import defaultdict as ddict
import numpy as np
from cvxpy.utilities.performance_utils import compute_once, lazyprop
import networkx as nx
from .boundary import Boundary
import src.geom.r2 as r2
from .views import FaceView, EdgeView, HalfEdgeView, VertexView
from copy import deepcopy
from src.cvopt.shape import BTile
import itertools


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
        def __init__(self, parent, index, v, **kwargs):
            Mesh2d._MeshObj.__init__(self, parent, index)
            self.v = v

        def faces(self):
            return

        @property
        def half_edges_from(self):
            return

    class Edge(_MeshObj):
        def __init__(self, parent, index, uv, **kwargs):
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
            return self.np / (self.np ** 2).sum() ** 0.5

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

        @property
        def geom(self):
            return self.u, self.v

        def to_mat(self):
            base = self.u - self.v
            base = base / np.linalg.norm(base)
            theta = np.angle(base)
            M = r2.compose_mat(angle=theta, translate=self.u)
            return M

        def face(self, index=False, verts=False):
            """ """
            if index:
                return self._P.index_of_face(self._face)
            elif verts:
                return self._face
            return self._P.get_face(self._face)

        def edges(self, index=True):
            """ Edge indicies corresponding to this half edges """
            u, v = tuple(sorted([self.u, self.v]))
            if index is True:
                # (self.u, self.v)
                ui1 = self._P.index_of_vertex(self.u)
                ui2 = self._P.index_of_vertex(self.v)
                return self._P.half_edges.to_edges[(ui1, ui2)]
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

        @property
        def geom(self):
            return self.verts

        def adjacent_faces(self):
            return

        @property
        def bottom_left(self):
            return self.verts[0]

        def edges(self, index=True):
            edges = self._P.faces.to_edges[self.index]
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
        elif w or h:
            fh, fw = None, None
            if w and h:
                fh, fw = h, w
            elif w is None and h is not None:
                fh, fw = h, h
            elif w is not None and h is None:
                fh, fw = w, w
            G = nx.grid_2d_graph(fw, fh)
        self._data = dict(he={}, edge={}, face={}, vert={})

        # Face <--> HE <--> Edge <--> point
        self._d_faces = ddict(tuple)        # face_ix : {he1, he2 ... he_n}
        self._d_hes = ddict(tuple)          # half_edge_ix : {n1_ix, n2_ix}
        self._e2he = ddict(tuple)           #
        self._d_edges = ddict(frozenset)    # edge_ix : {n1_ix, n2_ix}
        self._ix2g_verts = dict()           # index : geom
        self._g2ix_verts = dict()           # geom : index

        # from nx - fuck that
        if isinstance(g, (nx.DiGraph, nx.PlanarEmbedding)):
            for i, n in enumerate(sorted(g.nodes)):
                self._ix2g_verts[i] = n
                self._g2ix_verts[n] = i
            for i, (n1, n2) in enumerate(g.edges):
                pass
        # from boxes
        elif isinstance(g, list):
            coords = []     # lexo coords
            edges = []      # lexo edges
            for ent in g:
                edges += r2.verts_to_edges(ent.coords)
                coords += ent.coords

            for n in sorted(coords):
                if n not in self._g2ix_verts:
                    i = len(self._g2ix_verts)
                    self._ix2g_verts[i] = n
                    self._g2ix_verts[n] = i

            for i, ent in enumerate(g):
                hes = r2.verts_to_edges(ent.coords)
                he_indexes = []
                for half_edge in hes:
                    he_idx = len(self._d_hes)
                    he_indexes.append(he_idx)
                    self._d_hes[he_idx] = tuple([self._g2ix_verts[x] for x in half_edge])
                self._d_faces[i] = tuple(he_indexes)

            heinv = {v_ixs: k for k, v_ixs in self._d_hes.items()}
            seen = []
            for (u, v) in sorted(edges):#todo check this
                # for (u, v) in edges:
                iu, iv = self._g2ix_verts[u], self._g2ix_verts[v]
                if {u, v} not in seen:
                    i = len(seen)
                    seen.append({u, v})
                    self._d_edges[i] = frozenset([iu, iv])
                else:
                    i = seen.index({u, v})
                he_idx = heinv[(iu, iv)] if (iu, iv) in heinv else heinv[(iv, iu)]
                self._e2he[i] = tuple(list(self._e2he[i]) + [he_idx])

    # -----------------------------------------------------
    def relabel(self, mapping):
        """
        mapping is a dict of {old_node_coords : new_node_coord ...}
        """
        new = self.copy()
        for i, (cur_geom, new_geom) in enumerate(mapping):
            new._g2ix_verts.pop(cur_geom)
            new._ix2g_verts[i] = new_geom
            new._g2ix_verts[new_geom] = i
        return new

    def copy(self):
        new = Mesh2d()
        new._d_hes = deepcopy(self._d_hes)
        new._d_edges = deepcopy(self._d_edges)
        new._d_faces = deepcopy(self._d_faces)
        new._e2he = deepcopy(self._e2he)
        new._ix2g_verts = deepcopy(self._ix2g_verts)
        new._g2ix_verts = deepcopy(self._g2ix_verts)
        return new

    @classmethod
    def from_grid(cls, w, h):
        return cls(g=[BTile(x) for x in itertools.product(range(w), range(h))])

    # -----------------------------------------------------
    @property
    def nodes(self):
        return self._g2ix_verts

    @lazyprop
    def interior_half_edge_index(self):
        """
        immutable catalogue of INTERIOR half edges of size interior edges

        dict of {0: { hald_edge_index1 : hald_edge_index2 ... , }
                1:  { hald_edge_index2 : hald_edge_index1 ...   }}

        """
        adj = {0: {}, 1: {}}
        for e_ix, half_edges in self.edges.to_half_edges.items():
            if len(half_edges) == 1:
                continue
            ix_he1, ix_he2 = [self.index_of_half_edge(x) for x in half_edges]
            adj[0][ix_he1] = ix_he2
            adj[1][ix_he2] = ix_he1
        return adj

    # Views --------------------------------------------------
    @lazyprop
    def boundary(self):
        ks, bnds = [], []
        for k, he_ixs in self._e2he.items():
            if len(he_ixs) == 1:
                vert_index = self._d_hes[he_ixs[0]][0]
                bnds.append(self._ix2g_verts[vert_index])
                ks.append(k)
        return Boundary.from_kvs(self, ks, bnds)

    @lazyprop
    def faces(self):
        return FaceView(self)

    @lazyprop
    def edges(self):
        return EdgeView(self)

    @lazyprop
    def vertices(self):
        return VertexView(self)

    @lazyprop
    def half_edges(self):
        return HalfEdgeView(self)

    # indexing -----------------------------------------------
    def index_of_face(self, geom):
        return self.faces.inv[geom]

    def index_of_half_edge(self, geom):
        if isinstance(geom[0], int):
            return self.half_edges.index(geom)
        elif isinstance(geom[0], tuple):
            return self.half_edges.index_geom(geom)

    def index_of_edge(self, geom):
        if isinstance(geom, (set, frozenset)):
            geom = list(geom)
        if isinstance(geom[0], int): # index of
            return self.edges.index(frozenset(geom))
        return self.edges.index_geom(frozenset(geom))

    def index_of_vertex(self, geom):
        return self._g2ix_verts[geom]

    # OO helpers ---------------------------------------------
    def get_half_edge(self, ix, cls=None, **kwargs):
        geom, data = self.half_edges.data[ix]
        u, v = geom
        kwargs['face'] = self.half_edges.to_faces[ix]
        if cls is None:
            cls = Mesh2d.HalfEdge
        return cls(self, ix, u, v, **kwargs)

    def get_edge(self, ix, cls=None, **kwargs):
        uv = self.edges[ix]
        if cls is None:
            cls = Mesh2d.Edge
        return cls(self, ix, uv, **kwargs)

    def get_vertex(self, ix, cls=None, **kwargs):
        xy = self.vertices[ix]
        if cls is None:
            cls = Mesh2d.Vertex
        return cls(self, ix, xy, **kwargs)

    def get_face(self, ix, cls=None, **kwargs):
        verts = self.faces.to_vertices[ix]
        edges = self.faces.to_edges[ix]
        if cls is None:
            cls = Mesh2d.Face
        return cls(self, ix, verts, edges=edges, **kwargs)

    # embedding utils -------------------------------------
    def __contains__(self, item):
        if isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[0], (float, int)):
                # node
                return item in self.G
            elif isinstance(item[0], (list, tuple)):
                # half-edge
                return item in self.half_edges
        elif isinstance(item, Mesh2d.Face):
            return
        elif isinstance(item, Mesh2d.Vertex):
            return item.v in self.vertices
        elif isinstance(item, Mesh2d.Edge):
            return (item.u, item.v) in self.edges
        elif isinstance(item, Mesh2d.HalfEdge):
            return (item.u, item.v) in self.half_edges
        return False

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if other._d_hes == self._d_hes and \
                    other._d_edges == self._d_edges and \
                    other._d_faces == self._d_faces and \
                    other._e2he == self._e2he and \
                    other._ix2g_verts == self._ix2g_verts and \
                    other._g2ix_verts == self._g2ix_verts:
                return True
        return False

    def __str__(self):
        return '{}'.format(self.__class__.__name__)

    def info(self):
        st = ''
        st += str(self.vertices)
        st += str(self.half_edges)
        st += str(self.edges)
        st += str(self.faces)
        return st

