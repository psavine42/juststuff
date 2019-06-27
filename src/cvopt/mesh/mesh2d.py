from collections import defaultdict as ddict
import numpy as np
from cvxpy.utilities.performance_utils import compute_once, lazyprop
import networkx as nx
from .boundary import Boundary
import src.geom.r2 as r2
from .views import FaceView, EdgeView, HalfEdgeView, VertexView
from copy import deepcopy


class __x_Mesh2d(object):
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
            # vector = r2.to_homo([self.u, self.v])
            return M

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
        elif w or h:
            fh, fw = None, None
            if w and h:
                fh, fw = h, w
            elif w is None and h is not None:
                fh, fw = h, h
            elif w is not None and h is None:
                fh, fw = w, w
            G = nx.grid_2d_graph(fw, fh)
        else:
            pass
        # if isinstance(G, nx.PlanarEmbedding):
        #    self.G = G
        # else:
        #     res, self.G = nx.check_planarity(G)
        #     if res is False:
        #        raise Exception('not planar')
        self._d_faces = ddict(tuple) # face_ix {he1, he2 ... he_n} {nd_1, nd_2 ... nd__n}
        self._d_edges = ddict(tuple) # edge_ix {n1_ix, n2_ix}
        self._d_hes = ddict(tuple)   # edge_ix {n1_ix, n2_ix}
        self._d_verts = dict()      # index to geom
        self._g_verts = dict()      # geom to index

        # from nx - fuck that
        if isinstance(G, (nx.DiGraph, nx.PlanarEmbedding)):
            for i, n in enumerate(sorted(G.nodes)):
                self._d_verts[i] = n
                self._g_verts[n] = i
            for i, (n1, n2) in enumerate(G.edges):
                pass

        # from boxes
        elif isinstance(g, list):
            coords = [] # lexo coords
            edges = [] # lexo coords
            for ent in g:
                edges += r2.verts_to_edges(ent.coords)
                coords += ent.coords
            for i, n in enumerate(sorted(coords)):
                self._d_verts[i] = n
                self._g_verts[n] = i

            for i, ent in enumerate(g):
                hes = r2.verts_to_edges(ent.coords)
                added = []
                for half_edge in hes:
                    he_idx = len(self._d_hes)
                    added.append(he_idx)
                    self._d_hes[he_idx] = tuple([self._g_verts[x] for x in half_edge])
                self._d_faces[i] = tuple(added)

            for i, (u,v) in enumerate(sorted(edges)):
                self._d_edges[i] = (self._g_verts[u], self._g_verts[v])




    def relabel(self, mapping):
        """ """
        from copy import copy
        new = copy(self)
        for k, v in mapping:
            ix = new._g_verts[k]
            new._d_verts[ix] = v
            new._g_verts.pop(k)
            new._g_verts[v] = ix
        return new

    @property
    def nodes(self):
        return self.G.nodes

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
    def boundary_vertices(self):
        """ CCW list of vertecies on boundary
            IN OTHER WORDS - REPRESENTING WHERE
        """
        return self.boundary.vertices

    @compute_once
    def _boundary(self):
        return Boundary(self)

    @property
    def boundary(self):
        return self._boundary()

    # faces ------------------------------------------------------
    @compute_once  # CW
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
            face_verts = r2.sort_cw(self.G.traverse_face(u, v))
            nvers = len(face_verts)
            if nvers > 4:
                continue
            for i in range(2, nvers):
                if (face_verts[i], face_verts[0]) in self.G.edges:
                    face_verts = face_verts[0:i + 1]
                    break
            if stop is True:
                continue
            face_verts = tuple(r2.sort_cw(face_verts))
            if face_verts not in faces:
                faces.add(face_verts)
        faces = sorted(list(faces))
        return {i: verts for i, verts in enumerate(faces)}

    def faces(self):
        return list(self.faces_to_vertices().values())

    @compute_once  # CW
    def faces_to_half_edges(self):
        """
        dict of {face_index: [ (vertex_start, vertex_end), ...] ...}
        """
        f2e = {}
        for k, v in self.faces_to_vertices().items():
            ixs = list(range(len(v))) + [0]
            f2e[k] = [(v[ixs[i]], v[ixs[i + 1]]) for i in range(len(v))]
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
        return {n: i for i, n in enumerate(list(self.G.nodes))}

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

    # half edges -----------------------------------------------------
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

    @compute_once  # Lexigraphic
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
    # keeping these strict
    def index_of_face(self, geom):
        return self.faces_to_vertices().index(geom)

    def index_of_half_edge(self, geom):
        return self.half_edges().index(geom)

    def index_of_edge(self, geom):
        return self._edges().index(geom)

    def index_of_vertex(self, geom):
        return self.vertices().index(geom)

    # OO helpers ---------------------------------------------
    def get_half_edge(self, ix, cls=None, **kwargs):
        u, v, data = self.half_edges_data()[ix]
        kwargs['face'] = self.half_edges_to_faces()[(u, v)]
        if cls is None:
            cls = Mesh2d.HalfEdge
        return cls(self, ix, u, v, **kwargs)

    def get_edge(self, ix, cls=None, **kwargs):
        uv = self.edges[ix]
        if cls is None:
            cls = Mesh2d.Edge
        return cls(self, ix, uv, **kwargs)

    def get_vertex(self, ix, cls=None, **kwargs):
        xy = self.vertices()[ix]
        if cls is None:
            cls = Mesh2d.Vertex
        return cls(self, ix, xy, **kwargs)

    def get_face(self, ix, cls=None, **kwargs):
        verts = self.faces_to_vertices()[ix]
        edges = self.faces_to_edges()[ix]
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
            # vector = r2.to_homo([self.u, self.v])
            return M

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
        elif w or h:
            fh, fw = None, None
            if w and h:
                fh, fw = h, w
            elif w is None and h is not None:
                fh, fw = h, h
            elif w is not None and h is None:
                fh, fw = w, w
            G = nx.grid_2d_graph(fw, fh)
        else:
            pass
        # if isinstance(G, nx.PlanarEmbedding):
        #    self.G = G
        # else:
        #     res, self.G = nx.check_planarity(G)
        #     if res is False:
        #        raise Exception('not planar')
        self._data = dict(he={}, edge={}, face={}, vert={})

        # Face <--> HE <--> Edge <--> point
        self._d_faces = ddict(tuple)    # face_ix : {he1, he2 ... he_n}
        self._d_hes = ddict(tuple)      # half_edge_ix : {n1_ix, n2_ix}
        self._e2he = ddict(tuple)
        self._d_edges = ddict(frozenset)      # edge_ix : {n1_ix, n2_ix}
        self._ix2g_verts = dict()       # index : geom
        self._g2ix_verts = dict()       # geom : index

        # from nx - fuck that
        if isinstance(g, (nx.DiGraph, nx.PlanarEmbedding)):
            for i, n in enumerate(sorted(G.nodes)):
                self._ix2g_verts[i] = n
                self._g2ix_verts[n] = i
            for i, (n1, n2) in enumerate(G.edges):
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
            for (u, v) in sorted(edges):
                iu, iv = self._g2ix_verts[u], self._g2ix_verts[v]
                if {u, v} not in seen:
                    i = len(seen)
                    seen.append({u, v})
                    self._d_edges[i] = frozenset([iu, iv])
                else:
                    i = seen.index({u, v})
                he_idx = heinv[(iu, iv)] if (iu, iv) in heinv else heinv[(iv, iu)]
                self._e2he[i] = tuple(list(self._e2he[i]) + [he_idx])

    def relabel(self, mapping):
        """ """
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

    @property
    def nodes(self):
        return self._g2ix_verts

    # misc ---------------------------------------------------------
    @lazyprop
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
    @lazyprop
    def boundary(self):
        bnds = []
        for k, he_ixs in self._e2he.items():
            if len(he_ixs) == 1:
                vert_index = self._d_hes[he_ixs[0]][0]
                bnds.append(self._ix2g_verts[vert_index])
        return Boundary.from_points(self, bnds)

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
        return self.half_edges.index(geom)

    def index_of_edge(self, geom):
        return self.edges.index(set(geom))

    def index_of_vertex(self, geom):
        return self.vertices.index(geom)

    # OO helpers ---------------------------------------------
    def get_half_edge(self, ix, cls=None, **kwargs):
        u, v, data = self.half_edges.data[ix]
        kwargs['face'] = self.half_edges.to_faces[(u, v)]
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

