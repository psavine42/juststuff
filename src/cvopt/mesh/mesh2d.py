from collections import defaultdict as ddict
import numpy as np
from cvxpy.utilities.performance_utils import compute_once


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
        od = {i: [bnds[i]] for i in range(-1, len(bnds) - 1)}
        for x3, y3 in pnts:
            for i in range(-1, len(bnds) - 1):
                (x1, y1), (x2, y2) = bnds[i], bnds[i + 1]
                if (x3, y3) in [bnds[i], bnds[i + 1]]:
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
            face_verts = sort_cw(self.G.traverse_face(u, v))
            nvers = len(face_verts)
            if nvers > 4:
                continue
            for i in range(2, nvers):
                if (face_verts[i], face_verts[0]) in self.G.edges:
                    face_verts = face_verts[0:i + 1]
                    break
            if stop is True:
                continue
            face_verts = tuple(sort_cw(face_verts))
            if face_verts not in faces:
                faces.add(face_verts)
        faces = sorted(list(faces))
        return {i: verts for i, verts in enumerate(faces)}

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
        return Edge(self, ix, uv, **kwargs)

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
        od = {i: [bnds[i]] for i in range(-1, len(bnds) - 1)}
        for x3, y3 in pnts:
            for i in range(-1, len(bnds) - 1):
                (x1, y1), (x2, y2) = bnds[i], bnds[i + 1]
                if (x3, y3) in [bnds[i], bnds[i + 1]]:
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
        return [tuple(sorted([bnd[i], bnd[i + 1]])) for i in range(-1, len(bnd) - 1)]

    @property  # CCW
    def ext_half_edges(self):
        """
        not-valid half edges that are on the boundary
        list of [ half_edge_coord, ... ]
        """
        bnd_ccw = self._vertices()  # ccw
        return [(bnd_ccw[i], bnd_ccw[i + 1]) for i in range(len(bnd_ccw) - 1)] + [(bnd_ccw[-1], bnd_ccw[0])]

    @property  # CW
    def int_half_edges(self):
        """
        VALID half edges that are on the boundary
        list of [ half_edge_coord, ... ]
        """
        bnd_cw = list(reversed(self._vertices()))  # cw
        return [(bnd_cw[-1], bnd_cw[0])] + [(bnd_cw[i], bnd_cw[i + 1]) for i in range(len(bnd_cw) - 1)]

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
