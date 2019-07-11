from cvxpy.utilities.performance_utils import compute_once, lazyprop
from collections import defaultdict as ddict
import src.geom.r2 as r2


class _View(object):
    DAT_KEY = ''
    # vtype = None

    def __init__(self, parent):
        self._p = parent
        self.vtype = None

    # --------------------------------------
    @property
    def base(self):
        raise NotImplemented()

    @property
    def geom(self):
        raise NotImplemented()

    @property
    def to_half_edges(self):
        raise NotImplemented()

    @property
    def to_faces(self):
        raise NotImplemented()

    @property
    def to_vertices(self):
        raise NotImplemented()

    @property
    def to_edges(self):
        raise NotImplemented()

    # COMMON --------------------------------------
    def __getitem__(self, item):
        return self.geom[item]

    def __setitem__(self, key, value):
        """
        sets metadata of a key in respective dictionary

        key must be a tuple of (index of entity, property)
        """
        index, prop = key
        if index not in self._p._data[self.DAT_KEY]:
            self._p._data[self.DAT_KEY][index] = {}
        self._p._data[self.DAT_KEY][index][prop] = value

    def __iter__(self):
        for k, v in self.base.items():
            yield k, v

    def __contains__(self, item):
        return item in self.base

    def __str__(self):
        return str(self.base)

    def __repr__(self):
        return str(self.base)

    def __len__(self):
        return len(self.base)

    # def vtype(self):
    #    return self.base.

    # -----------------------------------------------
    def values(self):
        return list(self.base.values())

    def index(self, ent):
        """ """
        return self.inv[self.vtype(ent)]

    def index_geom(self, ent):
        return self.geom.index(ent)

    @lazyprop
    def inv(self):
        """ """
        return {v: k for k, v in self.base.items()}

    @property
    def data(self):  # todo
        """
        list of admissible half edges from networkx with data
        """
        return [(g, self._p._data[self.DAT_KEY].get(i, {})) for i, g in enumerate(self.geom)]


class HalfEdgeView(_View):
    """ half_edge_ix : {n1_ix, n2_ix} """
    DAT_KEY = 'he'
    vtype = tuple

    def __init__(self, parent):
        _View.__init__(self, parent)
        self.vtype = tuple

    @property
    def base(self):
        return self._p._d_hes

    @property
    def geom(self):
        """
        list of admissible half edges from networkx
        """
        return [(self._p._ix2g_verts[v[0]], self._p._ix2g_verts[v[1]])
                for i, v in self.base.items()]

    @lazyprop
    def to_vertices(self):
        return

    @lazyprop
    def to_vertex_geom(self):
        return {i:(self._p._ix2g_verts[v[0]], self._p._ix2g_verts[v[1]])
                for i, v in self.base.items()}

    @lazyprop   # Lexigraphic
    def to_edges(self):
        """ dict: { half_edge_coord : edge_index } """
        _edges = self._p
        he2e = dict()
        for k, (u, v) in self.base.items():
            he2e[(u, v)] = self._p.index_of_edge({u, v})
        return he2e

    @lazyprop  # Lexigraphic
    def to_edges_index(self):
        """ dict: { half_edge_coord : edge_index } """
        _edges = self._p
        he2e = dict()
        for k, (u, v) in self.base.items():
            he2e[k] = self._p.index_of_edge({u, v})
        return he2e

    @lazyprop
    def to_faces(self):
        """ dict of { (vertex_start, vertex_end) : face_index ...} """
        he2f = {}
        for face_ix, hes in self._p._d_faces.items():
            for he in hes:
                he2f[he] = face_ix
        return he2f


class VertexView(_View):
    DAT_KEY = 'vert'

    def __init__(self, parent):
        _View.__init__(self, parent)
        self.vtype = tuple

    @property
    def base(self):
        return self._p._ix2g_verts

    @lazyprop
    def geom(self):
        return [n for i, n in self.base.items()]

    @property
    def to_index(self):
        return self._p._g2ix_verts

    @lazyprop
    def to_faces(self):
        """
        dict of { node : {face_ix ...} ...}
        """
        v2f = ddict(set)
        for f, he_idxs in self._p._d_face.items():
            for he_idx in he_idxs:
                for v in self._p._d_hes[he_idx]:
                    v2f[v].add(f)
        return v2f


class EdgeView(_View):
    DAT_KEY = 'edge'

    def __init__(self, parent):
        _View.__init__(self, parent)
        self.vtype = frozenset

    @property
    def base(self):
        return self._p._d_edges

    @property
    def geom(self):
        """ returns list of sets containing vertex coords"""
        res = [None] * len(self.base)
        for k, (n1_ix, n2_ix) in self.base.items():
            res[k] = {self._p._ix2g_verts[n1_ix], self._p._ix2g_verts[n2_ix]}
        return res

    @lazyprop
    def to_vertices(self):
        """ dict of { edge_index : { vertex_coord ... } ... } """
        return

    @lazyprop
    def to_edges(self):
        return self.base

    @lazyprop
    def to_half_edges(self):
        """
        dict of { edge_index : { half_edge_coord ... } ... }
        """
        e2he = ddict(set)
        for he, e_ix in self._p.half_edges.to_edges.items():
            e2he[e_ix].add(he)
        return e2he

    @lazyprop
    def to_faces(self):
        """
         dict of { edge_index : {face_index_1, face_index_2 } ... }
        """
        e2f = ddict(set)
        for face_ix, edges in self._p.faces.to_edges.items():
            for e in edges:
                e2f[self._p.index_of_edge(e)].add(face_ix)
        return e2f


class FaceView(_View):
    DAT_KEY = 'face'

    def __init__(self, parent):
        _View.__init__(self, parent)
        self.vtype = tuple

    @property
    def base(self):
        """returns list of {face_index: tuple(vertex indices) ...} CW """
        return self._p._d_faces

    @property
    def geom(self):
        """returns list of vertex tuples """
        return [n for i, n in self.to_vertices.items()]

    @lazyprop  # CW - OK
    def to_vertices(self):
        """
        OK
        dict of {face_index: [vertex_coord ...] ...}

        Note - PlanarEmbedding considers the exterior plane as a face - needs
        to be filtered out - if the half edge belongs to t
        """
        f2v = {}
        for face_ix, hes in self.base.items():
            verts = []
            for he_idx in hes:
                n1, n2 = self._p._d_hes[he_idx]
                verts.append(self._p._ix2g_verts[n1])
                verts.append(self._p._ix2g_verts[n2])
            f2v[face_ix] = r2.sort_cw(list(set(verts)))
        return f2v

    @lazyprop  # CW
    def to_half_edges(self):
        """ dict of {face_index: [ (vertex_coord_start, vertex_coord_end), ...] ...} """
        f2he = {}
        for k, hes in self.base.items():
            verts = []
            for he_idx in hes:
                n1, n2 = self._p._d_hes[he_idx]
                verts.append((self._p._ix2g_verts[n1], self._p._ix2g_verts[n2]))
            f2he[k] = verts
        return f2he

    @lazyprop  # CW
    def to_edges(self):
        """ dict of {face_index: [ (vertex_coord_start, vertex_coord_end), ...] ...} """
        f2e = {}
        for k, hes in self.base.items():
            verts = []
            for half_edge_idx in hes:
                n1, n2 = self._p.half_edges.base[half_edge_idx]
                geom = frozenset([self._p._ix2g_verts[n1], self._p._ix2g_verts[n2]])
                verts.append(geom)
            f2e[k] = verts
        return f2e
