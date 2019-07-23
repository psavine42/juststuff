from .spatial import Mesh2d, rectangle
from cvxpy.utilities.performance_utils import compute_once
from collections import defaultdict as ddict
import networkx as nx
import numpy as np
import src.geom.r2 as r2
import cvxpy as cvx
import shapely
import shapely.affinity
from shapely.geometry import MultiPolygon, Polygon, LineString, LinearRing
import itertools
from src.cvopt.shape import BTile


class _TemplateBase():
    def __init__(self,
                 allow_rot=True,
                 weight=1,
                 name=None,
                 color=None,
                 access=None):
        # ABCMeta.__init__(self)
        self.name = name
        self.color = color
        self.weight = weight
        self._vertex_meta = ddict(dict)
        self._face_meta = ddict(dict)
        self._edge_meta = ddict(dict)
        self._half_edge_meta = ddict(dict)

    @property
    def meta(self):
        return {'weight': self.weight,
                'name': self.name,
                'color': self.color}

    def colors(self, edge=None, face=None, half_edge=None):
        raise NotImplemented()

    def add_color(self, value, **kwargs):
        raise NotImplemented()


class OnEdge(_TemplateBase):
    def __init__(self, **kwargs):
        _TemplateBase.__init__(self, **kwargs)

    def add_color(self, value, **kwargs):
        return


class TemplateTile(Mesh2d, _TemplateBase):
    """
    User-Facing object
    """
    def __init__(self, w, h,
                 allow_rot=True,
                 max_uses=None,
                 access=None, **kwargs):
        tiles = [BTile(x) for x in itertools.product(range(w), range(h))]
        _TemplateBase.__init__(self, **kwargs)
        Mesh2d.__init__(self, g=tiles)
        self.w = w
        self.h = h
        self.max_uses = max_uses
        self.allow_rot = allow_rot
        self.attach_at = []                       # todo remove?
        self.access_at = access if access else [] # todo remove?

    def __hash__(self):
        return

    def __str__(self):
        return self.describe(True, True, True, True)

    def describe(self, v=None, he=None, e=None, f=None):
        """ printing utility """
        def _desc(l, geom_index, meta_map):
            s = ''
            for k, v in enumerate(geom_index):
                s += '\n{}.{} -> {} '.format(l, k, v)
                if l in ['half_edge']:
                    if v in self.boundary.ext_half_edges:
                        s += 'on-bnd index {} '.format(self.boundary.ext_half_edges.index(v))
                if k in meta_map:
                    s += '::meta: {}'.format(meta_map[k])
            return s

        st = 'Temaplate {}, anchored at {}'.format(
            self.name, self.anchor(half_edge=True)
        )

        if v:
            st += _desc('vert', self.vertices.geom, self._vertex_meta)
        if he:
            st += _desc('half_edge', self.half_edges.geom, self._half_edge_meta)
        if e:
            st += _desc('edge', self.edges.geom, self._edge_meta)
        if f:
            st += _desc('face', self.faces.geom, self._face_meta)
        return st

    # -----------------------------------------------------------
    def add_color(self, value, **kwargs):
        self.add_meta(value, 'color', **kwargs)

    def add_meta(self,
                 value,
                 k,
                 half_edges=None,
                 edges=None,
                 faces=None,
                 verts=None,
                 boundary=None):
        """
        set metadata
        # todo - move this to the mesh2d API

        """
        if edges:
            for ix in edges:
                if boundary:
                    bnd_edge = self.boundary.edges[ix]
                    eix = self.index_of_edge(bnd_edge)
                    self._edge_meta[eix][k] = value
                else:
                    self._edge_meta[ix][k] = value

        elif half_edges:
            for ix in half_edges:
                if boundary:
                    bnd_edge = self.boundary.ext_half_edges[ix]
                    eix = self.index_of_half_edge(bnd_edge)
                    self._half_edge_meta[eix][k] = value
                else:
                    self._edge_meta[ix][k] = value

        elif faces:
            for ix in faces:
                self._face_meta[ix][k] = value

        elif verts:
            for ix in verts:
                self._vertex_meta[ix][k] = value

        else:
            raise Exception('')

    @property
    def is_symmetric(self):
        return True

    def anchor(self, vertex=False, edge=False, half_edge=False):
        """ return the vertex, half_edge, edge or face that is
            used to embed this template within a space
        """
        if vertex:
            return self.vertices[0]
        elif half_edge:
            he = self._d_hes[0]
            return self.vertices[he[0]], self.vertices[he[1]]
        else:
            raise NotImplemented('must be half_edge or vertex')

    def align_to(self, target_half_edge_geom):
        """ given a half edge tuple, returns a transformation matrix """
        return r2.align_vec(self.anchor(half_edge=True), target_half_edge_geom)

    @property
    def edge_colors(self):
        return self._edge_meta

    @property
    def half_edge_meta(self):
        return self._half_edge_meta

    def colors(self, edge=None, face=None, half_edge=None):
        """

        """
        res = set()
        if half_edge:
            for geom, data in self._half_edge_meta.items():
                res.add(data.get('color', None))
        return list(res.difference([None]))

    def as_graph(self):
        return self.G


class SimpleTiling(_TemplateBase):
    def colors(self, edge=None, face=None, half_edge=None):
        return []


class BoundaryTemplate(LineString, _TemplateBase):
    def __init__(self, pts,
                 allow_rot=True,
                 max_uses=None,
                 **kwargs):
        LineString.__init__(self, pts)
        _TemplateBase.__init__(self, **kwargs)
        self.max_uses = max_uses
        self.allow_rot = allow_rot

    def transform(self, xform):
        pts = shapely.affinity.affine_transform(self, xform)
        return TemplateTile(pts, **self.meta)

    @property
    def edge_colors(self):
        return self._edge_meta

    @property
    def nodes(self):
        return list(self.coords)

    @property
    def boundary(self):
        from src.cvopt.mesh.boundary import Boundary
        return Boundary(self)

    def vertices(self):
        return self.coords

    def add_color(self, value, half_edges=None, edges=None, faces=None, verts=None):
        """ set interior bountary """
        if edges:
            for ix in edges:
                bnd_edge = self.boundary.edges[ix]
                eix = self.index_of_edge(bnd_edge)
                self._edge_meta[eix]['color'] = value

        elif half_edges:
            bnds = self.boundary.int_half_edges
            for ix in half_edges:
                self._half_edge_meta[bnds[ix]]['color'] = value

        elif faces:
            for ix in faces:
                self._face_meta[ix]['color'] = value

        elif verts:
            for ix in verts:
                self._half_edge_meta[ix]['color'] = value
        else:
            raise Exception('')

