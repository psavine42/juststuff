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
    def data(self):
        return {'weight': self.weight, 'name': self.name,
                'color': self.color}

    def colors(self, edge=None, face=None, half_edge=None):
        raise NotImplemented()

    def add_color(self, value, **kwargs):
        raise NotImplemented()

    def add_formulation(self, formulation):
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
        bs = ''
        for i, (u, v) in enumerate(self.edges):
            bnd = 1 if (u, v) in self.boundary.edges else 0
            bs += '\n\t Edge {}, bnd {}, {} {} - color: {}'.format(i, bnd, u, v, self._edge_meta[i].get('color', ''))

        for i, (u, v) in enumerate(self.boundary.int_half_edges):
            data = self.G[u][v]
            bs += '\n\t HalfEdge {}, {} {} - color: {}'.format(i, u, v, data.get('color', ''))
        return bs

    # -----------------------------------------------------------
    def add_color(self, value, half_edges=None, edges=None, faces=None, verts=None):
        """ set interior bountary """
        mapping = {'color': {}, 'sign': {}}
        if edges:
            for ix in edges:
                bnd_edge = self.boundary.edges[ix]
                eix = self.index_of_edge(bnd_edge)
                self._edge_meta[eix]['color'] = value

        elif half_edges:
            for ix in half_edges:
                self.half_edges[(ix, 'color')] = value

        elif faces:
            for ix in faces:
                self._face_meta[ix]['color'] = value

        elif verts:
            for ix in verts:
                mapping[ix]['color'] = value

        else:
            raise Exception('')

    @property
    def is_symmetric(self):
        return True

    def anchor(self, vertex=False, edge=False, half_edge=False):
        if vertex:
            return self.vertices[0]
        elif half_edge:
            he = self._d_hes[0]
            return (self.vertices[he[0]], self.vertices[he[1]])
        else:
            raise NotImplemented('must be half_edge or vertex')

    def align_to(self, target_half_edge_geom):
        """ given a half edge tuple, returns a transformation matrix
        """
        return r2.align_vec(self.anchor(half_edge=True),
                            target_half_edge_geom)

    @property
    def edge_colors(self):
        return self._edge_meta

    def colors(self, edge=None, face=None, half_edge=None):
        """

        """
        res = set()
        if half_edge:
            for geom, data in self.half_edges.data:
                res.add(data.get('color', None))
        return list(res.difference([None]))

    def as_graph(self):
        return self.G


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
        return TemplateTile(pts, **self.data)

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

