from cvxpy.utilities.performance_utils import compute_once, lazyprop
import networkx as nx
import src.geom.r2 as r2
from .mesh2d import *
from collections import OrderedDict as odcit


class MeshMapping(object):
    def __init__(self, space: Mesh2d, tile: Mesh2d, xform):
        """
        map tile mesh onto space mesh with affine transformation
        stores a transformed copy of the tile, the space, and the original tile

        """
        self._data = odcit()
        self.space = space
        self.transform = xform
        self.base = tile
        # -----------------------------------------------
        base_verts = tile.vertices.geom
        hverts = np.dot(self.transform, r2.to_homo(base_verts, dtype=int).T).T  # .T
        new_vert = r2.from_homo(hverts, dtype=int, round=0)
        #
        self._data = [(base_verts[i], new_vert[i]) for i in range(len(base_verts))]
        self.tile = tile.relabel(self._data)

    @lazyprop
    def _data_inv(self):
        """ {new_vert_geom : base_vert_geom}"""
        return [(v, k) for k, v in self._data]

    def _rec_vert(self, new_vert_geom):
        """ given a """
        base_vert_geom = self._data_inv[new_vert_geom]
        base_vert_idx = self._index[base_vert_geom]
        return base_vert_idx

    def _mapping(self, fn, sym, m2t=None, geom=None):
        geom_main = fn.fget(self.space).geom
        geom_tile = fn.fget(self.transformed).geom
        mapping = {}
        for tile_ix, geom in enumerate(geom_tile):
            if geom in geom_main:
                main_ix = geom_main.index(geom)
                # if geom:
                #    g = geom_base[tile_ix]
                # if sym == 'v':
                #     base_ix = self._rec_vert(g)
                # elif sym == 'e':
                #     nix1, nix2 = self._rec_vert(g[0]), self._rec_vert(g[1])
                #     ng1 = self.base.vertices[nix1]
                #     ng2 = self.base.vertices[nix2]
                #     base_ix = self.base.index_of_edge(tuple(sorted([ng1, ng2])))
                # elif sym == 'h':
                #     nix1, nix2 = self._rec_vert(g[0]), self._rec_vert(g[1])
                #     ng1 = self.base.vertices[nix1]
                #     ng2 = self.base.vertices[nix2]
                #     base_ix = self.base.index_of_half_edge((ng1, ng2))
                # else: # face
                #     base_ix = tile_ix
                k, v = (main_ix, tile_ix) if m2t is True else (tile_ix, main_ix)
                # k, v = (main_ix, base_ix) if m2t is True else (base_ix, main_ix)
                mapping[k] = v
            else:
                print('missing', geom)
                return None
        return mapping

    def vertex_map(self, **kwargs):
        return self._mapping(Mesh2d.vertices, 'v', **kwargs)

    def edge_map(self, **kwargs):
        """ { base_edge_index : transformed_edge_index ... } """
        return self._mapping(Mesh2d.edges, 'e', **kwargs)

    def face_map(self, **kwargs):
        """ { base_face_index : transformed_face_index ... } """
        return self._mapping(Mesh2d.faces, 'f', **kwargs)

    def half_edge_map(self, **kwargs):
        return self._mapping(Mesh2d.half_edges, 'h', **kwargs)

    @lazyprop
    def faces(self):
        return list(map(lambda x:x[1], sorted([(k, v) for k, v in self.face_map().items()])))

    @lazyprop
    def edges(self):
        return list(map(lambda x:x[1], sorted([(k, v) for k, v in self.edge_map().items()])))

    @lazyprop
    def vertices(self):
        """ [ transformed_vert_index ... ]"""
        return list(map(lambda x:x[1], sorted([(k, v) for k, v in self.vertex_map().items()])))

    @compute_once
    def is_valid(self):
        m1 = self.vertex_map()
        m2 = self.face_map()
        m3 = self.edge_map()
        m4 = self.half_edge_map()
        return all([x is not None for x in [m1, m2, m3, m4]])

    @property
    def boundary(self):
        return self.tile.boundary

    @property
    def transformed(self):
        return self.tile

    def show(self, save=None):
        """ show the initial and transformed """
        return
