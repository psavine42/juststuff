from cvxpy.utilities.performance_utils import compute_once, lazyprop
import networkx as nx
import src.geom.r2 as r2
from .mesh2d import *
from collections import OrderedDict as odcit
import src.cvopt.utils as u
import matplotlib.pyplot as plt


class MeshMapping(object):
    def __init__(self, space: Mesh2d, tile: Mesh2d, xform):
        """
        map tile mesh onto space mesh with affine transformation
        stores a transformed copy of the tile, the space, and the original tile

        attributes:
            space: Mesh2d - the base space
            base: Mesh2d - the mesh to be transformed
            tile: Mesh2d - the mesh after transform is applied

        """
        self._data = odcit()
        self.space = space
        self.transform = xform
        self.base = tile
        # -----------------------------------------------
        base_verts = tile.vertices.geom
        hverts = np.dot(self.transform, r2.to_homo(base_verts, dtype=int).T).T  # .T
        new_vert = r2.from_homo(hverts, dtype=int, round=0)
        # all that SHOULD need to happen is the vertices get relabeled
        #
        self._data = [(base_verts[i], new_vert[i]) for i in range(len(base_verts))]
        self.tile = tile.relabel(self._data)

    @lazyprop
    def _data_inv(self):
        """ {new_vert_geom : base_vert_geom}"""
        return [(v, k) for k, v in self._data]

    # def _rec_vert(self, new_vert_geom):
    #     """ given a """
    #     base_vert_geom = self._data_inv[new_vert_geom]
    #     base_vert_idx = self._index[base_vert_geom]
    #     return base_vert_idx

    def _mapping(self, fn, m2t=None, geom=None, mgeom=None, tgeom=None):
        """

        opts:
            default: tile_index -> index_in_space
            m2t:     index_in_space -> tile_index
            geom:    return geometry of both indixes instead of elements
            mgeom:   instead of index of space element, use geometry
            tgeom:   instead of index of transformed element, use geometry

        examples:
        default (t2m):
            {0:3, ... }
        geom:
            {((0, 0), (0, 1)) : ((1, 3), (2, 3)) ... }

        """
        geom_base = fn.fget(self.base).geom
        geom_main = fn.fget(self.space).geom
        geom_transformed = fn.fget(self.transformed).geom
        mapping = {}
        for trns_ix, trns_geom in enumerate(geom_transformed):
            if trns_geom in geom_main:
                main_ix = geom_main.index(trns_geom)
                if geom:
                    # returning base tile geom -> geom after transform
                    el_trns, el_main = geom_base[trns_ix], trns_geom
                elif mgeom:
                    el_trns, el_main = trns_ix, trns_geom
                elif tgeom:
                    el_trns, el_main = geom_base[trns_ix], main_ix
                else:
                    el_trns, el_main = trns_ix, main_ix

                k, v = (el_main, el_trns) if m2t is True else (el_trns, el_main)
                # k, v = (main_ix, base_ix) if m2t is True else (base_ix, main_ix)
                mapping[k] = v
            else:
                # print('missing', trns_geom)
                return None
        return mapping

    def vertex_map(self, **kwargs) -> dict:
        """ dict { base_edge_index : transformed_edge_index ... } """
        return self._mapping(Mesh2d.vertices, **kwargs)

    def edge_map(self, **kwargs) -> dict:
        """ { base_edge_index : transformed_edge_index ... } """
        return self._mapping(Mesh2d.edges, **kwargs)

    def face_map(self, **kwargs) -> dict:
        """ { base_face_index : transformed_face_index ... } """
        return self._mapping(Mesh2d.faces, **kwargs)

    def half_edge_map(self, **kwargs) -> dict:
        return self._mapping(Mesh2d.half_edges, **kwargs)

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

    def show(self, save=None, size=7, he=False):
        """ show the initial and transformed
        opts:
            he: (boolean) if True, then label half edges instead of edges
        """
        if self.is_valid() is False:
            raise Exception('cannot display invalid maapping')
        if isinstance(size, int):
            size = (size, size)
        fig, ax = plt.subplots(1, figsize=size)
        ax = u.draw_edges(self.space, ax, color='gray')

        if he:
            ax = u.draw_half_edges(self.base, ax, label=True, color='black')
            ax = u.draw_half_edges(self.transformed, ax, label=True, color='red')
        else:
            ax = u.draw_edges(self.base, ax, label=True, color='black')
            ax = u.draw_edges(self.transformed, ax, label=True, color='red')

        u.finalize(ax, save=save)
        print(self.__repr__())
        # todo - why cant I call this from finalize in anothter File??
        if isinstance(save, str):
            plt.savefig(save)
            plt.clf()
            plt.close()
        else:
            plt.show()
        return

    def __repr__(self):
        st = ''
        for k, v in self.vertex_map().items():
            st += '\nv.{} -> {}'.format(k, v)
        for k, v in self.edge_map().items():
            st += '\ne.{} -> {}'.format(k, v)
        for k, v in self.face_map().items():
            st += '\nf.{} -> {}'.format(k, v)
        return st

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


