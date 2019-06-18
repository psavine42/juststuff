from .spatial import Mesh2d, rectangle
from cvxpy.utilities.performance_utils import compute_once
from collections import defaultdict as ddict
import networkx as nx
import numpy as np
from .utils import translate


class TemplateTile(Mesh2d):
    """
    Stores a mesh which can have
    """
    def __init__(self, w, h,
                 weight=1,
                 allow_rot=True,
                 max_uses=None,
                 name=None,
                 color=None,
                 access=None):
        g = nx.grid_2d_graph(w + 1, h + 1)
        Mesh2d.__init__(self, g=g)
        self.w = w
        self.h = h
        self.name = name
        self.max_uses = max_uses
        self.allow_rot = allow_rot
        self.weight = weight
        self.color = color
        self.attach_at = []                       # todo remove?
        self.access_at = access if access else [] # todo remove?
        self._face_meta = ddict(dict)
        self._edge_meta = ddict(dict)

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

    def add_color(self, value,
                  half_edges=None,
                  edges=None,
                  faces=None,
                  verts=None):
        """ set interior bountary """
        # if not isinstance(indices, (list, tuple)):
        #     indices = [indices]
        mapping = {'color':{}, 'sign':{}}

        def update_mapping(indexes, bnds=None):
            for ix in indexes:
                ixf = bnds[ix] if bnds else ix
                mapping['color'][ixf] = value

        if edges:
            for ix in edges:
                bnd_edge = self.boundary.edges[ix]
                eix = self.index_of_edge(bnd_edge)
                self._edge_meta[eix]['color'] = value

        elif half_edges:
            bnds = self.boundary.int_half_edges
            for ix in half_edges:
                mapping['color'][bnds[ix]] = value
            nx.set_edge_attributes(self.G, mapping['color'], 'color')

        elif faces:
            for ix in faces:
                self._face_meta[ix]['color'] = value

        elif verts:
            bnds = self.vertices()
            for ix in verts:
                mapping['color'][bnds[ix]] = value
            nx.set_node_attributes(self.G, mapping['color'], 'color')
        else:
            raise Exception('')

    def add_vert_color(self, nodes, value):
        pass

    @property
    def is_symmetric(self):
        return True

    def get_transformations(self):
        transforms = [np.eye(2)]
        if self.allow_rot is True:

            r1 = [np.cos(np.pi/2), np.sin(np.pi/2),
                  np.sin(np.pi / 2), -np.cos(np.pi/2)
                  ]
            transforms += [r1]
            if self.is_symmetric is False:
                r2 = [np.cos(np.pi / 2), np.cos(np.pi / 2),
                      np.cos(np.pi / 2), np.cos(np.pi / 2)
                      ]
                r3 = [np.cos(np.pi / 2), np.cos(np.pi / 2),
                      np.cos(np.pi / 2), np.cos(np.pi / 2)
                      ]
                transforms += [r2, r3]

        for t in transforms:
            hes = self.half_edges()
            hets = []
            for he in hes:
                he_t = np.asarray(he) @ t
                np.round_(he_t, 0)
                hets.append(tuple(map(tuple, he_t.tolist())))
            yield hets

    def transform(self, xform, half_edges=False, edges=False, faces=False, verts=False):
        M = np.eye(3)
        if isinstance(xform, (list, tuple)) and len(xform) == 2:
            M[0, 2] = xform[0]
            M[1, 2] = xform[1]

        if edges:
            return translate(self.edges, xform)
        elif faces:
            return translate(list(self.faces_to_edges().keys()), xform)
        elif verts:
            return translate(self.vertices(), xform)
        return

    @property
    def edge_colors(self):
        return self._edge_meta

    def colors(self, edge=None, face=None, half_edge=None):
        """

        """
        res = []
        if not edge and not face and not half_edge:
            return {data.get('color', None)
                    for _, _, data in self.half_edges_data()}.difference([None])
        elif half_edge:
            for u, v in self.boundary.int_half_edges:
                res.append(self.G[u][v].get('color', 0))
        return res

    @compute_once
    def as_tile(self):
        return rectangle(self.w, self.h)

    def as_graph(self):
        return nx.grid_2d_graph(self.w+1, self.h+1)


