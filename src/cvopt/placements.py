from .spatial import _VarGraphBase, Mesh2d
from collections import defaultdict as ddict
import cvxpy as cvx
from cvxpy import Variable
import numpy as np
from src.cvopt.utils import translate


class Placement(_VarGraphBase):
    def __init__(self, parent, template, edge_placements=[], face_placements=[],
                 xforms=[]):
        """
        :param template: <class>TemplateTile
        :param edge_placements: list of list of indices of edge
        :param face_placements: list of list of indices of face

        self.X[i] will
        """
        _VarGraphBase.__init__(self)
        self._maximize_adj = True
        self.P = parent
        self.template = parent.T[template]
        self.index = self.template.color
        self.placement_edges = edge_placements
        self.xforms = xforms
        self.placements = face_placements
        self.X = Variable(shape=len(face_placements), boolean=True, name='plcmt.{}'.format(self.template.color))
        self.J = Variable(shape=len(face_placements), pos=True, name='plcmt.{}'.format(self.template.color))

    def __len__(self):
        return self.X.shape[0]

    def _mutex_placements(self):
        lp = len(self.placements)
        sets = [set(x) for x in self.placements]
        mutexes = ddict(set)
        for i in range(lp):
            for j in range(i+1, lp):
                if len(set(sets[i]).intersection(sets[j])) > 0:
                   mutexes[i].add(j)
                   mutexes[j].add(i)
        return mutexes

    def placements_for_face(self, face_index, vars=False):
        """ which placements result in face_i == self.color """
        res = []
        for own_index, faces in enumerate(self.placements):
            if face_index in faces:
                if vars is True:
                    res.append(self.X[own_index])
                else:
                    res.append(own_index)
        return res

    @property
    def solution(self):
        x = np.where(np.asarray(self.X.value, dtype=int) == 1)[0]
        return [self.placements[i][0] for i in x]

    @property
    def constraints(self):
        """
        F - Set of Faces
        E - Set of Edges
        H - Set of Half Edges
        V - Set of Vertices
        """
        C, N = [], len(self.placements[0])
        face_vars = self.adj_vars(face=True)
        edge_vars = self.adj_vars(edge=True)

        # X_i => f(F)
        # self on face
        for i, ixs in enumerate(self.placements):
            # if X_i is 1 -> N  <= corresponding faces should sum to = N
            # if X_i is 0 -> 0  <= corresponding faces should sum to < N
            C += [N * self.X[i] <= cvx.sum(cvx.vstack([face_vars[x][self.index] for x in ixs]))]

        # X_i => f(X)
        # self on self - if X_i is placed, X_j cannot be placed
        for i, v in self._mutex_placements().items():
            # if X_i is 1 -> 0        >= sum X_j will == 0
            # if X_i is 0 -> N        >= sum X_j will >= 0
            C += [(1 - self.X[i]) * N >= cvx.sum(cvx.vstack(self.X[list(v)]))]
        # cannot exceed max times its used
        # C += self._jentries()
        C += [cvx.sum(self.X) <= self.template.max_uses]
        return C

    @property
    def G(self) -> Mesh2d:
        return self.P.G

    def boundary_xformed(self, index, edges=None, half_edges=None, vertices=None, geom=None):
        """ X_index -> [Edge, HalfEdge] indices once xform[index] is applied

            indicies of entities in M once X_i has been applied on self.template
        """
        if edges:
            f1 = self.template.boundary.edges
            f2 = self.G.index_of_edge
        elif half_edges:
            f1 = self.template.boundary.int_half_edges
            f2 = self.G.index_of_half_edge
        elif vertices:
            f1 = self.template.boundary.vertices
            f2 = self.G.index_of_vertex
        else:
            raise Exception
        if geom:
            f2 = lambda x:x
        return [f2(x) for x in translate(f1, self.xforms[index])]

    def transformed(self, index,
                    edges=None,
                    faces=None,
                    half_edges=None,
                    vertices=None,
                    interior=None,
                    exterior=None,
                    geom=None):
        if edges:
            d = self.template.edges
            f2 = self.G.index_of_edge
        elif half_edges:
            d = self.template.half_edges()
            f2 = self.G.index_of_half_edge
        elif vertices:
            d = self.template.vertices()
            f2 = self.G.index_of_vertex
        else:
            raise Exception
        if geom:
            f2 = lambda x: x

        if exterior:
            return self.boundary_xformed(index, edges=edges, half_edges=half_edges, vertices=vertices)
        base = [f2(x) for x in translate(d, self.xforms[index])]
        if interior:
            ext = self.boundary_xformed(index, edges=edges, half_edges=half_edges, vertices=vertices)
            return [x for x in base if x not in ext]
        return base

    def _implied_faces_colors(self, i):
        """
        # for i, face_ixs in enumerate(self.placements):
        # compute edges that will be colored by placement I
        # if X_i is 1 => Tiling_i st that

        returns list  [(face_index, color_index) ...]
        """
        face_ixs = self.placements[i]
        implied_face_colors = []
        for bnd_ix, edge_ix in enumerate(self.boundary_xformed(i, edges=True)):
            tmpl_ix = self.template.index_of_edge(self.template.boundary.edges[bnd_ix])
            edge_col = self.template.edge_colors.get(tmpl_ix, {}).get('color', None)
            if edge_col is None:
                continue
            for face_i in self.G.edges_to_faces()[edge_ix]:
                # face_i is indexed to Parent
                if face_i in face_ixs:
                    continue
                implied_face_colors.append((face_i, edge_col))
        return implied_face_colors

    @property
    def objective_max(self):
        base = self.template.weight * cvx.sum(self.X)
        return base

    # --------------------------------------------------------------
    def _jentries(self):
        C = []
        C += [self.J <= self.X]
        for i, face_ixs in enumerate(self.placements):
            implied_face_colors = self._implied_faces_colors(i)
            N = len(implied_face_colors)
            if N == 0:
                continue

            conds_for_i = []
            for implied_face_ix, implied_color in implied_face_colors:
                # one of these must be true for J to be 1
                Xs_other = self.P.placements[implied_color].placements_for_face(implied_face_ix, vars=True)
                conds_for_i += [cvx.sum(cvx.vstack(Xs_other))]

        return C

