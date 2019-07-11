from .formulations import Formulation
from cvxpy import Variable
import cvxpy as cvx
import numpy as np
import itertools
from src.cvopt.logical import *
from cvxpy.utilities.performance_utils import compute_once


class EdgeColorsAt(Formulation):
    def __init__(self, *args, color=None, he=None, **kwargs):
        """ placement """
        Formulation.__init__(*args, **kwargs)
        self._color = color
        self._half_edges = he

    def as_constraint(self, *args):
        return

    def as_objective(self, maximize=True):
        return


class EdgeConnectivity(Formulation):
    def as_constraint(self, X):
        """
        statement
        'if edge i is active, one of the adjacent edges should be active'

        """
        C = []
        # M = np.array()
        for i in range(self.num_actions):
            # nodes on u, v
            n1, n2 = self.space.edges[i]
            conn_u = list(self.space.G[n1].keys())
            conn_v = list(self.space.G[n2].keys())
            # sum of adjacent edges nmust be >= 1
            adj_vars = []
            for adj_v in conn_u:
                if adj_v == n2:
                    continue
                edge = sorted([n1, adj_v])
                adj_vars.append(self.space.index_of_edge(edge))
            for adj_n in conn_v:
                if adj_n == n1:
                    continue
                edge = sorted([n2, adj_n])
                adj_vars.append(self.space.index_of_edge(edge))
            cvx.sum(cvx.hstack(X[adj_vars]))

        return C


class ActiveEdgeSet(Formulation):
    def __init__(self, *args, edges=[], sums=None, **kwargs):
        """
        edges list of lists of edge indicies

        example:

        M = ....
        edges = [[0, 2, 3], [0, 3, 9]]
        sums = [1, 2]
        cgen = ActiveEdgeSet(M, edges=edges, sums=sums, as_constraint=True)
        cgen.as_constraint()
        >>

        """
        Formulation.__init__(*args, **kwargs)
        if sums is None:
            sums = [1] * len(edges)
        assert isinstance(edges, list)
        assert isinstance(edges[0], list)
        self.edges = edges
        self._max_sum = np.asarray(sums, dtype=int)

    def as_constraint(self, X):
        """
        assumes that indices of edges are elements of actions space

        action sum( X_i, X_j ... X_n) <= mx_sum

        size of A @ X < B
            A: [num_edge_groups, num_actions]
            X: [num_actions]
            B: [num_edge_groups]
        """
        M = np.zeros((len(self.edges), self.num_actions))
        for i, edge_group in enumerate(self.edges):
            M[i, edge_group] = 1
        return [M @ X <= self._max_sum]

    def as_objective(self, maximize=True):
        raise NotImplemented('nyi')


class AdjacencyEC(TestingFormulation):
    def __init__(self, *args, **kwargs):
        """
        INCORRECT DO NOT USE - todo maybe repurpose
        """
        Formulation.__init__(self, *args, **kwargs)

    @compute_once
    def mat(self):
        def match_col(p, mapping):
            """
            return a map of edge colors given action in mapping
            dict { edge_index, signed int }
            """
            template_colors = {}
            colors = p.template.half_edge_meta
            for local_edge, he_index in mapping.half_edge_map().items():
                edge_index = self.space.half_edges.to_edges_index[he_index]
                if local_edge in colors and colors[local_edge].get('color', None):
                    template_colors[edge_index] = colors[local_edge]['color']
            return template_colors

        M = np.zeros((self.num_actions, self.num_actions))
        for i in range(self.num_actions):
            p1, mapping = self[i]
            he_colors = match_col(p1, mapping)
            for j in range(self.num_actions):
                if i == j:
                    continue
                p2, mapping2 = self[j]
                he_colors2 = match_col(p2, mapping2)
                for he_index, color in he_colors.items():
                    if he_index not in he_colors2:
                        continue
                    if color == -he_colors2[he_index]:
                        M[i, j] = 0.2
        return M

    def as_constraint(self, *args):
        M = self.mat()
        C = [M @ self.stacked >= 1]
        return C

    def as_objective(self, maximize=True):
        M = self.mat()
        return cvx.sum(M @ self.stacked)


class AdjacencyEdgeJoint(Formulation):
    has_var = True

    def __init__(self, *args, num_color=None, color_weights=None, **kwargs):
        """
        Creates a variable for each Edge and maximize under the constraints that
        colors of placement will allow it.

        Implemented from 'Computing Layouts with Deformable Templates'
        """
        Formulation.__init__(self, *args, **kwargs)
        self.is_constraint = True
        self.is_objective = True

        self._num_edge = len(self.space.edges)
        self._num_color = num_color
        self._size = self._num_color * self._num_edge

        self.J = Variable(shape=self._size, name='joints', boolean=True)
        W = np.ones((self._num_edge, self._num_color))
        if color_weights is not None and len(color_weights) == num_color:
            for i, w in color_weights:
                W[:, i] = w
        self.W = np.reshape(W, (self._num_edge * self._num_color))

    def __repr__(self):
        st = ''
        st += '{} {}'.format('J', self.J.shape)
        return st

    def as_constraint(self, *args):
        """
        J i, j cannot be present concurrently with every
        tile that is adjacent to E_i but does not have the matching color.
        """
        E1 = np.zeros((self._num_edge, self._num_color, self.num_actions))
        E2 = np.zeros((self._num_edge, self._num_color, self.num_actions))
        for i in range(self.num_actions):
            p1, mapping = self[i]
            he_colors = mapping.match_col()
            for half_edge, color in he_colors.items():
                if color > 0:
                    E1[half_edge, color-1, i] = 1
                elif color < 0:
                    E2[half_edge, -color-1, i] = 1

        E1 = np.reshape(E1, (self._num_edge * self._num_color, self.num_actions))
        E2 = np.reshape(E2, (self._num_edge * self._num_color, self.num_actions))

        # and Constraint formulation
        return [0 <= E1 @ self.stacked + E2 @ self.stacked - 2 * self.J,
                1 >= E1 @ self.stacked + E2 @ self.stacked - 2 * self.J]

    def as_objective(self, maximize=True):
        return cvx.sum(self.W * self.J)





