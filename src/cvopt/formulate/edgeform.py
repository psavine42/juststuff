from . import Formulation, TestingFormulation, sum_objectives
from cvxpy.utilities.performance_utils import compute_once
from cvxpy import Minimize, Maximize
from src.cvopt.logical import *
import src.geom.r2 as r2
import numpy as np
from typing import List


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


class ShortestPath(Formulation):
    def __init__(self, args, source, sink):
        """ """
        Formulation.__init__(self, args)
        self._source = source
        self._target = sink
        self.S = np.zeros(len(self.space.vertices), dtype=int)
        self.S[self._target] = 1
        self.S[self._source] = -1
        self.is_objective = True
        self.is_constraint = True

    def as_constraint(self):
        """
        statement
        'if edge i is active, one of the adjacent edges should be active'

        edge to node
        if edge_i is active - then nodes[k1, k2] are active
        if node_k is a sink or source (M @ A)[k] >= 1
        else (M @ A)[k] >= 2

        """
        C = []
        he_dict = self.space.half_edges.base
        UV = np.zeros((len(self.space.vertices), len(he_dict)), dtype=int)
        VW = np.zeros((len(self.space.vertices), len(he_dict)), dtype=int)

        for edge_ix, (u, v) in he_dict.items():
            UV[u, edge_ix] = 1
            VW[v, edge_ix] = 1

        C += [UV @ self.action - VW @ self.action == self.S]
        return C

    def as_objective(self, maximize=True):
        return Minimize(cvx.sum(self.action))

    def display(self):
        data = Formulation.display(self)
        res = [i for i, v in enumerate(self.action.value.tolist()) if v > 0.5]
        data['vertices'] = self._source + self._target
        data['half_edges'] = res
        return data


class ShortestTree(Formulation):
    def __init__(self, space, source, target, **kwargs):
        """

        """
        Formulation.__init__(self, space, **kwargs)
        self.is_objective = True
        self.is_constraint = True
        self._source = source
        self._target = target
        self._X = [Variable(self.space.num_hes, boolean=True)
                   for i in range(len(self._target))]

    def as_constraint(self):
        """
        statement
        'shortest path between source and N targets'

        """
        UV = np.zeros((self.space.num_verts, self.space.num_hes), dtype=int)
        VW = np.zeros((self.space.num_verts, self.space.num_hes), dtype=int)

        for edge_ix, (u, v) in self.space.half_edges.base.items():
            UV[u, edge_ix] = 1
            VW[v, edge_ix] = 1

        C = []
        for i, var in enumerate(self._X):
            vert = np.zeros(self.space.num_verts, dtype=int)
            vert[self._target[i]] = 1
            vert[self._source] = -1
            C += [UV @ var - VW @ var == vert]
        return C

    def as_objective(self, maximize=True):
        return Minimize(cvx.sum(cvx.max(cvx.vstack(self._X), 0)))

    def display(self):
        data = Formulation.display(self)
        res = [i for i, v in enumerate(self.action.value.tolist()) if v > 0.5]
        data['vertices'] = self._source + self._target
        data['half_edges'] = res
        return data

    @property
    def action(self):
        """ Variable(shape=num_half_edges, boolean=True) """
        return cvx.max(cvx.vstack(self._X), 0)


class RouteConstraint(Formulation):
    DOMAIN = {'mesh'}

    def __init__(self, space, routes: List[ShortestTree], **kwargs):
        """
        generic class for adding additional constraints to routing formulations

        todo | these do not have their own objective, but since they take
        todo | formulations as arguments, this should be a tree - also more elegant registration
        """
        Formulation.__init__(self, space, routes, **kwargs)
        self._routes = routes
        self.is_objective = True
        self.is_constraint = True

    def as_constraint(self, *args):
        C = []
        for route in self._routes:
            C += route.constraints()
        return C

    def as_objective(self, **kwargs):
        """ """
        return sum_objectives(self._routes)

    def display(self):
        data = Formulation.display(self)
        for route in self._routes:
            rdata = route.display()
            data['vertices'] += rdata['vertices']
            data['half_edges'] += rdata['half_edges']
        return data


class RouteNoEdgeOverlap(RouteConstraint):
    def __init__(self, space, routes: List[ShortestTree], **kwargs):
        """
        'no routes shall share an edge'
        note - this does not constrain routes to overlap each other vertices

        """
        RouteConstraint.__init__(self, space, routes, **kwargs)

    def as_constraint(self, *args):
        """ edges of routes cannot overlap """
        C = [] # RouteConstraint.constraints(self)
        exprs = []
        for route in self._routes:
            C += route.constraints()
            exprs.append(route.action)
        C += [cvx.sum(cvx.vstack(exprs), 0) <= 1]
        return C


class RouteNoVertOverlap(RouteConstraint):
    def __init__(self, space, routes: List[ShortestTree], **kwargs):
        """
        vertices of routes shall not overlap
        """
        RouteConstraint.__init__(self, space, routes, **kwargs)

    def as_constraint(self, *args):
        """
        todo - FIX!!!! this is not correct at the moment
        """
        C = RouteConstraint.constraints(self)
        edge2vert = np.zeros((self.space.num_verts, self.space.num_hes))

        for edge_ix, (u, v) in self.space.half_edges.base.items():
            edge2vert[u, edge_ix] = 1

        # [num_routes , num_verts]
        verts = cvx.vstack([edge2vert @ route.action for route in self._routes])
        C += [cvx.sum(verts, 0) <= 1]
        return C


class MinTJunctions(Formulation):
    pass


class MaximizeParallel(Formulation):
    def __init__(self, space, threshold=None, fn=None):
        Formulation.__init__(self, space)
        self._dist = threshold
        self._fn = fn

    @compute_once
    def mat(self):
        """ on regular meshes this can be done with indexing, but
        assuming irregular / levels and all sorts of shit, so doing geometrically
        """
        geom = np.asarray(self.space.edges.geom)
        mids = r2.centroid(geom)
        unit = geom[:, 1] - geom[:, 0]
        M = np.zeros(len(geom), len(geom))

        for i in range(len(geom)):
            for j in range(i+1, len(geom)):
                a = r2.angle_between(unit[i], unit[j])
                d = np.sqrt((mids[i, 0] - mids[j, 0]) ** 2 +
                            (mids[i, 1] - mids[j, 1]) ** 2)
                if np.isclose(a, 0) and d >= self._dist:
                    M[i, j] = 1
        return M

    def as_constraint(self, *args):
        """ if """
        return

    def as_objective(self, maximize=True):
        return Maximize(cvx.sum(self.mat() @ self.action))


class ActiveEdgeSet(Formulation):
    def __init__(self, *args, edges=[], sums=None, **kwargs):
        """


        edges: list of lists of edge indicies
        example:

        M = ....
        edges = [[0, 2, 3], [0, 3, 9]]
        sums = [1, 2]
        cgen = ActiveEdgeSet(M, edges=edges, sums=sums, as_constraint=True)
        C = cgen.as_constraint()
        >>

        """
        Formulation.__init__(*args, **kwargs)
        if sums is None:
            sums = [1] * len(edges)
        assert isinstance(edges, list)
        assert isinstance(edges[0], list)
        self._edges = edges
        self._max_sum = np.asarray(sums, dtype=int)

    def as_constraint(self):
        """
        statement
        'no more than max_sum of these edges can be active'

        assumes that indices of edges are elements of actions space

        action sum( X_i, X_j ... X_n) <= mx_sum

        size of A @ X < B
            A: [num_edge_groups, num_actions]
            X: [num_actions]
            B: [num_edge_groups]
        """
        M = np.zeros((len(self._edges), self.num_actions))
        for i, edge_group in enumerate(self._edges):
            M[i, edge_group] = 1
        return [M @ self.action <= self._max_sum]

    def as_objective(self, maximize=True):
        raise NotImplemented('nyi')


class GridLine(Formulation):
    KEY = 'grid_lines'
    DOMAIN = {'discrete'} # both can work

    def __init__(self, space, edges=[], **kwargs):
        """
        maximize the number of edges on gridlines
        :param space:
        :param geom:
        """
        Formulation.__init__(self, space, **kwargs)
        self._edge_indices = edges

    def as_constraint(self):
        """
        'no interior edges can intersect a gridline'

        **note** - the statement:
            'all tile boundary edges must lie on gridlines'
            is either too restrictive, or equivelant to the original problem

        returns list of Constraint Inequality Expressions X <= M
        """
        M = np.ones(self.num_actions)
        for i in range(self.num_actions):
            p, mapping = self[i]
            edges_ti = [x for x in mapping.edges if x in mapping.boundary.edges]
            for v in self._edge_indices:
                if v in edges_ti:
                    M[i] = 0
                    break
        return [self.action <= M]

    def as_objective(self, maximize=True):
        """
        todo 'maximize the number of edges that intersect with the edges marked as gridlines'
        """
        raise NotImplemented('not yet implemented')


class AdjacencyEdgeJoint(Formulation):
    has_var = True

    def __init__(self, *args, num_color=None, color_weights=None, **kwargs):
        """
        Creates a variable for each Edge and maximize under the constraints that
        colors of placement will allow it. Approximates adjacency without a hard constraint.

        Implemented from 'Computing Layouts with Deformable Templates', equation (3)

        # todo eliminate the half edges which are not on interior
        """
        Formulation.__init__(self, *args, **kwargs)
        self.is_constraint = True
        self.is_objective = True

        self._num_edge = len(self.space.edges)
        self._num_color = num_color
        self._size = self._num_color * self._num_edge

        self.J = Variable(shape=self._size, name='joint_colors', boolean=True)
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
        from paper:
            ' J_i,j cannot be present concurrently with every
              tile that is adjacent to E_i but does not have the matching color.'
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


# ------------------------------------------------------------
# Incorrect / Deprecated - keeping yall for a bit
# ------------------------------------------------------------
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
        C = [M @ self.action >= 1]
        return C

    def as_objective(self, maximize=True):
        M = self.mat()
        return cvx.sum(M @ self.action)
