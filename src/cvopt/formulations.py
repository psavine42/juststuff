import numpy as np
from cvxpy import Variable
import cvxpy as cvx
"""

"""


domains = {'mesh', 'continuous'}


class Formulation(object):
    KEY = ''
    DOMAIN = {}

    def __init__(self, space):
        self._space = space
        self._actions = []

    def register_action(self, a):
        if a in self._actions:
            print('already registered ')
        self._actions.append(a)

    def as_objective(self, maximize=True):
        pass

    def as_constraint(self, *args):
        """ list of Constraint Expressions """
        pass


class AdjacencyEdgeColor(Formulation):
    KEY = 'adj_edge_color'
    DOMAIN = {}
    pass


class AdjacencyFace(Formulation):
    pass


class GridLine(Formulation):
    KEY = 'grid_lines'
    DOMAIN = {'mesh'} # both can work
    """ 
    maximize the number of edges on gridlines 
    """
    def __init__(self, space, edges=[]):
        """

        :param space:
        :param geom:
        """
        Formulation.__init__(self, space)
        self._edge_indices = edges

    def as_constraint(self, placements):
        """
        'no interior edges can intersect the gridline'

        **note** - the statement:
            'all tile boundary edges must lie on gridlines'
            is either too restrictive, or equivelant to the original problem

        returns list of Constraint Inequality Expressions X <= M
        """
        C = []
        for p in placements:
            M_p = np.ones(len(p), dtype=int)
            for i in range(len(p)):
                edges_ti = p.transformed(i, edges=True, interior=True)
                # check if v is in the boundary
                for v in self._edge_indices:
                    if v in edges_ti:
                        M_p[i] = 0
                        break
            C += [p.X <= M_p]
        return C

    def as_objective(self, maximize=True):
        """
        'maximize the number of edges that intersect with the edges marked as gridlines'
        """
        return


class VerticesOnBound(Formulation):
    KEY = 'columns'
    DOMAIN = {'mesh'}  # both can work

    def __init__(self, space, vertices=[], weights=[]):
        """
        statement:
            'These vertices must be on the boundary of tile placements'
            -- or--
            'These vertices cannot be on the interior of tile placements'

        :param space: mesh space
        :param vertices: list of indices
        """
        Formulation.__init__(self, space)
        self._vertices = vertices
        if weights:
            assert len(vertices) == len(weights), \
                'if weights are provided, must be same dim as vertices'
        else:
            weights = np.ones(len(vertices))
        self._weights = weights

    @property
    def num_actions(self):
        return sum([len(a) for a in self._actions])

    def as_constraint(self):
        """
        if the vertex is strictly contained by a transformation M_p,i
            then M_p,i => 0
        :returns list of Constraint Expressions X <= M
        """
        C = []
        for p in self._actions:
            M_p = np.ones(len(p), dtype=int)
            for i in range(len(p)):
                new_verts = p.transformed(i, interior=True, vertices=True)
                for v in self._vertices:
                    if v in new_verts:
                        # if any marked vertex is found to violate the placement,
                        # that placement is constrained to be 0
                        M_p[i] = 0
                        break
            C += [p.X <= M_p]
        return C

    def as_objective(self, maximize=True):
        """
        'maximize the number of tile boundary vertices that intersect with the marked vertices'

        implementation:
            create Variable for each vertex
            for placement X_p,i if placement violates constraint, 0, else 1
            if (X_0 or X_1 or ... or X_n ) cause a violation mul operation results in 0

        """
        # Variable(shape=len(self._vertices), boolean=True)
        # M @ X  -> { v | v in {0, 1}}
        X = []
        M = np.zeros((len(self._vertices), self.num_actions))

        cnt = 0
        for p in self._actions:
            X.append(p.X)
            for j in range(len(p)):
                new_verts = p.transformed(j, exterior=True, vertices=True)
                for k, v in enumerate(self._vertices):
                    if v in new_verts:
                        M[k, cnt] = 1
                cnt += 1

        objective = cvx.sum(self._weights * (M @ cvx.vstack(X)))
        return {'constraints': None,
                'objective': objective}



