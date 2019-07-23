import numpy as np
from cvxpy import Variable
import cvxpy as cvx
"""

"""


domains = {'discrete', 'mesh', 'continuous'}


def sum_objectives(objs):
    """ utility for gathering objectives into a single expression """
    base = None
    for route in objs:
        o = route.objective()
        if base is None:
            base = o
            continue
        if o is None:
            continue
        base = base + o
    return base


class Formulation(object):
    creates_var = False
    DOMAIN = {}
    META = {'constraint': True, 'objective': True}

    def __init__(self, space, is_constraint=None, is_objective=None, name=None):
        """ Base Class

        inputs: Can be formulations, Variables, geometries or matricies
        actions: Variables outputs
        spaces : the domain in which the formulation exists


        DOMAIN - {
            discrete -> Quadratic assignment problem

        }

        problem_impl : using this implies that

        """
        self._space = space
        self._inputs = []   #
        self._actions = []  # outputs lol

        self.name = name
        self.is_constraint = is_constraint
        self.is_objective = is_objective

        self._generated_constraints = False
        self._generated_objectives = False

    def register_action(self, a):
        """ register an Action/Placement object
        """
        if a not in self._actions:
            self._actions.append(a)

    def set_geom(self, geom):
        raise NotImplemented('not implemented in base class ')

    def __getitem__(self, item):
        """ return the mapping corresponding to the action item """
        if item >= self.num_actions:
            raise Exception('index out bounds')
        cums = 0
        for a in self._actions:
            next_sum = len(a) + cums
            if next_sum > item:
                return a, a.maps[item - cums]
            else:
                cums = next_sum

    @property
    def stacked(self):
        """ returns the vars for all actions concatted to a single row"""
        if len(self._actions) == 1:
            return self._actions[0].X
        return cvx.hstack([x.X for x in self._actions])

    @property
    def action(self):
        return self.stacked

    @property
    def inputs(self):
        return self._actions

    @property
    def outputs(self):
        return None

    @property
    def space(self):
        """ domain """
        return self._space

    @property
    def num_actions(self):
        return sum([len(a) for a in self._actions])

    # display -----------------------
    def __str__(self):
        st = '{}'.format(self.__class__.__name__)
        if self.name:
            st += ':'.format(self.name)
        return st

    def __repr__(self):
        return self.__str__()

    def display(self) -> dict:
        """
        returns a dictionary of how results are to be displayed
        each key contains indicies of corresponding items on self.space
        to add addtional attributes, each list entry can be tuple of (index, dict)
        """
        return {'vertices': [],
                'half_edges': [],
                'faces': [],
                'edges': []}

    def describe(self, **kwargs):
        s = ''
        for s in self._inputs:
            s += s.describe(**kwargs)
        return s

    # generators -----------------------
    def reset(self):
        """ reset the generator states - used for relaxation problems"""
        self._generated_constraints = False
        self._generated_objectives = False
        for c in self._inputs:
            c.reset()

    def as_objective(self, **kwargs):
        """ objective Expression """
        raise NotImplemented()

    def as_constraint(self, *args):
        """ list of Constraint Expressions """
        raise NotImplemented()

    def constraints(self):
        if self._generated_constraints is True \
                or self.is_constraint is False:
            return []
        self._generated_constraints = True
        return self.as_constraint()

    def objective(self):
        if self._generated_objectives is True \
                or self.is_objective is False:
            return None
        self._generated_objectives = True
        return self.as_objective()


class ConstaintFormulation(Formulation):
    pass


class ObjectiveFormulation(Formulation):
    pass


class Stage(Formulation):
    pass


class TestingFormulation(Formulation):
    """ just keeping track of what to use aor not"""
    pass


# ------------------------------------------------
class NoOverlappingFaces(Formulation):
    DOMAIN = {'discrete'}

    def as_constraint(self, *args):
        M = np.zeros((len(self.space.faces), self.num_actions), dtype=int)
        cnt = 0
        for p in self._actions:
            for i, mapping in enumerate(p.maps):
                ixs = list(mapping.face_map().values())
                M[ixs, cnt] = 1
                cnt += 1
        actions = self.stacked # cvx.hstack([x.X for x in self._actions])
        return [M @ actions <= 1]


class VerticesNotOnInterior(Formulation):
    DOMAIN = {'discrete'}

    def __init__(self, space, vertices=[], weights=[], **kwargs):
        """
        statement:
            'These vertices must be on the boundary of tile placements'
            -- or--
            'These vertices cannot be on the interior of tile placements'

        :param space: mesh space
        :param vertices: list of indices
        """
        Formulation.__init__(self, space, **kwargs)
        self._vertices = vertices
        if weights:
            assert len(vertices) == len(weights), \
                'if weights are provided, must be same dim as vertices'
        else:
            weights = np.ones(len(vertices))
        self._weights = weights

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

        add M @ X term to the problem objectives

        implementation:
            create Variable for each vertex
            for placement X_p,i if placement violates constraint, 0, else 1
            if (X_0 or X_1 or ... or X_n ) cause a violation mul operation results in 0

        example:

        """
        if maximize is False:
            raise NotImplemented('not yet implemented')

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
        return objective


class TileLimit(Formulation):
    def __init__(self, space, tile, upper=None, lower=None, **kwargs):
        """

        """
        Formulation.__init__(self, space, **kwargs)
        self._tile_index = tile
        self._upper = upper
        self._lower = lower
        self.is_constraint = True

    def as_constraint(self, *args):
        C = []
        if self._upper:
            C += [cvx.sum(self._actions[self._tile_index].X) <= self._upper]
        if self._lower:
            C += [cvx.sum(self._actions[self._tile_index].X) >= self._lower]
        return C


class IncludeNodes(Formulation):
    def __init__(self, space, nodes=[], **kwargs):
        """ nodes must be included in the tree
            in other words, atleast one edge for each node
            must be active
        """
        Formulation.__init__(self, space, **kwargs)
        self.nodes = nodes

    def as_constraint(self, X_edges):
        C = []
        for i, n in enumerate(self.nodes):
            if not isinstance(n, int):
                # its an index
                node_geom = n
                # node_i = self.space.index_of_vertex(n)
            else:
                node_geom = self.space.vertices()[n]

            active = []
            for adj_node in list(self.space.G[node_geom].keys()):
                edge = sorted([node_geom, adj_node])
                active.append(self.space.index_of_edge(edge))
            C += [cvx.sum(cvx.hstack(X_edges[active])) >= 1]
        return C


class DeadZone(Formulation):
    def __init__(self, space, edges=None, ):
        Formulation.__init__(self, space)

    def as_constraint(self, *args):
        pass


class TileAt(Formulation):
    def __init__(self, *args, placement=None, he=None, **kwargs):
        """ placement """
        Formulation.__init__(*args, **kwargs)
        self._placement = placement
        self._half_edge = he

    def as_constraint(self, *args):
        return

    def as_objective(self, maximize=True):
        return







