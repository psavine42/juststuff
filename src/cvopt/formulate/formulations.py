import numpy as np
from cvxpy import Variable
import cvxpy as cvx
from collections import defaultdict as ddict
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


class _FormRecursive(object):
    pass


class Formulation(_FormRecursive):
    creates_var = False
    DOMAIN = {}
    META = {'constraint': True, 'objective': True}

    def __init__(self, space,
                 is_constraint=None,
                 is_objective=None,
                 obj=None,
                 name=None):
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
        self._in_dict = ddict(list)  # key valued
        self._actions = []  # outputs lol

        self._name = name
        self.is_constraint = is_constraint
        self.is_objective = True if is_objective is True or \
            obj is not None else False
        self._obj_type = None
        if obj is not None:
            if obj in (cvx.Maximize, cvx.Minimize):
                self._obj_type = obj

        self._generated_constraints = False
        self._generated_objectives = False
        self._obj = None
        self._constr = []
        self._solve_args = {}

    @property
    def solver_args(self):
        return self._solve_args

    @property
    def name(self):
        if self._name is None:
            return self.__class__.__name__

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
            if isinstance(self._actions[0], Variable):
                return self._actions[0]
            else:
                return self._actions[0].X

        alist = []
        for x in self._actions:
            if isinstance(x, Variable):
                alist.append(x)
            else:
                alist.append(x.X)
        return cvx.hstack(alist)

    @property
    def action(self):
        return self.stacked

    @property
    def inputs(self):
        return self._actions

    @inputs.setter
    def inputs(self, args):
        self._actions = args

    @property
    def outputs(self):
        return None

    @property
    def vars(self):
        return list(self._in_dict.values())

    @property
    def space(self):
        """ domain """
        return self._space

    @property
    def num_actions(self):
        l = 0
        for a in self._actions:
            if isinstance(a, Variable):
                l += a.shape[0]
            else:
                l += len(a)
        return l

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
        for cs in self._in_dict.values():
            for c in cs:
                c.reset()

    def register_inputs(self, *others):
        """ """
        for other in others:
            self._in_dict[other.name].append(other)

    def as_objective(self, **kwargs):
        """ objective Expression """
        raise NotImplemented()

    def as_constraint(self, *args):
        """ list of Constraint Expressions """
        raise NotImplemented()

    def gather_input_constraints(self):
        C = []
        for k, formulations in self._in_dict.items():
            for formulation in formulations:
                C += formulation.as_constraint()
        return C

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


def form_canon(cls, *args, **kwargs):
    inst = cls(*args, **kwargs)
    return inst.objective(), inst.constraints()


class Noop(Formulation):
    def as_constraint(self, *args):
        return []

    def as_objective(self, **kwargs):
        return None


class FeasibleSet(Formulation):
    META = {'constraint': False, 'objective': True}

    def __init__(self, **kwargs):
        Formulation.__init__(self, None, is_objective=True, **kwargs)

    def as_objective(self, **kwargs):
        return cvx.Minimize(0)

    def as_constraint(self, *args):
        """ list of Constraint Expressions """
        return []

class ConstaintFormulation(Formulation):
    pass


class ObjectiveFormulation(Formulation):
    pass


class TestingFormulation(Formulation):
    """ just keeping track of what to use aor not"""
    pass


# ------------------------------------------------

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







