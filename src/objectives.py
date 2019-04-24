from abc import ABC, abstractmethod
import random
import numpy as np
from collections import defaultdict as ddict

s = 10
random.seed(s)


class _CostFn(ABC):
    @abstractmethod
    def cost(self, layout):
        pass

    def __call__(self, *args, **kwargs):
        return self.cost(*args)


# by hand
def accessble_objective(layout):
    """ each room can  """
    return len(layout.accessability_violations())


def dimension_objective(layout, data):
    """
    state is a layout
    penalize rooms which are not within the prior's constraints.
    constraint is a distribution
    """
    cost = 0.
    for room in layout.rooms():
        cost += log_lik(room.area, data) + log_lik(room.aspect, data)
    return cost


def footprint_objective(layout, footprint):
    """
    falls within the footprint
    cost is the normalized area outside of footprint
    """
    bounds = layout.boundaries()    # shapely polygon
    # expected = problem.footprint    # shapely polygon
    res = footprint.union(bounds)
    return -1 * (footprint.area - res.area) / footprint.area


def convexity_objective(layout):
    """ penalize irregular shapes """
    cost = 0
    for shape in layout.rooms():
        # convexity constraint does not apply
        if shape.prog_type in ['stairs', 'hall']:
            continue

        area = shape.area
        reg = (shape.convex_hull().area - area) / area
        cost += reg
    return -cost


class MetroHeuristic(_CostFn):
    def __init__(self, wdim=0.1, wext=0.1, wbound=0.1, wcvx=0.1, priors=None, footprint=None):
        self._wdim = wdim
        self._wext = wext
        self._wbnd = wbound
        self._wcvx = wcvx
        self._priors = priors
        self._footprint = footprint

    def cost(self, layout) -> float:
        c_dim = self._wbnd * dimension_objective(layout, self._priors)
        c_flr = self._wbnd * footprint_objective(layout, self._footprint)
        c_acc = self._wbnd * accessble_objective(layout)
        c_reg = self._wcvx * convexity_objective(layout)
        return sum([c_dim, c_flr, c_acc, c_reg])


# ---------------------------------------------------
class ConstraintsHeur(_CostFn):
    """
    Treat cost function as sum of constraints applied to
    """
    def __init__(self, problem, wmap=None, default=1, **kwargs):
        self._constraints = problem.constraints()
        self._hist = []
        self._weights = {}
        typeset = {x.__class__.__name__ for x in self._constraints}
        if isinstance(wmap, (float, int)):
            self._weights = {x: wmap for x in typeset}
        elif wmap is None:
            self._weights = {x: random.random() for x in typeset}
        elif isinstance(wmap, dict):
            self._weights = {**{x: default for x in typeset}, **wmap}

        typeset = sorted(list(typeset))
        self._constraint_to_idx = {x: i for i, x in enumerate(typeset)}
        self._idx_to_constraint = {i: x for i, x in enumerate(typeset)}

        # create a matrix of [ents, constraints]
        self._ent_to_idx = {x.name: i for i, x in enumerate(problem.program)}
        self._const_size = len(self._constraint_to_idx)
        self._prob_size = len(problem)

    @property
    def size(self):
        return self._prob_size, self._const_size

    @property
    def upper_bound(self):
        return sum([c.upper_bound * self._weights[c.__class__.__name__]
                    for c in self._constraints])

    @property
    def lower_bound(self):
        return 0.

    def cost(self, layout, show=False):
        ttl = 0
        cost = ddict(float)
        for constraint in self._constraints:
            key = constraint.__class__.__name__
            res = constraint.forward(layout)
            assert isinstance(res, (float, int)), 'invalid constraint result' + str(constraint) + str(res)
            cost[key] += (res * self._weights[key])
            ttl += (res * self._weights[key])
        # self._hist.append(cost)
        if show is True:
            print(cost)
        # print(cost)
        return ttl

    def reward(self, layout, explain=False, encode=False):
        ttl = 0
        avg = 0
        if encode is True:
            mat = np.zeros(self.size)
        cost = ddict(tuple)
        for constraint in self._constraints:
            key = constraint.__class__.__name__
            res = constraint.reward(layout)
            eps = self._weights[key]
            ttl += (res * eps)
            avg += eps
            if encode is True:
                if isinstance(constraint.ent, str):
                    c_i = self._constraint_to_idx[key]
                    e_i = self._ent_to_idx[constraint.ent]
                    mat[e_i, c_i] += res

            if explain is True:
                if isinstance(constraint.ent, str):
                    cost[(constraint.ent, key)] = (res, eps)
                else:
                    print((constraint.ent, key), (res, eps))
        if explain is True:
            return ttl, avg, cost
        if encode is True:

            return ttl / avg, mat
        return ttl / avg

    def info(self):
        st = ''

    def __repr__(self):
        return '{}: {} constraints, \n{} '.format(
            self.__class__.__name__, len(self._constraints), str(self._weights)
        )




class ConstraintBinary(ConstraintsHeur):
    def cost(self, layout, show=False):
        ttl = 0
        cost = {k: 0 for k in self._weights.keys()}
        for constraint in self._constraints:
            key = constraint.__class__.__name__
            res = 0 if constraint.is_satisfied(layout) is True else 1
            assert isinstance(res, (float, int)), 'invalid constraint result' + str(constraint) + str(res)
            cost[key] += (res * self._weights[key])
            ttl += (res * self._weights[key])
        return ttl



def LSQloss(problem):
    """
        def fun_rosenbrock(x):
            return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])

    """
    for room in problem.program:
        bounds = room.uvbounds



