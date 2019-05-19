from abc import ABC, abstractmethod
import random
import numpy as np
from collections import defaultdict as ddict
from collections import OrderedDict as odict
from src.problem.objective_utils import *
from skimage import measure
from skimage import filters

# todo is needed here?
# import torch

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
        if show is True:
            print(cost)
        return ttl

    def reward(self, *args, **kwargs):
        return self._reward(*args, **kwargs)

    def _reward(self, layout, explain=False, encode=False):
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

    def __repr__(self):
        return '{}: {} constraints, \n{} '.format(
            self.__class__.__name__, len(self._constraints), str(self._weights)
        )


class ObjectiveModel(ConstraintsHeur):
    def __init__(self, *args, **kwargs):
        ConstraintsHeur.__init__(self, *args, **kwargs)
        self.__prev_state = None

    def reward(self, layout):
        return


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


class DiscreteSpaceObjective:
    def __init__(self, targets, incomplete_reward=-1.):
        self._target = list(sorted(targets))
        self.num_goals = len(self._target)
        self.sum_goals = sum(self._target)
        self._f = 1 - np.e
        self.goal = np.asarray(self._target) / self.sum_goals
        self.incomplete_reward = incomplete_reward

    def reward(self, layout, encode=False, avg=True, **kwargs):
        """ layout is a DiscreteLayout"""
        # layout
        areas = list(layout._areas)
        if len(areas) < self.num_goals:
            if encode is True:
                return self.incomplete_reward, np.array([0., 0.])
            return self.incomplete_reward

        areas = list(sorted(map(lambda x: len(x), areas)))
        r = np.zeros(self.num_goals)
        for i, (pred, target) in enumerate(zip(areas, self._target)):
            val = 1 - abs(target - pred) / target
            r[i] = np.exp(val) + self._f

        t = np.sum(r)
        if avg:
            t /= self.num_goals
        if encode is True:
            return t, r
        return t

# ------------------------------------------------------------------------
#




def binary_edge(mat):
    edges = mat - skm.erosion(mat)


# todo - edge length
# todo - global overlaps
# todo -  'overlap': True
_cdict = {0: {'adj': {1}, 'aspect': 0.5, 'area': 0.5, 'convex': True, },
          1: {'adj': {0}, 'aspect': 0.5, 'area': 0.5, 'convex': True,  }

          }


def generate_constraint_dict(num_spaces=10, x=20, y=20, return_state=False, area=1.):
    def item():
        nadj = random.randint(0, 3)
        return {'aspect': random.uniform(0.5, .99),
                'area': random.uniform(0.1, .5),
                'convex': True,
                'adj': [random.randint(0, num_spaces-1) for _ in range(nadj)]
                }

    items = {i: item() for i in range(num_spaces)}
    sum_a = sum((x['area'] for x in items.values()))
    for i in items.keys():
        items[i]['area'] /= sum_a
        items[i]['adj'] = {i+1} if i < (len(items) - 1) else {0}
    return items


def generate_footprint_constr(size, ones=True):
    """ create a footprint >= size"""
    from src.building import Area
    c = np.asarray([int(random.uniform(0.4, 0.6) * size) for _ in range(2)])
    m = np.min(c)
    footprint = Area([(c[0] - m, c[1] - m), (c[0] - m, c[1] + m),
                      (c[0] + m, c[1] + m), (c[0] + m, c[1] - m)
                ])
    return footprint


# def problem0(size):
#     return [{'aspect': 0.5, 'area': 0.5, 'convex': 1, 'adj': 1 if i == 0 else 0}
#             for i in range(2)]


# def problem1(num_spaces=3, x=20, y=20, return_state=False):
#     s = np.zeros((num_spaces, x, y))
#     bbox = [[0, 0, x-1, y-1]]
#     seed = random.choice([0, 1])
#     for i in range(num_spaces - 1):
#         # kd-split
#         # print(seed)
#         split_l = random.uniform(.2, .8)
#         if seed == 0:  # X
#             # s[:, 0:int(split_l *x), :] = 0
#             s[i, 0:int(split_l *x), :] = 1
#         else:
#             # s[:, :, 0:int(split_l * x)] = 0
#             s[i, :, 0:int(split_l * x)] = 1
#
#         if i == 0:
#             s[1:] -= s[i]
#         else:
#             s[0:i] -= s[i]
#             s[i+1:] -= s[i]
#         s = np.clip(s, 0, 1)
#         seed ^= 1
#
#     s[num_spaces-1] = 1
#     for i in range(num_spaces - 1):
#         s[num_spaces - 1] -= s[i]
#
#     state = np.clip(np.abs(s), 0, 1).astype(int)
#     # print(state)
#     bounds = max_boundnp(state)
#     dilations = [skm.dilation(state[i]) for i in range(num_spaces)]
#     items = {}
#     for i in range(num_spaces):
#         area = np.sum(state[i])
#         adj = []
#         for j in range(num_spaces):
#             if i != j and len(np.where(state[i] + dilations[j] == 2)[0]) > 0:
#                 adj.append(j)
#         items[i] = {'aspect': aspect(bounds[i][2:]),
#                       'area': area / (x * y),
#                       'convex': convexity(state[i], area=area),
#                       'adj': adj}
#     if return_state is True:
#         return items, state
#     return items


class AdjConstraints:
    # todo finish
    def __init__(self, adj_dict):
        """ {0: {2, 2}"""
        self._adj = adj_dict
        self.N = len(adj_dict)
        self._adj_mat = np.zeros((self.N, self.N))
        for k, v in adj_dict.items():
            self._adj_mat[k, v] = 1
            self._adj_mat[v, k] = 1
        self._adj = np.stack(np.where(self._adj_mat == 1))
        self.__dilations, self.__rooms = None, None

    def reset(self):
        self.__rooms = None
        self.__dilations = None

    def to_input(self):
        return self._adj_mat

    def reward(self, layout, action=None):
        state = layout.active_state
        if self.__dilations is None:
            self.__dilations = np.zeros_like(state)
            for i in range(self.__dilations.shape[0]):
                self.__dilations[i] = skm.dilation(layout.active_state[action])

        else:
            self.__dilations = skm.dilation(layout.active_state[action])

        # for
        return


class DiscProbDim:
    def __init__(self, problem, constraints, weights=None, use_comps=False):
        const = constraints if constraints else _cdict

        self.keys = sorted(list(set().union(*(d.keys() for d in const.values()))))
        self.size = (len(const.keys()), len(self.keys))

        self._constraints = [[None] * len(self.keys) for _ in range(self.size[0]) ]

        self._adj_mat = np.zeros((self.size[0], self.size[0]))
        # print(self.keys)
        if 'adj' in self.keys:
            for k, cs in const.items():
                for v in cs.get('adj', []):
                    self._adj_mat[k, v] = 1
                    self._adj_mat[v, k] = 1

        # print(self.size)
        for i in range(self.size[0]):
            for j, k in enumerate(self.keys):
                self._constraints[i][j] = const[i].get(k, None)

        self._wmat = np.ones(self.size)
        if weights:
            for i, w in enumerate(weights):
                self._wmat[:, i] = w

        # optional Flags
        self.use_comps = use_comps

        # self.area = self.size[0] * (self.size[1] - 1) + self.size[0] ** 2 if 'adj' in self.keys else
        self.area = self.size[0] * self.size[1]
        self._f = 1 - np.e
        self.__bounds, self.__rooms, self.__fp_area, self.__dilated = None, None, None, None
        self.reset()

    def to_input(self):
        """"""
        cr = np.asarray(self._constraints.copy())
        if 'adj' in self.keys:
            idx = self.keys.index('adj')
            cr[:, idx] = 1
        return cr.astype(float)

    def reset(self):
        self.__fp_area = None
        self.__dilated = [None] * self.size[0]
        self.__rooms = [None] * self.size[0]
        self.__bounds = [None] * self.size[0]
        self.__areas = [None] * self.size[0]

    def __repr__(self):
        st = str(self.keys)
        st += '\n'
        return st

    @property
    def fp_area(self):
        return self.__fp_area

    def reward(self, layout, encode=True, avg=True, action=None):
        """ return a matrix of size [Problem.length, num_constraints]
            representing reward for mat[space_i, constraint_j]
        """
        if action is None:
            self.__fp_area = np.where(layout.footprint == 1)[0].shape[0]
        rooms = layout.rooms(stack=True)    # these are binary images
        bounds = max_boundnp(rooms)
        areas = np.sum(rooms, axis=(1, 2))
        dilations = [skm.dilation(r) for r in rooms]
        # todo - this should be cached - and only the
        res = np.zeros(self.size)
        for i in range(rooms.shape[0]):

            # if no area, skip - its all zeros
            if areas[i] == 0:
                continue

            for j, (k, v) in enumerate(zip(self.keys, self._constraints[i])):
                if k == 'adj':
                    if len(v) == 0:
                        res[i, j] = 1
                        continue
                    n = 0
                    # todo this should be a seperate features set and eval
                    for other in v:
                        adj = dilations[i] + rooms[other]
                        n += 1 if np.where(adj == 2)[0].shape[0] >= 3 else 0
                    res[i, j] = n / len(v)

                elif k == 'aspect':
                    xmin, ymin, xmax, ymax = bounds[i][2:]
                    aspect = min(xmax - xmin, ymax - ymin) / max(xmax - xmin, ymax - ymin)
                    res[i, j] = 1 if aspect >= v else 0

                elif k == 'area':
                    res[i, j] = 1 - abs(v - areas[i] / self.__fp_area)

                elif k == 'convex':
                    cvs = np.sum(skm.convex_hull_image(rooms[i]))
                    res[i, j] = 1 - (cvs - areas[i]) / cvs

        # apply weights
        # todo add masking for missing constraints

        if self.use_comps:
            # penalty for disconnected components
            for n in [num_components(rooms[i]) for i in range(len(rooms))]:
                # print(n)
                # if n == 0:
                #    res[i] = 0.
                if n != 0:
                    res[i] /= n
            # res = res / ncomps[: None]

        res = np.multiply(res, self._wmat)
        reward = np.sum(res)

        if avg:
            reward /= self.area
            # there is a bonus for having all
            # reward *= len(np.where(areas > 0.0)) / len(areas)
            # NOTE - DO NOT ADJUST TO [-1, 1] - this leeds to zero gradients at ~0
            # reward = np.exp(reward) + self._f
        if encode:
            return reward, res
        return reward


class DiscreteProbArea:
    @staticmethod
    def encode(layout, area_targets):
        """
        create tensor of size [S, N, M] with

        """
        S = len(layout.problem)
        N, M = layout.size
        assert len(area_targets) == S

        tensor = np.zeros((S, N, M))
        ttl = sum(area_targets)
        for i, c in enumerate(area_targets):
            tensor[i] = c/ttl

        if layout.footprint is not None:
            tensor[:, np.where(layout.footprint == 1)] = 1
        return tensor


class DiscreteProbConvexity:
    def wrt_dp_area(self, layout, area_prob_tensor):
        # [ 3, N, M ]
        state_tensor = layout.state
        # [ S, N, M ]
        ap_wo_lines = np.clip(area_prob_tensor - state_tensor[1], 0, 1)
        mus = np.mean()


class DiscProbAdjacency:
    def encode(self, x):
        """ given """


class DiscProbAspect:
    def encode(self, x):
        """ options here are:
        1). consider the max of each [:, i, j]

        """
    def reward(self, bbox_params, param):
        pass





