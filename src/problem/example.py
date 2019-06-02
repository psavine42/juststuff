import networkx as nx
from src.building import Area
from src.layout_ops import *
from .base import *
import math
from src.problem.objective_utils import *


# ---------------------------------------------------------------------
def setup_2room():
    """ how to setup the problem """

    footprint = Area([(0, 0), (1, 0), (5, 1)], name='footprint')

    problem = Problem()
    room1 = ProgramEntity('r1', 'room', problem=problem)
    problem.add_constraint(MaxDimConstraint())

    room2 = {'name': 'r2', 'program': 'xr'}

    pass

epi = math.e / math.pi
_2room_adj = {
    'r1': {'adj': {'r2'}, 'prog': 'room', 'area': 150, 'aspect': 0.6},
    'r2': {'adj': {'r1'}, 'prog': 'room', 'area': 150, 'aspect': 0.6},
}

_h1_adj = {
    'bed1': {'adj': {'hall'}, 'prog': 'bed', 'area': 150, 'aspect': epi },
    'bed2': {'adj': {'hall'}, 'prog': 'bed', 'area': 100, 'aspect': epi },

    'hall': {'adj': {'bed1', 'bed2', 'living'}, 'prog': 'hall'},

    'living': {'adj': {'dining'}, 'prog': 'room', 'area': 200, 'aspect': epi},

    'kitchen': {'adj': {'dining'}, 'prog': 'kitchen', 'area': 150, 'aspect': epi},

    'dining': {'adj': {'hall'}, 'prog': 'room', 'area': 150, 'aspect': epi},

    'bath1': {'adj': {'hall'}, 'prog': 'bath', 'area': 80, 'aspect': epi},
    'bath2': {'adj': {'bed1'}, 'prog': 'bath', 'area': 70, 'aspect': epi },
}


_COSTS_WALL = {
    'exterior_wall': 20,
    'partition': 10,
    'shaft': 15,
    'struct': 22,
}



def generate_statistics(size):
    """ make a layout and get its statistics """
    c = np.asarray([int(random.uniform(0.4, 0.6) * size) for _ in range(2)])
    m = np.min(c)

    for i in range(size):
        # kd-split
        x = random.choice([0, 1])
        split_l = random.uniform(.1, .9),
        pass

    state = np.zeros(size, h, w)
    bounds = max_boundnp(state)
    for i in range(len(state)):
        area = np.sum(state[i])
        item = {'aspect': aspect(state[i]),
                'area': area,
                'convex': convexity(state[i], area=area),
                'adj': []}


def setup_problem(adj, footprint, **kwargs):
    prob = Problem()
    print('area', footprint.area)
    # footprint_objective --------------------
    prob.footprint = footprint
    _adj_done = set()
    for k, vals in adj.items():

        prog = vals.get('prog', 'room')
        ent = ProgramEntity(k, prog)
        prob.add_program(ent)

        prob.add_constraint(FootPrintConstraint(k, footprint))

        # dimension_objective -----------------
        if 'area' in vals:
            prob.add_constraint(AreaConstraint(k, vals['area']))

        if 'aspect' in vals:
            prob.add_constraint(AspectConstraint(k, min=0.5, max=1.))

        # convexity_objectives ----------------
        if prog not in ['hall', 'stairs']:
            prob.add_constraint(ConvexityConstraint(k))

        # adjacency_objectives ----------------
        for adj_ent in vals.get('adj', []):
            kyed = tuple(sorted([adj_ent, k]))
            if kyed not in _adj_done:
                _adj_done.add(kyed)
                prob.add_constraint(AdjacencyConstraint(k, adj_ent, dim=3.))

    # print('created:\n', prob)
    return prob


def setup_house_rect(**kwargs):
    footprint = Area([(0, 0), (35, 0), (35, 30), (0, 30)], name='footprint')
    return setup_problem(_h1_adj, footprint, **kwargs)


def setup_house_L(**kwargs):
    footprint = Area([(0, 0), (40, 0), (40, 35), (30, 35), (30, 20), (0, 20)], name='footprint')
    return setup_problem(_h1_adj, footprint, **kwargs)


def setup_2room_rect(dimx=20, dimy=15, **kwargs):
    footprint = Area([(0, 0), (dimx, 0), (dimx, dimy), (0, dimy)], name='footprint')
    return setup_problem(_2room_adj, footprint, **kwargs)


def setup_modular1():
    """ The problem in just modules """
    problem = Problem()
    cnt = 0
    size = 4*2 + 3 * 3 + 3 * 1

    for i in range(4):
        ent = ProgramEntity('unit'.format(cnt), '2bed', problem=problem)
        problem.add_constraint(MaxDimConstraint(ent.name, mn=2, mx=2))
        problem.add_constraint(MinDimConstraint(ent.name, mn=1, mx=1))
        cnt += 1
    for i in range(3):
        ent = ProgramEntity('unit'.format(cnt), '1bed', problem=problem)
        problem.add_constraint(MaxDimConstraint(ent.name, mn=1, mx=1))
        problem.add_constraint(MinDimConstraint(ent.name, mn=1, mx=1))
        cnt += 1
    for i in range(3):
        ent = ProgramEntity('unit'.format(cnt), '3bed', problem=problem)
        problem.add_constraint(MaxDimConstraint(ent.name, mn=3, mx=3))
        problem.add_constraint(MinDimConstraint(ent.name, mn=1, mx=1))
        cnt += 1

    footprint = Area([(0, 0), (2, 0), (2, 10), (0, 10)], name='footprint')
    problem.add_constraint(FootPrintConstraint(footprint))


# --------------------------------------------------
def problem0(size):
    """ 2 spaces trivial """

    return [{'aspect': 0.5, 'area': 0.5, 'convex': 1, 'adj': [1 if i == 0 else 0]}
            for i in range(2)]


def problem1(num_spaces=3, x=20, y=20, return_state=False):
    """ 3spaces """
    s = np.zeros((num_spaces, x, y))
    bbox = [[0, 0, x-1, y-1]]
    seed = random.choice([0, 1])
    for i in range(num_spaces - 1):
        # kd-split
        # print(seed)
        split_l = random.uniform(.2, .8)
        if seed == 0:  # X
            # s[:, 0:int(split_l *x), :] = 0
            s[i, 0:int(split_l *x), :] = 1
        else:
            # s[:, :, 0:int(split_l * x)] = 0
            s[i, :, 0:int(split_l * x)] = 1

        if i == 0:
            s[1:] -= s[i]
        else:
            s[0:i] -= s[i]
            s[i+1:] -= s[i]
        s = np.clip(s, 0, 1)
        seed ^= 1

    s[num_spaces-1] = 1
    for i in range(num_spaces - 1):
        s[num_spaces - 1] -= s[i]

    state = np.clip(np.abs(s), 0, 1).astype(int)
    # print(state)
    bounds = max_boundnp(state)
    dilations = [skm.dilation(state[i]) for i in range(num_spaces)]
    items = {}
    for i in range(num_spaces):
        area = np.sum(state[i])
        adj = []
        for j in range(num_spaces):
            if i != j and len(np.where(state[i] + dilations[j] == 2)[0]) > 0:
                adj.append(j)
        items[i] = {'aspect': aspect(bounds[i][2:]),
                    'area': area / (x * y),
                    'convex': convexity(state[i], area=area),
                    'adj': adj}
    if return_state is True:
        return items, state
    return items


def problem12(num_spaces=3,  x=20, y=20, return_state=False):
    """ generate spaces by growing tiles .... """
    s = np.zeros((num_spaces, x, y))
    seeds = np.random.random_integers(0, max(x, y), 2 * num_spaces)

    return



