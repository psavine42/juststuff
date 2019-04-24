import networkx as nx
from src.building import Area
from src.layout_ops import *
from .base import Problem, ProgramEntity
import math

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


def to_discrete():
    pass


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





