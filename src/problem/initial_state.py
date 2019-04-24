from abc import ABC, abstractmethod
from src.problem import Problem
from src.building import Room
from src.layout import BuildingLayout
from shapely.geometry import Point, box
import src.layout_ops as lops
from src.actions.basic import split_geom_by_areas
import random


""" 
Initializers 

todo 
    - seed by grid
    - seed by graph-layout
    - 

"""
random.seed(1)


class _LayoutInit(ABC):
    def __init__(self, problem):
        self._problem = problem

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class PointsInBound(_LayoutInit):
    def __init__(self, problem: Problem, env, seed=None, size=2):
        _LayoutInit.__init__(self, problem)
        self._env = env
        self._size = size
        if seed:
            random.seed(seed)

    def __call__(self, *args, **kwargs):
        """
        for groth-based models
        """
        size = self._size
        fp = self._problem.footprint
        items = self._problem.program
        nitems = len(self._problem)
        minx, miny, maxx, maxy = fp.bounds
        seeds = []
        while len(seeds) < nitems:
            x = random.randint(minx, maxx-size)
            y = random.randint(miny, maxy-size)
            if fp.contains(Point(x, y)):
                program = items.pop(0)
                room = Room.from_geom(box(x, y, x + size, y + size), **program.kwargs)
                ok = True
                for s in seeds:
                    if room.intersection(s) and not room.intersection(s).is_empty:
                        ok = False
                        break
                if ok is True:
                    seeds.append(room)
                else:
                    items.append(program)
        return BuildingLayout(self._problem, rooms=seeds, **kwargs)


class InitializeFullPriority(_LayoutInit):
    """
    initialization algorithm from
    Constraint-aware Interior Layout Exploration for Precast Concrete-based Buildings
    Han Liu · Yong-Liang Yang · Sawsan AlHalawani · Niloy J. Mitra

    """
    def __init__(self, problem: Problem, env, allow_overlap=False, size=2):
        _LayoutInit.__init__(self, problem)
        self._env = env
        self._init_size = size

    def __call__(self, *args, **kwargs):
        """
        layout given a

        :return:
        """
        seeds = []
        probl = self._problem
        names = [x.name for x in probl.program]
        const = {x.ent: x._min for x in probl.constraints(lops.AreaConstraint)}
        const = {**{name: 100 for name in names}, **const}
        names.sort(key=lambda x: len(probl.G[x]))
        queue = [(probl.footprint, names)]

        while queue:
            geom, members = queue.pop(0)

            # take top two largest rooms
            ent1 = members.pop(0)
            ent2 = members.pop(0)
            sub_region1, sub_region2 = [ent1], [ent2]

            for ent in members:
                in1 = ent1 in probl.G[ent]
                in2 = ent2 in probl.G[ent]
                assign = None
                if in1 is True and in2 is True:
                    # connected to both
                    # exists a room r i that is connected to both r̄ 1 and r̄ 2 ,
                    # we check the number of joint connected rooms between
                    # r i against r̄ 1 and r̄ 2 ; and assign r i to the group with
                    # the larger number of overlapping connections
                    ent1_wght = len(probl.G[ent].intersection(probl.G[ent1]))
                    ent2_wght = len(probl.G[ent].intersection(probl.G[ent2]))
                    if ent1_wght == ent2_wght:
                        assign = len(probl.G[ent1]) < len(probl.G[ent2])
                    else:
                        assign = int(ent1_wght > ent2_wght)
                elif in1 is True:
                    assign = 1
                elif in2 is True:
                    assign = 0
                else:    # not connected to either
                    r1_area = sum(map(lambda x: const[x], sub_region1))
                    r2_area = sum(map(lambda x: const[x], sub_region2))
                    assign = r1_area < r2_area

                if assign == 1:
                    sub_region1.append(ent)
                else:
                    sub_region2.append(ent)

            r1_area = sum(map(lambda x: const[x], sub_region1))
            r2_area = sum(map(lambda x: const[x], sub_region2))

            # split geom based on areas
            r1_geom, r2_geom = split_geom_by_areas(geom, r1_area, r2_area)

            if len(sub_region1) == 1:
                seeds.append([r1_geom, sub_region1[0]])
            else:
                queue.append([r1_geom, sub_region1])

            if len(sub_region2) == 1:
                seeds.append([r2_geom, sub_region2[0]])
            else:
                queue.append([r2_geom, sub_region2])

        rooms = [Room.from_geom(geom, **probl[name].kwargs) for geom, name in seeds]
        return BuildingLayout(self._problem, rooms=rooms, **kwargs)


