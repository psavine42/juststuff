from shapely import ops
from shapely import geometry
from shapely.geometry import Polygon, LineString, MultiPolygon
# from shapely.affinity import translate
from src.layout import BuildingLayout
from src.building import Room
import random
from copy import copy
import numpy as np

s = 10
random.seed(s)
np.random.seed(s)


class hashabledict(dict):
    def __key(self):
        return tuple((k, self[k]) for k in sorted(self))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()


class Split(object):
    @classmethod
    def by_line(cls, geom, line):
        pass

    @classmethod
    def by_line_id(cls, geom, edge_index):
        pass


class _RejectionSampledAction(object):
    def __init__(self, lim=1000):
        self._lim = lim
        self._cache = set()

    def forward(self, layout, **kwargs):
        return None

    def __call__(self, layout, **kwargs):
        iters = 0
        while iters < self._lim:
            new_layout = self.forward(layout, **kwargs)
            if new_layout is not None and new_layout.is_valid is True:
                self._cache = set()
                return new_layout
            kwargs = {}
            iters += 1
        print('COULD NOT FIND A VALID ACTION')
        return None


class SetToBounds(_RejectionSampledAction):
    """
    IF the boundary is cadywompous, it will be reset to an axis-aligned box
    """
    def forward(self, layout, **kwargs):
        new_layout = copy(layout)
        room = new_layout.random()
        mnx, mny, mxx, mxy = room.bounds
        new_room = Room([(mnx, mny), (mnx, mxy), (mxx, mxy), (mxx, mny)], **room.kwargs)
        # if new_room.is_valid:
        #     new_layout.update_room(new_room)
        return repair_overlaps(new_layout, new_room)


class MoveWall(_RejectionSampledAction):
    """

    """
    def __init__(self, magnitude=2, allow_out=True, **kwargs):
        _RejectionSampledAction.__init__(self, **kwargs)
        self._magnitude = magnitude
        self.allow_outside = allow_out

    def params(self, layout):
        room = layout.random()
        action_set = len(list(room.exterior.coords))

        # parameters for a given State
        item = np.random.randint(0, action_set)
        sign = np.random.choice([-1, 1])
        magn = np.random.randint(1, self._magnitude)
        return dict(room=room.name, item=item, mag=magn, sign=sign)

    def forward(self, layout, room=None, item=None, mag=None, sign=None, **kwargs):
        new_layout = copy(layout)
        room = new_layout[room] # new_layout.random() if room is None else

        to_update = room[item]
        if len(to_update) == 1:     # point
            new_room = room
            print('GOT POIINT')

        elif len(to_update) == 2:   # line - only perpendicular moves allowed
            (i1, p1), (i2, p2) = to_update[0], to_update[1]
            vec = np.asarray(p1) - np.asarray(p2)
            if np.linalg.norm(vec) == 0:
                return None
            ux, uy = vec / np.linalg.norm(vec)
            d = [uy, -ux]
            xform = sign * mag * np.asarray(d)
            new_room = room.update(item, xform)

        else:   # whole thing, translated no restrictions on direction
            xform = sign * mag
            new_room = room.update(item, xform)

        if new_room and new_room.is_valid:
            if self.allow_outside is False:
                fp = layout.problem.footprint
                if not fp.contains(new_room):
                    return None
            new_layout.update_room(new_room)
            return repair_overlaps(new_layout, new_room)


class MoveWallSticky(_RejectionSampledAction):
    def forward(self, layout, **kwargs):
        param = slide_wall_params(layout)
        return slide_wall(layout, **param)


class WeightedAction(_RejectionSampledAction):
    def __init__(self, actions, **kwargs):
        _RejectionSampledAction.__init__(self, **kwargs)
        self._actions = actions

    def forward(self, layout, **kwargs):
        pass


def repair_overlaps(new_layout, new_room):
    if new_room is None or new_room.is_valid is False:
        return None
    for other in new_layout.rooms():
        if other.name == new_room.name:
            continue
        if other.intersects(new_room) is True:
            intersect = other.intersection(new_room)
            if not isinstance(intersect, Room):
                continue
            new_other = other.difference(intersect)
            if not new_other or isinstance(new_other, (LineString, MultiPolygon)):
                return None
            new_layout.update_room(new_other)
    return new_layout


# Transition Operators -------------------------------------
def slide_wall_params(layout):
    room = layout.random()
    name = room.name
    mx, my, bx, by = room.bounds
    # print(room.bounds)
    bm = min([bx - mx
              - 1, by - my - 1, 6])
    dist = 0
    # print(bm)
    # while dist == 0:
    #    dist = random.randint(-bm, bm)
    dist = random.choice([-1, 1])
    grow = random.choice([0, 1])
    on_x = random.choice([False, True])
    return dict(grow=grow, name=name, on_x=on_x, dist=dist)


def slide_wall(layout: BuildingLayout, name=None, grow=None, on_x=None, dist=None):
    new_layout = copy(layout)
    if grow is None or dist is None or on_x is None:
        param = slide_wall_params(layout)
        grow = param['grow']
        dist = param['dist']
        on_x = param['on_x']
        name = param['name']

    room = new_layout[name]
    tkey = 'xoff' if on_x is True else 'yoff'
    kwrg = {tkey: dist}

    if grow:
        # translate and union
        new_room = room.translate(**kwrg)
        new_room = room.union(new_room)
        new_layout.update_room(new_room)
        new_layout = repair_overlaps(new_layout, new_room)
        # for other in new_layout.rooms():
        #     if other.name == new_room.name:
        #         continue
        #     if other.intersects(new_room) is True:
        #         intersect = other.intersection(new_room)
        #         if not isinstance(intersect, Room):
        #             continue
        #         new_other = other.difference(intersect)
        #         if not new_other or isinstance(new_other, (LineString, MultiPolygon)):
        #             return new_layout.empty_copy()
        #         new_layout.update_room(new_other)
    else:
        # shrink
        # translate and intersect
        translated = room.translate(**kwrg)
        new_room = room.intersection(translated)
        if not isinstance(new_room, Room):
            return new_layout.empty_copy()
        difference = room.difference(new_room)

        # only adjacent ones will be affected
        for other in new_layout.rooms():
            if other.name == new_room.name:
                continue
            if other.intersects(difference) is False:
                continue
            new_othr = other.translate(**kwrg)
            new_othr = other.union(new_othr)
            new_area = new_othr.intersection(difference)
            if not new_area:
                continue
            fnl_othr = other.union(new_area)
            if not fnl_othr or isinstance(fnl_othr, (LineString, MultiPolygon)):
                return new_layout.empty_copy()
            new_layout.update_room(fnl_othr)
        new_layout.update_room(new_room)
    return new_layout


def swap_rooms_params(layout):
    r1 = layout.random()
    r2 = layout.random()
    return dict(r1=r1.name, r2=r2.name)


def swap_rooms(layout, params=None):
    new_layout = copy(layout)
    if params is None:
        r1 = new_layout.random()
        r2 = new_layout.random()
    else:
        r1 = new_layout[params['r1']]
        r2 = new_layout[params['r1']]
    while r1 == r2:
        r2 = new_layout.random()
    p1, n1 = r1._program, r1._name
    p2, n2 = r2._program, r2._name
    r1._program, r1._name = p2, n2
    r2._program, r2._name = p1, n1
    return new_layout


def slide_wall2(room):
    ext = list(room.exterior.coords)


def split_geom_by_areas(geom, area1, area2):
    area = geom.area
    ratio = min(area1, area2) / (area1 + area2)
    box_dim = (area * (1 - 2 * ratio)) ** 0.5
    xmn, ymn, xmx, ymx = geom.bounds
    print(geom, area, area1, area2, ratio)
    if xmx - xmn > ymx - ymn:
        # vertical split box to the left
        dims = xmn-box_dim, ymn, xmn, box_dim
        splitc = geom.union(geometry.box(*dims)).centroid.x
        points = [splitc, -1e4], [splitc, 1e6]
    else:
        # horizantal split box to below
        dims = xmx-box_dim, ymn-box_dim, xmx, ymn
        splitc = geom.union(geometry.box(*dims)).centroid.y
        points = [-1e4, splitc], [1e6, splitc]

    new_geom = ops.split(geom, LineString(points))
    assert len(new_geom.geoms) == 2
    return new_geom.geoms



