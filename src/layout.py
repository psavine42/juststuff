from .interfaces import Layout
from copy import copy
from .building import Room, Area
from shapely import ops
from shapely.geometry import MultiLineString, Point
import random
import typing as T
from collections import defaultdict as ddict
from collections import OrderedDict as odict
import uuid
import numpy as np
from matplotlib.path import Path


class BuildingLayout(Layout):
    def __init__(self, env, rooms=None, adj=None, size=(256, 164)):
        self._problem = env
        self._cost = None
        self._rooms = odict()
        self._size = size
        self._uid = uuid.uuid4()
        self._img_dict = {}
        if isinstance(rooms, list):
            for r in rooms:
                self._rooms[r.name] = r
        self._adj = adj if adj else ddict(set)
        if len(self._adj) == 0 and len(self._rooms) > 0:
            self._compute_adj()

    def __iter__(self):
        for r in self._rooms.values():
            yield r

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._rooms[item]
        elif isinstance(item, int):
            # fuck yall IndexError nigga !
            return self._rooms[list(self._rooms)[item]]

    def __setitem__(self, key, value):
        self._rooms[key] = value

    def __len__(self):
        return len(self._rooms)

    def __copy__(self):
        roomlist = [copy(x) for x in self.rooms()]
        return self.__class__(
            self._problem, rooms=roomlist, adj=self._adj, size=self._size
        )

    def copy(self):
        return self.__copy__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._uid == other._uid
        return False

    def __repr__(self):
        st = ''
        for r in self.rooms():
            st += '\n{}'.format(str(r))
        return st

    def __hash__(self):
        rs = sorted(self.rooms(), key=lambda x: x.name)
        return hash(tuple([hash(r) for r in rs]))

    # Encode/decode --------------------------------------------
    def to_grid(self):
        """
        draw layout into a numpy array with indexes for
        https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask

        returns:
        -----
        np.array(self._size).ntype(int)
        """
        self._img_dict = {}
        nx, ny = self._size
        X = np.zeros((ny, nx))
        mul = 256 // (self.__len__() - 1)
        # xmn, ymn, xmx, ymx = self._problem.footprint.bounds
        scale = self.get_scale_for_size(nx, ny)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        for i, room in enumerate(self.geoms):
            # generate color and add to dictionary
            icol = (i + 1) * mul
            self._img_dict = {room.name: icol}
            # create a patch on meshgrid
            vert = (np.array(list(room.exterior.coords)) * scale).astype(int)
            grid = Path(vert).contains_points(points, radius=0)
            grid = grid.reshape((ny, nx))
            X[grid] += icol
        return np.flipud(X)

    def to_vec4(self):
        """
        returns rooms as a concatenated vector of room vectors
        each room vector corresponds to [x, y, l, w] of room
        """
        vec = []
        for r in self.rooms():
            vec.extend(r.uvbounds)
        return np.asarray(vec)

    def get_scale_for_size(self, nx, ny):
        xmn, ymn, xmx, ymx = self._problem.footprint.bounds
        return min([nx / (xmx - xmn), ny / (ymx - ymn)])

    def to_tensor(self, nx, ny, h):
        """ tensor of size [num_rooms, size omega.x, omega.y] """
        X = np.zeros((h, nx, ny))
        scale = self.get_scale_for_size(nx, ny)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        for i, room in enumerate(self.geoms):
            vert = (np.array(list(room.exterior.coords)) * scale).astype(int)
            grid = Path(vert).contains_points(points, radius=0)
            grid = grid.reshape((ny, nx))
            X[i][grid] = 1.

        return np.flipud(X).copy()

    def write_mat(self, geoms, nx, ny):
        """ tensor of size [num_rooms, size omega.x, omega.y] """
        scale = self.get_scale_for_size(nx, ny)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        X = np.zeros((len(geoms), nx, ny))
        for i, room in enumerate(geoms):
            vert = (np.array(list(room.exterior.coords)) * scale).astype(int)
            grid = Path(vert).contains_points(points, radius=0)
            grid = grid.reshape((ny, nx))
            X[i][grid] = 1.
        return np.flipud(X).copy()

    # ----------------------------------------------------------
    def empty_copy(self):
        return self.__class__(self._problem)

    def add_room(self, item: Area):
        self._rooms[item.name] = item

    def update_room(self, room):
        # print(room)
        self._rooms[room.name] = room
        # self.recompute_adj_for(room)

    def random(self):
        return random.choice(self.rooms())

    def rooms(self, names=None) -> T.List[Room]:
        if names is None:
            return list(self._rooms.values())
        return [self._rooms[k] for k in names if k in self._rooms]

    def boundaries(self):
        shp = None
        for r in self.rooms():
            if shp is None:
                shp = r
                continue
            shp = shp.union(r)
        return shp

    # --------------------------------------------------------
    def _adj_pair(self, r1, r2):
        try:
            na = ops.shared_paths(r1.exterior, r2.exterior)
            if na.is_empty or not na.is_valid:
                return
            t1, t2 = na
            if not t1.is_empty or not t2.is_empty:
                self._adj[r1.name].add(r2.name)
                self._adj[r2.name].add(r1.name)
                return True
        except:
            print('error:', r1, r2)

    def _compute_adj(self):
        rooms = self.rooms()
        for i in range(len(rooms)):
            for j in range(i, len(rooms)):
                self._adj_pair(rooms[i], rooms[j])

    def recompute_adj_for(self, room):
        name = room.name
        _adj = copy(self._adj[name])
        for v in _adj:
            self._adj[v].remove(name)
        self._adj[name] = set()
        rooms = self.rooms()
        for j in range(len(rooms)):
            if name != rooms[j].name:
                self._adj_pair(room, rooms[j])

    # --------------------------------------------------------
    @property
    def geoms(self):
        return self.rooms()

    def is_fully_within(self, room):
        bnd = self.problem.footprint
        for point in room.exterior.coords:
            if not bnd.contains(Point(point)):
                return False
        return True

    @property
    def is_valid(self):
        if len(self._rooms) != len(self._problem):
            return False
        return len(self._rooms) > 0 \
               and all(x.validate() is True for x in self.rooms()) \
               and all(self.is_fully_within(x) is True for x in self.rooms())

    @property
    def walls(self):
        for r in self.__iter__():
            yield r.exterior

    @property
    def adj(self):
        return self._adj

    @property
    def problem(self):
        return self._problem

    def show(self):
        import src.utils
        src.utils.plotpoly([self])




