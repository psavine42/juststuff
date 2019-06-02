from .interfaces import Layout
from copy import copy
from .building import Room, Area
from shapely import ops
from shapely.geometry import MultiLineString, Point, LineString
import random
import typing as T
from collections import defaultdict as ddict
from collections import OrderedDict as odict
import uuid
import numpy as np
from matplotlib.path import Path
import networkx as nx
import src.geom.cg as cg
import skimage.morphology as skm


def fillable(G):
    """ not used """
    res = np.zeros(G.shape, dtype='bool')
    # if i > 0 and G[i - 1, j] == 2:
    # find indices where this is true
    imask, jmask = np.where(G == 2)
    imask, jmask = imask.copy(), jmask.copy()  # make them writable
    imask -= 1  # shift i indices
    imask, jmask = imask[imask >= 0], jmask[jmask >= 0]
    res[imask, jmask] = True
    # [..] do the other conditions in a similar way
    return res


def get_scale_for_size(footprint, nx, ny):
    if footprint is not None:
        xmn, ymn, xmx, ymx = footprint.bounds
        return min([nx / (xmx - xmn), ny / (ymx - ymn)])
    else:
        return nx, ny


def write_mat(geoms, footprint, nx, ny):
    """ tensor of size [num_rooms, size omega.x, omega.y] """
    scale = get_scale_for_size(footprint, nx, ny)
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


class _LayoutBase(Layout):

    def __init__(self, problem):
        self._problem = problem

        self._rooms = odict()
        self._state = None

        self._uid = uuid.uuid4()
        self._index_to_name = {}
        self._name_to_index = {}

    def __repr__(self):
        st = ''
        for r in self.rooms():
            st += '\n{}'.format(str(r))
        return st

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._uid == other._uid
        return False

    def __init_state(self):
        pass

    @property
    def problem(self):
        return self._problem

    def rooms(self, names=None):
        if names is None:
            return list(self._rooms.values())
        return [self._rooms[k] for k in names if k in self._rooms]

    @property
    def is_valid(self):
        return True

    @property
    def state(self):
        return self._state


class BuildingLayout(_LayoutBase):
    def __init__(self, env, rooms=None, adj=None, size=(256, 164)):
        _LayoutBase.__init__(self, env)
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

    def show(self):
        import src.utils
        src.utils.plotpoly([self])


class DiscreteLayout(_LayoutBase):
    """
    ***NOTE*** This one is mutable for efficiency
    dim0 = cursor
    dim1 = lines
    dim2 = footprint (optional)
    """
    def __init__(self, problem, rooms=None, depth=2, adj=None, size=(30, 30), **kwargs):
        _LayoutBase.__init__(self, problem)
        self._problem = problem
        self._rooms = odict()
        self._index_to_name = {}
        self._size = size
        self.N, self.M = size[0], size[1]
        if isinstance(rooms, list):
            for i, r in enumerate(rooms):
                self._rooms[r.name] = r
                self._index_to_name[i] = r.name

        self._adj = adj if adj else ddict(set)
        self._state = None
        self._depth = depth

    def __init_state(self):
        self._state = np.zeros((self._depth, self._size[0], self._size[1]))

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._rooms[item]
        elif isinstance(item, int):
            return self._rooms[self._index_to_name[item]]
        else:
            raise Exception('did not get an int or str')

    def add_step(self, step, draw=1, erase=0):
        pass

    def add_line(self, start, end, **kwargs):
        points = cg.discretize_segment(start, end)
        for pnt in points:
            res = self.add_step(pnt, **kwargs)
            if res is False:
                return False
        return True

    def size(self, idx=None):
        if idx:
            return self._size[idx]
        return self._size

    @property
    def footprint(self):
        return None

    def to_image(self, arg):
        return self._state[-1]


class CellComplexLayout(_LayoutBase):
    """
    ***NOTE*** This one is mutable for efficiency
    dim0 = cursor
    dim1 = lines
    dim2 = footprint (optional)
    """
    def __init__(self, problem, rooms=None, depth=2, adj=None, size=(30, 30), origin=(0, 0)):
        _LayoutBase.__init__(self, problem)
        self._problem = problem
        self._cost = None
        self._rooms = odict()
        self._index_to_name = {}
        self._size = size
        self.N, self.M = size[0], size[1]
        if isinstance(rooms, list):
            for i, r in enumerate(rooms):
                self._rooms[r.name] = r
                self._index_to_name[i] = r.name

        self._adj = adj if adj else ddict(set)
        self._state = None
        self._discrete_footprint = None
        self._origin = origin
        self._prev = origin
        self._G = None
        self._T = None
        self.__init_graph()
        self.__init_state()
        self._ncomp = 1
        self._depth = depth
        # cycles are rooms
        # a room is a set of trajectory points
        # intersectiong between them gives adjaceny
        self._areas = []

    def __init_state(self):
        self._state = np.zeros((self._depth, self._size[0], self._size[1]))
        # self._state[0, self._prev[0], self._prev[1]] = 1
        fp = write_mat([self._problem.footprint], self._problem.footprint, self.N, self.M)
        self._discrete_footprint = fp

    def _add_discrete_node(self, i, j, if_exists=False):
        if i < self._size[0] - 1 and ((i + 1, j) in self._G or if_exists is False):
            self._G.add_edge((i, j), (i + 1, j))
        if j < self._size[1] - 1 and ((i, j + 1) in self._G or if_exists is False):
            self._G.add_edge((i, j), (i, j + 1))
        if i > 0 and ((i - 1, j) in self._G or if_exists is False):
            self._G.add_edge((i, j), (i - 1, j))
        if j > 0 and ((i, j - 1) in self._G or if_exists is False):
            self._G.add_edge((i, j), (i, j - 1))

    def __init_graph(self):
        self._G = nx.Graph()
        self._T = nx.Graph()
        n, m = self._size
        for i in range(n):
            for j in range(m):
                self._add_discrete_node(i, j)
        self._ncomp = nx.number_connected_components(self._G)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._rooms[item]
        elif isinstance(item, int):
            return self._rooms[self._index_to_name[item]]
        else:
            raise Exception('did not get an int or str')

    def erase(self, step):
        i, j = step
        self._T.remove_node((i, j))
        self._add_discrete_node(i, j, if_exists=True)

    def add_step(self, step, draw=1, erase=0):
        step = tuple(step)
        assert len(step) == 2, 'invalid step ' + str(step)

        # if there is a footprint, and the step is outside of the footprint
        if self._discrete_footprint and self._discrete_footprint[step] != 1:
            return False

        if step in self._G and draw == 1:
            # update G - check is it loop or parallel
            self._G.remove_node(step)
            new_conn_comps = nx.number_connected_components(self._G)
            if new_conn_comps != self._ncomp:
                self._areas = nx.connected_components(self._G)
                self._ncomp = new_conn_comps

            # update T - state and path
            self._state[1, step[0], step[1]] = 1
            self._T.add_node(step)
            if self._prev in self._T:
                self._T.add_edge(self._prev, step)

        # update agent location
        px, py = self._prev
        self._state[0, px, py] = 0
        self._state[0, step[0], step[1]] = 1
        self._prev = step
        return True

    def add_line(self, start, end, **kwargs):
        points = cg.discretize_segment(start, end)
        for pnt in points:
            res = self.add_step(pnt, **kwargs)
            if res is False:
                return False
        return True

    def size(self, idx=None):
        if idx:
            return self._size[idx]
        return self._size

    @property
    def footprint(self):
        return self._discrete_footprint

    def to_image(self):
        return self._state[-1]

    @property
    def num_rooms(self):
        return self._ncomp

    @property
    def cursor(self):
        return self._prev


class BWRooms(DiscreteLayout):
    """
    Each Room gets a COLOR in Dimension 0
    # todo expirement if should be negative 1
    dim-1 -> footprint - 1 if not allowed else 0
    """
    def __init__(self, problem, dim_labels={}, **kwargs):
        DiscreteLayout.__init__(self, problem, **kwargs)
        self._footprint_dim = -1
        self._colors = np.linspace(0.3, 1, len(problem))

        self.__init_state()

    def rooms(self, names=None):
        if names is not None:
            return []
        room_mats = []
        for i in range(len(self._problem)):
            room_mats.append(np.where(self._state == self._colors[i]))
        return room_mats

    def add_step(self, step, draw=1, erase=0):
        i, xmin, ymin, xmax, ymax = step
        self._state[0, xmin:xmax, ymin:ymax] = self._colors[i] * draw
        self._state[0] -= self._state[self._footprint_dim]
        # todo this is a hard clip should ?
        self._state = np.clip(self._state, 0, 1)
        return True

    @property
    def footprint(self):
        return self._state[self._footprint_dim]


def concat_lengthwise(state, num_spaces):
    """ State with first num_spaces as bitmasks
     [S, N, M] -> [ S*N, M]
     """
    return np.expand_dims(np.concatenate(state, 0), axis=0)


def hatch(state, num_spaces):
    """ [S, N, M] -> [ N, M] with hatches """
    hatched = cg.generate_hatches(state.shape, num_spaces)
    return np.sum(np.clip(state[:num_spaces] + hatched - 1, 0, 1), 0)


# ----------------------------------------------------------------
def _preproc_center_hw(box):
    """
    input [ xcent, ycent, size_y, size_y]
    center = (0, 0)
    return [ xmin, ymin, xmax, ymax]
    """
    clip = (1 + np.clip(box[0:2], -1, 1)) / 2
    hw = np.abs(box[2:])
    return np.concatenate([clip - hw, clip + hw])


def _preproc_norm_box(box):
    """
    input [ xmin, ymin, xmax, ymax ] e (-2, 2)
    return [ xmin, ymin, xmax, ymax] center = (0.5, 0.5), d=(0, 1)
    """
    clip = box[0:2] / 2

    return np.concatenate([clip - hw, clip + hw])


def _preproc_iden(box):
    return box


# Rooms as image stacks ----------------------------------------------------------------
class StackedRooms(DiscreteLayout):
    """
    This SHOULD just be a way of indexing the layers ...
    Each Room gets a dimsion
    # todo expirement if should be negative 1
    dim-1 -> footprint : 1 if not allowed, 0 if available
    """

    def __init__(self, problem, depth=None, box_fmt='', problem_dict={}, **kwargs):
        DiscreteLayout.__init__(self, problem, **kwargs)
        self._footprint_dim = -1
        self._depth = depth if depth else 1 + len(self.problem)
        self._prev = None
        self._num_spaces = len(self._problem)
        self._problem_dict = problem_dict
        self.__init_state()

        #
        self._coord = np.asarray([self.N, self.M, self.N, self.M])
        self._edges = np.zeros_like(self.active_state)
        self._create_box_min_max = lambda x: x
        if box_fmt == 'chw':
            self._create_box_min_max = _preproc_center_hw

    def to_input(self):
        return np.expand_dims(np.concatenate(self._state, 0), axis=0)

    def to_image(self, input_state):
        return input_state

    @property
    def input_state_size(self):
        return list(self._state.shape)

    @property
    def active_state(self):
        return self._state[:self._num_spaces]

    @active_state.setter
    def active_state(self, state):
        assert state.shape == self.active_state.shape, \
            'state in {}, active {}'.format(state.shape, self.active_state.shape)
        self._state[:self._num_spaces] = state

    @property
    def footprint(self):
        return self._state[self._footprint_dim]

    def rooms(self, names=None, stack=True):
        """ """
        if names is not None:
            return []
        if stack is True:
            return self.active_state
        return np.stack([self._state[i] for i in range(self._num_spaces)])

    def __init_state(self):
        self._state = np.zeros((self._depth, self._size[0], self._size[1])).astype(int)
        if self._problem.footprint is not None:
            fp = write_mat([self._problem.footprint], self._problem.footprint, self.N, self.M)
        else:
            fp = np.ones((self._size[0], self._size[1]))
        self._state[self._footprint_dim] = fp

    def add_step(self, box_args, draw=1, erase=0):
        i, pre = box_args
        c = self._create_box_min_max(pre)

        legal = True
        if np.where(c < 0.0) or np.where(c > 1.):
            legal = False
        c = np.multiply(self._coord, np.clip(c, 0, 1)).astype(int)
        # print(i, c)
        if c[1] == c[3] or c[0] == c[2]:
            return False, False

        self._state[i, c[0]:c[2], c[1]:c[3]] = draw
        self._state[i] *= self.footprint

        if i == 0:
            self._state[1:self._num_spaces] -= self._state[i]
        elif i == self._num_spaces:
            self._state[:self._num_spaces-1] -= self._state[i]
        else:
            self._state[0:i] -= self._state[i]
            self._state[i+1:len(self._problem)] -= self._state[i]
        # todo this is a hard clip should anything be negitve
        self._state = np.clip(self._state, 0, 1)
        self._prev = c
        return True, legal


class ProbStack(StackedRooms):
    def __init__(self, *args, init_dist=None, eps=0.1, **kwargs):
        """

        init_dist: options 'None' - default is uniform

        Properties:
            N:  x-dimension of state
            M:  y-dimension of state
            depth:  channels of state

            footprint:  NxM Tensor
            active_state:
            state:

        """
        StackedRooms.__init__(self, *args, **kwargs)
        self._init_dist = init_dist
        self._eps = eps
        self.__init_state()

    def __init_state(self):
        self._state = np.zeros((self._depth, self.N, self.M)).astype(float)
        # todo add this back in latter
        # fp = write_mat([self._problem.footprint], self._problem.footprint, self.N, self.M)
        self._state[self._footprint_dim] = 1.

        state = self.active_state
        # set the likelyhood of that pixel being space_i
        for k, v in self._problem_dict.items():
            state[k, :, :] = v['area']

        # add some noise
        if self._init_dist is None:
            noise = np.random.uniform(size=(self._num_spaces, self.N, self.M)) * self._eps
            state = softmax(state + (noise - np.max(noise) / 2), axis=0)

        elif self._init_dist == 'scaled':
            state *= self._eps

        elif self._init_dist == 'scaled-un':
            noise = np.random.uniform(size=(self._num_spaces, self.N, self.M)) * self._eps
            state = np.multiply(state, noise)

        elif self._init_dist == 'zeros':
            state = np.zeros_like(state)

        self._state[:self._num_spaces] = state

    def rooms(self, names=None, stack=True, state=None):
        """ return a binary plane for each space """
        if state is None:
            state = self._state[:self._num_spaces]
        res = np.zeros_like(self.active_state)
        maxs = np.argmax(state, axis=0)
        for i in range(self._num_spaces):
            ix = np.where(maxs == i)
            res[i, ix[0], ix[1]] = 1
        return res

    def to_input(self):
        """ """
        return self._state.copy()

    def add_step(self, inputs, draw=1, erase=0):
        """ the agent directly modifies state - so this is a no-op"""
        self._state[:self._num_spaces] = inputs.copy()
        return True, True

    @property
    def input_state_size(self):
        return list(self._state.shape)

    def to_image(self, input_state=None):
        return self.rooms(stack=True, state=input_state)



class KDRooms(StackedRooms):
    def add_step(self, box_args, draw=1, erase=0):
        """
        ACTIONS :
            MOVE+UP     1
            DOWN+LEFT   1   -
            DOWN+RIGHT  1   -
            SPLIT       1   -

        """


class RoomStack(StackedRooms):
    def to_input(self):
        return self._state

    @property
    def input_state_size(self):
        return list(self._state.shape)

    def to_image(self, input_state):
        used = input_state[:self._num_spaces]
        spaces = np.einsum('jik,j->ik', used, np.linspace(0.25, 0.9, self._num_spaces))
        return np.clip(np.stack([spaces, self.footprint * 0.1, np.zeros(self._size)]), 0, 1)


class HatchedRooms(StackedRooms):
    """ """
    def to_input(self):
        used = self.active_state
        hatches = cg.generate_hatches(used.shape, self._num_spaces)
        spaces = np.sum(np.multiply(hatches, used), 0)
        # todo this is FUCKED ND erosions are shyte
        lines = np.sum(used - skm.erosion(used), 0)
        return np.clip(np.stack([spaces, lines, self.footprint]), 0, 1)

    @property
    def input_state_size(self):
        return [3] + list(self._size)

    def to_image(self, input_state):
        return input_state


class ColoredRooms(StackedRooms):
    def to_input(self):
        used = self._state[:self._num_spaces]
        spaces = np.einsum('jik,j->ik', used, np.linspace(0.4, 0.9, self._num_spaces))
        return np.clip(np.stack([spaces, self.footprint]), 0, 1)

    @property
    def input_state_size(self):
        return [2] + list(self._size)

    def to_image(self, input_state):
        return np.concatenate([input_state, np.zeros((1, *self._size))])


class ColoredRoomsCenter(StackedRooms):
    def __init__(self, *args, **kwargs):
        kwargs['box_fmt'] = 'chw'
        StackedRooms.__init__(self, *args, **kwargs)

    def to_input(self):
        spaces = np.einsum('jik,j->ik', self.active_state, np.linspace(0.25, 0.9, self._num_spaces))
        return np.clip(np.stack([spaces, self.footprint]), 0, 1)

    @property
    def input_state_size(self):
        return [2] + list(self._size)

    def to_image(self, input_state):
        return np.concatenate([input_state, np.zeros((1, *self._size))])



def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    if len(X.shape) == 1:
        p = p.flatten()

    return p

