from math import pi, sin, cos
import numpy as np
import random
import src.utils
import itertools
import networkx as nx
import shapely.geometry as geom


class BranchBound:
    def __init__(self):
        pass


class Cell:
    def __init__(self, *point, dx, dy):
        self._point = point
        self._dimx = dx
        self._dimy = dy


class Lsystem:
    class LNode:
        pass

    def __init__(self):
        pass

    def step(self):
        pass


class Node:
    def __init__(self, data):
        self._data = data
        self._lh = None
        self._rh = None
        self.id = random.random()

    def __gt__(self, other):
        return self._data > other._data

    def __lt__(self, other):
        return self._data > other._data

    def insert(self, node):
        if self._lh is None and self._rh is None:
            self._lh = node
        elif self._lh is not None and node > self._lh:
            self._rh = node
        elif self._lh is not None and node < self._lh:
            self._rh = self._lh
            self._lh = node
        # elif self._lh is not None and self._rh is not None:
        #        if


def rot_mat(theta=0.):
    return np.asarray([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)],
    ])


def mat2d(theta=0.):
    return np.asarray([
        [cos(theta), -sin(theta), 0],
        [sin(theta), cos(theta),  0],
        [0.,        0,            1]
    ])


def simplel():
    xforms = [[[1, 0],
               [1, 0]]
            ]
    boundary = None
    split_prob = 0.3
    active_tgt = 5

    origin = Node(np.zeros((1, 2)))
    q = [origin]

    while q and active_tgt >= len(q):
        node = q.pop(0)
        # chose xform
        # xform = random.choice(xforms)
        theta = random.random(0, pi)
        x = random.random(1, 10)

        # apply xform
        new_node = Node(xform(node._data))
        # check constraints
        node._rh = new_node
        q.append(new_node)

        if random.random() < split_prob:
            xform = random.choice(xforms)
            n2 = Node(xform(node._data))
            node._lh = n2
            q.append(n2)

            # append new node to activeset
    return origin


class SimpleL:
    def __init__(self,
                 **kwargs
                 ):
        self.G = nx.Graph()
        self.boundary = kwargs.get('boundary', None)

        self._origin = kwargs.get('origin', (8, 0))
        self._chk_inters = kwargs.get('chk_inters', False)
        self._max_branch = kwargs.get('max_branch', 5)
        self._split_probs = kwargs.get('split_prob', [0.5, 0.5])
        self._lines = []

    def _gen_angle(self, node):
        # (0, 1) -> (-pi/2, pi/2)
        return pi * (random.random() - 0.5)

    def _gen_dist(self, node):
        return 10 * random.random()

    def add_node(self, new_cord, parent):
        new_node = tuple(new_cord[0].tolist()), new_cord[1]
        self.G.add_node(new_node[0], **new_node[1])
        self.G.add_edge(parent[0], new_node[0])

        new_line = geom.LineString([parent[0], new_node[0]])
        self._lines.append(new_line)

    def _test_inters(self, node, new_point):
        if self._chk_inters is False:
            return False
        new_line = geom.LineString([node[0], new_point.tolist()])
        for l in self._lines:
            # inter = new_line.intersection(l)
            #  print(inter)
            if new_line.crosses(l):
                return True
        self._lines.append(new_line)
        return False

    def check_valid(self, node, new):
        if not self.boundary.contains(geom.Point(new.tolist())):
            return False
        if self._test_inters(node, new) is True:
            return False
        return True

    def xform(self, node):
        while True:
            theta = self._gen_angle(node)
            x = self._gen_dist(node)

            new_vec = np.dot(rot_mat(theta), node[1]['dir'])
            new_pos = np.dot(np.asarray([[x, 0]]), new_vec)
            new = (new_pos + np.asarray(node[0]))[0]

            if self.check_valid(node, new) is False:
                continue

            print(new_vec, new)
            new_node = tuple(new.tolist()), {'dir': new_vec}
            self.G.add_node(new_node[0], **new_node[1])
            self.G.add_edge(node[0], new_node[0])
            # self._lines.append(new_line)
            return new_node

    def stats(self):
        pass

    def post_process(self):
        pos = nx.spring_layout(self.G)
        bnd = np.stack(pos.values())
        bnd[:, 0] += np.abs(bnd[:, 0].min())
        bnd[:, 1] += np.abs(bnd[:, 1].min())

        mxx, mxy = bnd[:, 0].max(), bnd[:, 1].max()
        xm, ym, xx, yx = self.boundary.bounds
        scale = max(mxx / (xx - xm), mxy / (yx - ym))
        print(bnd)
        mapping = {}
        for i, orig in enumerate(pos.keys()):
            mapping[orig] = tuple((bnd[i] / scale).tolist())
        print(mapping)
        G = nx.relabel_nodes(self.G, mapping, False)
        return G

    def run(self):
        prob_b = self._split_probs
        n_branch = np.linspace(1, len(prob_b), len(prob_b), dtype=int)

        origin = (self._origin,
                  {'dir': rot_mat(pi/2),
                   'ix': 0, 'ord': 0,
                   'uv': np.asarray([0., 1.])})
        self.G.add_node(origin[0], **origin[1])
        q = [origin]

        while q and self._max_branch >= len(q):
            node = q.pop(0)
            num_div = int(np.random.choice(n_branch, p=prob_b))

            for i in range(num_div):
                new_node = self.xform(node)
                if new_node is None:
                    continue
                q.append(new_node)

        return self.G


class SimpleLH(SimpleL):
    """
    todo:
        1). attractors
        2). no overlapping
        3). each node has a goal.
        4). room regularity by program
        5). Art gallery solver for privacy optimization
        5).
    """
    def __init__(self, **kwargs):
        SimpleL.__init__(self, **kwargs)
        self.angles = kwargs.get('angles', [-pi/2, 0, pi/2])
        self.dists = np.linspace(2, 10, 9).tolist()

    def check_valid(self, node, new):
        if not self.boundary.contains(geom.Point(new.tolist())):
            return False
        if self._test_inters(node, new) is True:
            return False
        return True

    def xform(self, node):
        # calculate attractors forces

        random.shuffle(self.angles)
        random.shuffle(self.dists)

        for theta, dist in itertools.product(self.angles, self.dists):

            new_dir = node[1]['uv'] @ rot_mat(theta)
            new_vec = new_dir * dist
            new = new_vec + np.asarray(node[0])

            if self.check_valid(node, new) is False:
                continue

            new_node = tuple(new.tolist()), {'uv': new_dir,
                                             'ix': len(self.G),
                                             'ord': node[1]['ord']+1}
            self.G.add_node(new_node[0], **new_node[1])
            self.G.add_edge(node[0], new_node[0])

            return new_node

    def post_process(self):
        return self.G


class RoomSearch:
    """ LH system modified to draw outlines """
    def __init__(self):
        pass



class SplitSearch:
    def __init__(self, boundary):
        self._boundary = boundary

    def run(self):
        pass



if __name__ == '__main__':
    solve = SimpleL()
    G = solve.run()

    # G = src.utils.cells_to_nx(root)
    src.utils.simple_plot(G)
