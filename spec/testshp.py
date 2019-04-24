from unittest import TestCase
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.affinity import translate
from shapely import ops
import pprint
import itertools

import pylab
import torch
import networkx as nx
from math import pi
import math
import numpy as np

import src.problem.example as prob
from src.problem import Problem, PointsInBound
import src.objectives as objectives

import src.algo.nns as nns
import src.algo.branchbound as bnb
import src.algo.grg as gr

from src.actions.basic import *
from src.building import Area, Room
from src.layout import BuildingLayout
import src.layout_ops as lops

import src.utils as utils
from src.utils import plotpoly

import matplotlib.pyplot as plt
import visdom
import torchnet as tnt

s = 10
random.seed(s)
np.random.seed(s)


class TestShpl(TestCase):
    def make1(self):
        p1 = box(0, 0, 5, 5)
        p2 = translate(box(0, 0, 3, 4), xoff=5)
        p3 = translate(box(0, 0, 4, 4), xoff=5, yoff=4)
        p4 = translate(p1, xoff=0, yoff=5)
        return MultiPolygon([p1, p2, p3, p4])

    def make2(self):
        p0 = translate(box(0, 0, 4, 4), xoff=3, yoff=3)
        p1 = box(0, 0, 7, 3)
        p2 = translate(box(0, 0, 3, 4), xoff=7)
        p3 = translate(box(0, 0, 4, 4), xoff=7, yoff=4)
        p4 = translate(box(0, 0, 4, 5), xoff=3, yoff=7)
        p5 = translate(box(0, 0, 3, 7), xoff=0, yoff=3)
        return MultiPolygon([p0, p1, p2, p3, p4, p5])

    def test_mpl1(self):
        mpl = self.make1()
        p1 = mpl.geoms[0]
        plotpoly(mpl)

    def test_mplp(self):
        mpl = self.make1()
        p1 = mpl.geoms[0]
        p2 = mpl.geoms[1]
        t1, t2 = ops.shared_paths(p1.exterior, p2.exterior)
        print(t1)
        print(t2)

    def to_layout(self, mpl=None):
        if mpl is None:
            mpl = self.make1()
        items = []
        for i, geom in enumerate(mpl):
            items.append(Room.from_geom(
                geom, name='r{}'.format(i), program='room')
            )
        return BuildingLayout(None, rooms=items)

    def _move1(self, param):
        layout = self.to_layout()
        new_layout = slide_wall(layout, **param)
        assert layout['r0']
        return new_layout

    def test_move2(self):
        layout = self.to_layout()
        room = layout['r0']
        new_ent = room.translate(xoff=2)
        fnl_ent = new_ent.union(room)
        print(fnl_ent)

    def test_move1(self):
        on_x = [True, False]
        grow = [True, False]
        dist = [2, -2]
        figs, tits = [], []
        mpl = self.make2()
        figs.append(self.to_layout(mpl))
        tits.append('base')
        for i, (x, d, g) in enumerate(itertools.product(on_x, dist, grow)):
            param = {'name': 'r0', 'on_x': x, 'grow': g, 'dist': d}
            layout = self.to_layout(mpl)
            new_layout = slide_wall(layout, **param)
            print('iter: ', i, param)
            figs.append(new_layout)
            tits.append(str(param))
        plotpoly(figs, tits)

    def test_move3(self):
        pass


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def sl_args(num_exper=5):
    split_prob = [[0.5, 0.5],
                  [0.7, 0.3],
                  [0.85, 0.15],
                  [0.7, 0.2, 0.1]]
    no_self_inters = [True, False]
    max_branch = [3, 5, 7, 10]

    angles = [[-pi/2, 0, pi/2],
              [-pi/2, -pi/4, 0, pi/4, pi/2]]

    prod = itertools.product(split_prob,
                             no_self_inters,
                             # dist_space,
                             max_branch)
    for spl, ints, mxb in prod:
        for i in range(num_exper):
            yield i, dict(split_prob=spl, max_branch=mxb, chk_inters=ints)


class MockEnv(object):
    def __init__(self, problem, init):
        self.initialize = init
        self.problem = problem
        self.loss = objectives.ConstraintsHeur(
            problem, wmap=dict(AreaConstraint=5, FootPrintConstraint=0)
        )
        self.__ub = self.loss.upper_bound
        self.action = MoveWall(allow_out=False)
        self.num_actions = 3

    def initial_state(self, batch_size=1):
        layouts = []
        for i in range(batch_size):
            layouts.append(self.initialize())
        return layouts

    def step(self, layout, **params):
        return self.action.forward(layout, **params)

    def reward(self, layout):
        # (-1, 1)
        if layout is None:
            return -1
        try:
            # if goal is not reached, its negative, else 1
            normed = self.loss.reward(layout)
            if (normed - 1.) > 1e-2:
                return normed
            return normed - 1
            # return self.loss.reward(layout)
            # return (3 -  self.loss(layout)) / 3
        except:
            return 0

    def reward2(self, layout, encode=False, **kwargs):
        """(-1, 1)"""
        if layout is None:
            if encode is True:
                return -.5, np.zeros(self.loss.size)
            return -.5
        return self.loss.reward(layout, encode=encode)

    def _transition(self, layout, action, room_index=None, item_index=None):
        size = self.num_actions // 2

        sign = 1 if action == size else -1
        mag = -1 * action if action < size else action - size

        room_name = layout[room_index].name
        params = dict(room=room_name, item=item_index, mag=mag, sign=sign)
        next_layout = self.action.forward(layout, **params)
        return next_layout

    def step_enc(self, layout, action, room_index=None, item_index=None):
        """ case 1 - output is right or left or Noop (0) HA !  [0, 1, 3]"""
        next_layout = self._transition(layout, action, room_index, item_index)
        reward, mat = self.reward2(next_layout, encode=True)
        fail = True if next_layout is None else False
        return dict(layout=next_layout, reward=reward, fail=fail, feats=mat)

    def step_norm(self, layout, action, room_index=None, item_index=None):
        """ case 1 - output is right or left or Noop (0) HA !  [0, 1, 3]"""
        next_layout = self._transition(layout, action, room_index, item_index)
        reward = self.reward2(next_layout)
        fail = True if next_layout is None else False
        return dict(layout=next_layout, reward=reward, fail=fail)


def make_env():
    problem = prob.setup_2room_rect()
    return MockEnv(problem, PointsInBound(problem, None))


class TestQ(TestCase):
    def setUp(self):
        self.Env = make_env()

    def test_inputs(self):
        # problem = prob.setup_2room_rect()
        Env = make_env()
        # [image_h, image_w, num_out]
        size = (20, 20, 2)
        trainer = gr.FloatingTreeExp1(Env, size=size)
        layout = trainer.env.initialize()
        print(Env.loss(layout))
        state = gr.layout_to_tensor(layout, size)
        print(state.size())

        # print(action)
        action = trainer.policy_net.conv1(state)
        print('in_feats', trainer.policy_net.head.in_features)
        action = trainer.policy_net(state).max(1)[1].view(1, 1)
        print(action)

    def test_encode(self):
        layout = self.Env.initialize()
        reward, mat = self.Env.loss.reward(layout, encode=True)
        print(reward)
        print(mat)
        pprint.pprint(self.Env.loss._constraint_to_idx)
        #$utils.plotpoly([layout], show=False)
        # viz = visdom.Visdom()
        # viz.matplot(plt)

    def test_ep(self):
        trainer = gr.FloatingTreeExp1(self.Env, size=(20, 20, 2))
        trainer.train(episodes=1000, steps=50)

    def test_epline(self):
        title = 'm=7,mem=8k,redos,lnorm[-1,1]'
        Env = make_env()
        trainer = gr.RL2(Env, size=(20, 20, 3), mode='line',
                         title=title, mem_size=8000, out_size=3)
        trainer.train(episodes=1000, steps=50)

    def test_reinforce(self):
        title = 'reinforce_stop_soft_flag1'
        Env = make_env()
        trainer = gr.Reinforced(Env,
                                title=title,
                                size=(20, 20, 3),
                                log_every=40,
                                num_actions=3)
        trainer.train(episodes=100000,
                      steps=1200,
                      fig_size=(20, 20))

    def test_random(self):
        Env = make_env()
        trainer = gr.RandomWalk(Env,
                                title='random200',
                                log_every=40,
                                num_actions=3)
        trainer.train(episodes=1000, steps=1200, fig_size=(10, 10))

    def test_plot(self):
        Env = make_env()
        layout = Env.initialize()
        utils.plotpoly([layout], show=False)
        viz = visdom.Visdom()
        viz.matplot(plt)

    def test_lyt(self):

        Env = make_env()
        layout = Env.initialize()
        trainer = gr.RL2(Env, size=(20, 20, 3))
        res = trainer.layout_to_tensor(layout, 0, 2)
        # print(res[0, -1])
        assert list(res.size()) == [1, 3, 20, 20]
        res = trainer.layout_to_tensor(layout, 0)
        assert list(res.size()) == [1, 3, 20, 20]
        # print(res[0, -1])

    def test_canny(self):
        Env = make_env()
        layout = Env.initialize()
        enc = gr.NNEncoder(Env.problem)
        state = enc.room_features(layout)
        canny = nns.Canny()
        inp = torch.stack((state[0], state[0], state[0])).unsqueeze(0)
        out = canny(inp)
        print(out)

    def test_reward(self):
        Env = make_env()
        layout = Env.initialize()
        r = Env.loss.reward(layout, explain=True)
        pprint.pprint(r)
        assert r[0] / r[1] < 1.0
        r1 = Room(box(0, 0, 10, 15), name='r1')
        r2 = Room(box(10, 0, 20, 15), name='r2')
        l2 = BuildingLayout(Env.problem, rooms=[r1, r2])
        print(Env.problem.footprint.area, r1.area, r2.area)
        ttl, avg, costs = Env.loss.reward(l2, explain=True)
        print(ttl, avg)
        print('reward', ttl / avg )
        pprint.pprint(costs)



class TestFT(TestCase):
    def test1(self):
        roots = gr.floating_trees(None)
        print(roots[0].get_boundary())
        utils.show_trees(roots)

    def test2(self):
        roots = gr.floating_trees(None)
        r1, r2 = roots
        # print(roots[0].get_boundary())
        print(len(r1))

        r1new = r1.update(2, np.asarray([2, 2]))
        print(len(r1new))
        utils.show_trees([r1new, r2])

    def test3(self):
        problem = prob.setup_2room_rect()
        layout = PointsInBound(problem, None)()
        cost_fn = objectives.ConstraintsHeur(problem, AreaConstraint=3)

        print(layout.is_valid, cost_fn(layout))
        print(layout.to_vec4())
        transition = MoveWall()
        model = gr.Annealer()
        state = model(layout, cost_fn, transition, num_iter=10000)
        return state
        # utils.plotpoly([layout, state])

    def test4(self):
        problem = prob.setup_2room_rect()
        layout = PointsInBound(problem, None)()
        room = layout.random()
        print(room)
        assert len(room[0]) == 4
        assert len(room[4]) == 2
        [print(x) for x in [room[1], room[2], room[3], room[4]]]
        move_action = MoveWall()
        new_layout = move_action(layout)

        # utils.plotpoly([layout, new_layout])

    def test5(self):
        # wierd polygon intersections
        l1 = [(1, 1), (1, 10), (10, 10), (10, 5) , (15, 5), (15, 1)]
        l2 = [(7, 5), (7, 10), (17, 10), (17, 5), (7, 5)]
        p1, p2 = Room(l1, name='r1'), Room(l2, name='r2')
        # print(Polygon(l1).intersection(Polygon(l2)))
        print(p2.intersection(p1))
        inters = p2.intersection(p1)
        p2 = p2.difference(inters)

        problem = prob.setup_2room_rect()
        lyt = BuildingLayout(problem, rooms=[p1, p2])
        utils.plotpoly([lyt])
        # return p1, p2


class TestL(TestCase):
    def test_grid_search(self):
        num_tests = 4

        points = [(0, 0), (20, 0), (20, 15), (0, 15)]
        i = 0
        e = -1
        seen = set()
        plot = dict()
        Gs = []
        for j, kwargs in sl_args(num_tests):
            print(i, j, kwargs)
            cfg = str(kwargs)
            if cfg not in seen:
                e += 1
                plot[e] = []

            name = '{} - {} - {} :: {}'.format(e, cfg, j, i)
            plot[e].append(name)
            Gs.append(self._run_config(points, name, **kwargs))
            i += 1

        num_expirements = e
        n_row = int(math.ceil(num_expirements ** 0.5))

        utils.plot_figs(Gs, num_tests)
        # from PIL import Image
        # for k, v in plot.items():
        #     for file in v:
        #         np.asarray(Image.open(file).convert('RGB'))
        return

    def plot_save(self, G):
        utils.simple_plot(G, kys=['ord'], save='./data/img/{}.png'.format(G.name))
        nx.write_gpickle(G, './data/pkl/{}.pickle'.format(G.name))

    def _run_config(self, points, name, clss=bnb.SimpleLH, save=False, **kwargs):
        footprint = Area(points, name='footprint')
        solve = clss(boundary=footprint, **kwargs)
        G = solve.run()
        G.name = name
        for i in range(len(points)):
            G.add_edge(points[i - 1], points[i])
        if save is True:
            self.plot_save(G)
        return G

    def test_1(self):
        random.seed(62)
        points = [(0, 0), (20, 0), (20, 15), (0, 15)]
        self._run_config(points, 'sl1', clss=bnb.SimpleL)

    def test_inters(self):
        random.seed(62)
        points = [(0, 0), (20, 0), (20, 15), (0, 15)]
        self._run_config(points, '2inters', chk_inters=True)

    def test_l2(self):
        random.seed(62)
        points = [(0, 0), (20, 0), (20, 15), (0, 15)]
        self._run_config(points, '2')



