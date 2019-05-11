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
import torch.nn as nn

import src.problem.example as prob
from src.problem import Problem, PointsInBound
import src.objectives as objectives
from src.model.arguments import Arguments

import src.algo.nns as nns
import src.algo.branchbound as bnb
import src.algo.grg as gr

from src.actions.basic import *
from src.envs import *
from src.building import Area, Room
from src.layout import *
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




def make_env():
    problem = prob.setup_2room_rect()
    return MockEnv(problem, PointsInBound(problem, None))


def test_run(self):
    # how input is encoded
    #
    enc = gr.NNEncoder
    input_confs = [enc.encode_constraints, enc.edges]

    # secondary inputs with goal state
    input_hints = [True, False]

    # neural net mdoel
    # todo Refinenet
    # Model outputs
    # 1) action logits
    # 2) value
    # 3) hidden states
    # 4) index of element
    # 5) location of element (x,y) - then need to interpolate to element
    # 6) point (x, y) to place something
    # 7) magnitude (eg how far to move something
    #
    nn_models = [nns.DQN, nns.DQNC, nns.ActorCritic, nns.ContinuousActor, ]

    # 1) discrete {up,down,left,right}
    # 2) continuous { x, y }
    # 3) discrete { pos , neg } - to move in positive or negative direction
    action_schemes = [DrawLineUDLR, MoveWall, MoveWallUDLR]

    # whether or not there is a 'do nothing' action
    # and whether a magnitidue is specified
    use_distance = [True, False]
    allow_pass = [True, False]

    # 1) {-1, 1}, {-1, 0, 1},
    # 2) numgoals, numgoals-with-reward/penalty
    # 3) reward at end of episode
    # 4) (-1, 1)(continuous)
    reward_schemes = []

    # penalty for an illegal action
    # if None, then interpolate to nearest legal
    illegal_action = [-1, None]

    # trainer class
    # REINFORCE, DQN,
    # todo AC3, PPO, DNC
    trainers = [gr.Reinforced, gr.DQNTrainer,  # tested
                gr.DrawLSTM
                ]

    opts = dict(trainers=trainers,
                illegal_action=illegal_action,
                reward_schemes=reward_schemes,
                allow_pass=allow_pass,
                use_distance=use_distance,
                action_schemes=action_schemes,
                nn_models=nn_models,
                input_schemes=input_confs,
                input_hints=input_hints,
                )

    return opts


def base_args(title, viz=None):
    import datetime
    args = Arguments()
    ds = datetime.datetime.now().strftime(" %B-%d-%Y-%I:%M%p")
    args.title = title + ds
    args.log_every = 50
    args.episodes = 100000
    args.steps = 10
    args.viz = viz

    args.env = Arguments()
    args.env.random_init = True
    args.env.random_objective = False
    args.env.incomplete_reward = -1

    # args.env = env_args

    args.loss = Arguments()
    args.loss.gamma = 0.99
    args.loss.entropy_coef = 0.01
    args.loss.gae_lambda = 0.95
    args.loss.value_loss_coef = 0.5
    args.loss.max_grad_norm = 0.5

    # nn_args = Arguments(out_dim=256)
    args.nn = Arguments()
    args.nn.out_dim = 256
    return args


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
        trainer = gr.DQNTrainer(self.Env, size=(20, 20, 2))
        trainer.train(episodes=1000, steps=50)

    def test_epline(self):
        title = 'm=7,mem=8k,redos,lnorm[-1,1]'
        Env = make_env()
        trainer = gr.RL2(Env, size=(20, 20, 3), mode='line',
                         title=title, mem_size=8000, out_size=3)
        trainer.train(episodes=1000, steps=50)

    def test_reinforce(self):
        title = 'reinfrc_stop_flag_noactpen_hard100step'
        Env = make_env()
        trainer = gr.Reinforced(Env,
                                title=title,
                                size=(20, 20, 3),
                                log_every=40,
                                num_actions=3)
        trainer.train(episodes=100000,
                      steps=100,
                      fig_size=(20, 20))

    def test_random(self):
        trainer = gr.RandomWalk(self.Env,
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


class MiscGeom(TestCase):
    def test1(self):
        problem = prob.setup_2room_rect()
        goal = [10, 10]
        size = math.ceil(sum(goal) ** 0.5)
        lyt = DiscreteCellComplexLayout(problem, size=(size, size))
        obj = DiscreteSpaceObjective(goal)

        # utils.layout_disc_to_viz(lyt)

        env_action = UDLRStep()
        lyt, success = env_action.forward(lyt, action=0)
        lyt, success = env_action.forward(lyt, action=2)
        print(obj.reward(lyt, True))
        lyt, success = env_action.forward(lyt, action=0)
        lyt, success = env_action.forward(lyt, action=0)
        lyt, success = env_action.forward(lyt, action=0)
        # img = lyt.to_image()
        # viz.image(img)

        utils.layout_disc_to_viz(lyt)
        r, m = obj.reward(lyt, True)
        print(r, m)

    def test2(self):
        problem = prob.setup_2room_rect()
        goal = [10, 10]
        size = math.ceil(sum(goal) ** 0.5)
        lyt = CellComplexLayout(problem, size=(size, size))
        obj = DiscreteSpaceObjective(goal)

        env_action = UDLRStep(False)
        # for i in range(2):
        lyt, success = env_action.forward(lyt, action=2)
        lyt, success = env_action.forward(lyt, action=6)
        print(obj.reward(lyt, True))
        for i in range(4):
            lyt, success = env_action.forward(lyt, action=4)

        # img = lyt.to_image()
        # viz.image(img)

        utils.layout_disc_to_viz(lyt)
        r, m = obj.reward(lyt, True)
        print(r, m)

    def test3(self):
        problem = prob.setup_2room_rect()
        goal = [10, 10]
        size = math.ceil(sum(goal) ** 0.5)
        lyt = CellComplexLayout(problem, size=(size, size))
        obj = DiscreteSpaceObjective(goal)

        env_action = UDLRStep(False)
        for i in range(3):
            lyt, success = env_action.forward(lyt, action=6)
        lyt, success = env_action.forward(lyt, action=4)
        for i in range(3):
            lyt, success = env_action.forward(lyt, action=7)
        print(obj.reward(lyt, True))
        #for i in range(4):
        #     lyt, success = env_action.forward(lyt, action=4)

        utils.layout_disc_to_viz(lyt)

    def test_run1(self):
        problem = prob.setup_2room_rect()
        goal = [10, 10]
        size = math.ceil(sum(goal) ** 0.5)
        act = UDLRStep(False)
        obj = DiscreteSpaceObjective(goal)
        env = DiscreteEnv(problem, True, obj, size, act)

        policy = nns.LSTMDQN(size, size, act.num_actions,
                             in_size=2, feats_in=2, feats_size=10)
        target = nns.LSTMDQN(size, size, act.num_actions,
                             in_size=2, feats_in=2, feats_size=10)

    def test_acc_seq(self):
        res = [[2, [-1.41, -0.86, 1.38, 1.16]],
               [1, [0.0, 0.0,  0.5, 0.5]],
               [2, [-1.06, -0.8,  1.89, 0.92]]]

        env = DiscreteEnv2(None, None, [10, 10],
                           depth=6,
                           state_cls=ColoredRooms,
                           num_spaces=3,
                           random_init=True)
        print(env.objective.to_input())
        for i,x in res:
            print('-----')
            print(np.where(np.asarray(x) < 0.0 ))
            res = env.step([i, np.asarray(x)])
            # print(res['prev_state'])
            print(i, x)
            print(res['legal'])

    def test_disclyt(self):
        problem = prob.setup_2room_rect()
        state = StackedRooms(problem, size=(10, 10))

        objective = DiscProbDim(None)
        print(objective.keys)
        # action = DrawBox()
        state.add_step([0, 0, 0, 4, 4])
        r1, m1 = objective.reward(state)
        print(r1)
        print(m1)
        state.add_step([1, 5, 5, 8, 8])

        # print(np.stack(state.rooms()))
        r2, m2 = objective.reward(state)
        print(r2)
        print(m2)
        print(state.state)

    def test_ac2(self):
        # problem = prob.setup_2room_rect()
        title = 'ac-cont-mix'

        args = Arguments()

        # args.title = title
        # args.log_every = 20
        #
        # env_args = Arguments()
        # env_args.random_init = True
        # env_args.random_objective = False
        # env_args.incomplete_reward = -1
        # env_args.lr = 0.0005
        # args.env = env_args
        #
        # loss_args = Arguments()
        # loss_args.gamma = 0.99
        # loss_args.gae_lambda = 0.95
        # loss_args.entropy_coef = 0.01
        # loss_args.value_loss_coef = 0.5
        # loss_args.max_grad_norm = 0.5
        #
        # nn_args = Arguments(out_dim=256)
        # nn_args.out_dim = 256
        num_spaces = 3
        depth = 6
        action_dim = 4
        S = [num_spaces, 20, 20]

        env = DiscreteEnv2(None, None, S[1:],
                           depth=depth,
                           num_spaces=num_spaces,
                           random_init=True)

        common = nns.CnvController2h([1, S[1] * depth, S[2]],
                                     hints_shape=env.objective.area,
                                     out_dim=args.nn.out_dim)

        #  nns.ConvController(S, out_dim=nn_args.out_dim), ]
        granular = [False, True]

        model = nns.GaussianActorCriticNet(
            S, action_dim,
            phi_body=common,
            actor_body=nn.Linear(nn_args.out_dim, nn_args.out_dim),
            critic_body=nn.Linear(nn_args.out_dim, nn_args.out_dim),
            granular=granular[0]
        )
        trainer = gr.ACTrainer2(env,
                                title=args.title,
                                model=model,
                                lr=0.0005,
                                log_every=args.log_every)
        trainer.train(episodes=10000, steps=10, loss_args=loss_args)

    def test_ac3(self):
        # problem = prob.setup_2room_rect()
        args = base_args('test-20step-stack-2h-tgtcode[0, 1]-noimp_0005')
        args.episodes = int(1e6)
        args.env.lr = 0.0005
        args.inst = Arguments()
        args.inst.depth = 6
        args.inst.box_fmt = 'chw'
        args.steps = 20
        num_spaces = 3
        action_dim = 4
        S = [num_spaces, 20, 20]

        env = DiscreteEnv2(None, None, S[1:],
                           state_cls=RoomStack,
                           inst_args=args.inst,
                           num_spaces=num_spaces,
                           terminal=NoImprovement(limit=4),
                           random_init=True)

        common = nns.CnvController2h(env.input_state_size,
                                     hints_shape=env.objective.area,
                                     out_dim=args.nn.out_dim)

        # granular = [False, True]
        model = nns.GaussianActorCriticNet(
            S, action_dim,
            actor_body=nn.Linear(args.nn.out_dim, args.nn.out_dim),
            critic_body=nn.Linear(args.nn.out_dim, args.nn.out_dim),
            granular=False,
            phi_body=common,
        )
        trainer = gr.ACTrainer2(
            env,
            title=args.title,
            lr=args.env.lr,
            model=model,
            log_every=args.log_every
        )
        trainer.train(episodes=args.episodes,
                      steps=args.steps,
                      loss_args=args.loss)


    def test_ac1(self):
        from src.model.arguments import Arguments
        problem = prob.setup_2room_rect()
        title = 'ac-10step_penalty100k'

        args = Arguments()

        args.title = title
        args.log_every = 100

        env_args = Arguments()
        env_args.random_init = True
        env_args.random_objective = False
        env_args.incomplete_reward = -1
        args.env = env_args

        loss_args = Arguments()
        loss_args.gamma = 0.99
        loss_args.gae_lambda = 0.95
        loss_args.entropy_coef = 0.01
        loss_args.value_loss_coef = 0.5
        loss_args.max_grad_norm = 0.5

        # -----------------------------
        viz = visdom.Visdom()
        viz.text(args.print())
        print(args)

        base = 10
        goal = [base, base]

        size = math.ceil(sum(goal) ** 0.5)
        act = UDLRStep(always_draw=False)
        obj = DiscreteSpaceObjective(goal)
        env = DiscreteEnv(problem, obj, size, act, random_init=True)
        trainer = gr.ACTrainer(env, title=args.title, log_every=args.log_every)

        trainer.train(episodes=100000, steps=base, lr=0.0005, loss_args=loss_args)
        insts = trainer._instances
        num_insts = len(insts)
        print(num_insts)
        gr.save_model(trainer.model, './data/{}.pkl'.format(title))
        for i in range(5):
            utils.layout_disc_to_viz(insts[-i][-1])
        print('num_solutions', len(trainer._solutions))


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



