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
from arg_config import base_args
from src.problem.example import *


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
        lyt = CellComplexLayout(problem, size=(size, size))
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
            actor_body=nn.Linear(args.nn.out_dim, nn_args.out_dim),
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

    def test_stream(self):
        state_dim = [4, 20, 20]
        model = nns.StreamNetFull(
            state_dim=state_dim,
            zdim=64,
            debug=True
        )
        x = torch.randn(1, *state_dim)
        y = torch.randn(1, 12)
        out = model((x, y), None)

        args = base_args('spec')
        args.inst = Arguments()
        args.objective = Arguments()
        args.objective.use_comps = True
        args.episodes = 1000
        args.steps = 5
        args.inst.depth = 4
        env = DiscreteEnv2(None, None, [20, 20],
                           state_cls=ProbStack,
                           inst_args=args.inst,
                           num_spaces=3,
                           problem_gen=problem1,
                           objective_args=args.objective,
                           random_init=True)

        obs_data = env.initialize()
        # debug
        for k, v in out.items():
            if k in ['entropy', 'log_prob']:
                print(k, v.item())
            elif v is None:
                pass
            else:
                print(k, v.size())
        assert 'entropy' in out
        area = state_dim[1] * state_dim[2]

        # im just gonna redo the reward calcs cuz this sucks
        rooms = env._instance.rooms(stack=True)
        areas = np.sum(rooms, axis=(1, 2))
        cvx_areas = [np.sum(skm.convex_hull_image(rooms[i])) for i in range(3)]

        # tests for good numerical properties
        # todo:
        # check that rewards make sense
        print('\nreward', obs_data['reward'], '\ntargets\n', env.objective.keys)
        [print(x) for x in env._objective._constraints]

        print('targets\n', obs_data['feats'])
        print('--')
        print('areas', areas)
        print('convex areas', cvx_areas)
        print([(cvx_areas[i] - areas[i]) / cvx_areas[i]  for i in range(3)])
        assert env._objective.fp_area == area, \
            'footprint area {} {}'.format(env._objective.fp_area, area)
        assert env._objective.fp_area > 0, 'footprint cannot be 0'
        assert area == env._objective.fp_area

    def test_p1(self):
        args = base_args('res-action8,lr:1e-4,m15_BN')
        args.objective = Arguments()
        args.objective.use_comps = True

        # """
        args.train.episodes = int(1e6)
        args.train.detail_every = 200
        args.train.log_every = 50
        """
        args.train.episodes = 5  # int(1e6)
        args.train.detail_every = 1
        args.train.log_every = 1
        """

        args.train.steps = 15
        # Layout Instance ARguments
        args.inst.depth = 4
        args.inst.init_dist = 'scaled-un'  # 'scaled-un'
        args.inst.cls = ProbStack

        args.env.incomplete_reward = 0.0001
        args.env.terminal = NoImprovement(limit=4)

        #
        args.nn.zdim = 64
        args.train.lr = 0.0001

        env = DiscreteEnv2(None, None, [20, 20],
                           state_cls=ProbStack,
                           inst_args=args.inst,
                           num_spaces=3,
                           problem_gen=problem1,
                           objective_args=args.objective,
                           terminal=args.env.terminal,
                           unsolved_problem_reward=args.env.incomplete_reward,
                           random_init=args.env.random_init)

        # print(env._state_cls)
        # print(env.input_state_size)
        model = nns.StreamNet2(
            state_dim=env.input_state_size,
            zdim=args.nn.zdim,
            debug=True,
        )
        trainer = gr.ACTrainer3(
            env,
            title=args.title,
            lr=args.train.lr,
            model=model,
            argobj=args,

        )
        trainer.train(episodes=args.train.episodes,
                      steps=args.train.steps,
                      loss_args=args.loss)

    def test_supervised(self):



    def test_lpent(self):
        import src.probablistic.funcs as fp
        import torch.distributions as dist
        softmax = nn.Softmax2d()
        size = (1, 3, 5, 5)
        # baseline is random uniform
        x1 = torch.FloatTensor(*size).uniform_(0.2, 0.4)

        x2 = torch.zeros(*size)
        x2[:, 0, 0:2, :] = 1.
        x2[:, 1, 2:, :2] = 1.
        x2[:, 2, 2:, 2:] = 1.

        x3 = (x1 + x2) / 2

        tests= [x1, x3, x2]

        print(x3)
        # kl_hard = [fp.kl_divirgence2d(x).mean() for x in tests]
        kl_soft = [fp.kl_divirgence2d(softmax(x)).mean() for x in tests]

        # ent_hard = [fp.entropy2d(x).mean() for x in tests]
        ent_soft = [fp.entropy2d(softmax(x)).mean() for x in tests]
        print('ents', ent_soft)
        print('probs', kl_soft)

        print([dist.Categorical(x.squeeze()).entropy().mean() for x in tests])
        print([dist.Categorical(logits=x.squeeze()).entropy().mean() for x in tests])
        print([dist.Categorical(softmax(x).squeeze()).entropy().mean() for x in tests])
        print([dist.Categorical(softmax(x).squeeze()).entropy().size() for x in tests])

        for x in tests:
            dc = dist.Categorical(logits=x.permute(0, 2, 3, 1))
            # print(dc.logits.size())
            maxs = x.max(dim=1)[0].squeeze()
            sample = dc.sample()
            # print(sample.size(), maxs.size())
            # print(maxs.size())
            print(dc.entropy().sum(), dc.log_prob(sample).mean(), dc.log_prob(maxs).mean())

    def test_ac1(self):
        problem = prob.setup_2room_rect()
        title = 'ac-10step_penalty100k'
        args = base_args('-')
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



