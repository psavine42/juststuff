import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import os

import torch
from torch.optim import LBFGS, Adam
import torch.nn as nn
import torch.random
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torchvision.transforms as T
from shapely.geometry import LineString, MultiLineString, MultiPolygon
import operator
from lib.hessian.hessian import hessian, gradient

from src.algo.nns import DQN, ReplayMemory, Transition
import src.algo.nns as nns
from src.problem import Problem
from src.structs.floating_tree import FloatingNode, PolyTree
from src.building import Room
from src.layout import BuildingLayout
import random
import matplotlib
from collections import OrderedDict as odict
from itertools import count
from PIL import Image
import torchnet as tnt
import src.utils as utils
import src.actions.basic as A
import src.geom.cg as gcg
from src.metrics.meters import AllMeters

s = 10
torch.manual_seed(s)
random.seed(s)
np.random.seed(s)


def lossfn(X):
    return (4 * X[0] - X[1] ** 2 + X[2] ** 2 - 1) ** 2


def lossfn2(X):
    return torch.abs(4 * X[0] * X[1] + X[2] * X[3] - 1)


def lossfn3(X, show=False):
    """ fill the space """
    bounds_not_overlap_x = torch.abs(torch.abs(X[0] - X[4]) - torch.abs(X[2] - X[6]))
    bounds_not_overlap_y = torch.abs(torch.abs(X[1] - X[5]) - torch.abs(X[3] - X[7]))
    area_filled = torch.abs(1 - 4 * (X[2] * X[3] + X[6] * X[7]))

    aspect1 = torch.abs(X[2] - X[3]) / torch.max(X[2:4])
    aspect2 = torch.abs(X[6] - X[7]) / torch.max(X[6:8])

    tgt_area1 = torch.abs(0.5 - 4 * (X[2] * X[3]))
    tgt_area2 = torch.abs(0.5 - 4 * (X[6] * X[7]))
    if show is True:
        print([round(x.item(), 3) for x in
               [bounds_not_overlap_x, bounds_not_overlap_y,
                area_filled,
                aspect2, aspect1,
                tgt_area1, tgt_area2
                ]])
    return bounds_not_overlap_y + bounds_not_overlap_x + area_filled + \
        aspect2 + aspect1 + tgt_area1 + tgt_area2


def lossfn4(X):
    """ fill the space """
    bounds_not_overlap_x = torch.abs(torch.abs(X[0] - X[4]) - torch.abs(X[2] - X[6]))
    bounds_not_overlap_y = torch.abs(torch.abs(X[1] - X[5]) - torch.abs(X[3] - X[7]))
    # bounds_not_overlap_x = torch.abs(torch.abs(X[2] - X[6]) - torch.abs(X[0] - X[4]))
    # bounds_not_overlap_y = torch.abs(torch.abs(X[3] - X[7]) - torch.abs(X[1] - X[5]))
    area_filled = torch.abs(1 - 4 * (X[2] * X[3] + X[6] * X[7]))

    dims1 = (1 - 2* X[2]) + (1 - 2 * X[6])
    dims2 = (0.5 - 2* X[3]) + (0.5 - 2* X[7])

    tgt_area1 = torch.abs(0.5 - 4 * (X[2] * X[3]))
    tgt_area2 = torch.abs(0.5 - 4 * (X[6] * X[7]))

    print([round(x.item(), 3) for x in
           [bounds_not_overlap_x, bounds_not_overlap_y,
            area_filled,
            dims1, dims2,
            tgt_area1, tgt_area2
            ]])
    return bounds_not_overlap_y + bounds_not_overlap_x + area_filled + \
        dims1 + dims2 + tgt_area1 + tgt_area2


def prob1():
    p1 = [0.25, 0.25, 0.25, 0.25]
    p2 = [0.75, 0.75, 0.25, 0.25]
    return torch.tensor(p1 +p2, requires_grad=True)


def sol1():
    p1 = [0.5, 0.25, 0.5, 0.25]
    p2 = [0.5, 0.75, 0.5, 0.25]
    return torch.tensor(p1 +p2, requires_grad=True)


def ehess(lossfn, prob=prob1):
    """
    Γ_v ← Γ_i + v
    """
    X = prob()
    energy = lossfn(X)

    # sample v from eigenvectors of hessians of E(Γ_i)
    h = hessian(energy, X, create_graph=True)
    hn = h.detach().numpy()
    eig_vals, eig_vecs = np.linalg.eig(hn)
    eig_vecs = torch.from_numpy(eig_vecs)
    print(eig_vals)
    min_eig = np.min(eig_vals)
    eig_idx = np.argsort(eig_vals)

    # create soft-constraint preserving locations
    new_xs = []
    for i in range(5):
        eix = np.where(eig_idx == i)
        v_move = eig_vecs[eix]
        print(v_move )
        new_x = X + v_move
        new_xs.append(new_x)
        # check energy
        # E(Γ v ) ≈ E(Γ i ) + (Γ v − Γ i ) T H| Γ i (Γ v − Γ i )/2
        # E(Γ v ) ≈ E(Γ i ) + SUM(γ_j^2 * λ_j /2)
        check = lossfn(new_x) - energy + eig_vals[i]/2
        # assert np.allclose(check, 0)
        print(check)

    # optimize for hard
    # gi = torch.dot(X, eig_vecs[0])
    # print()
    # eig = torch.eig(h, eigenvectors=True)
    # print(neig)


class Opt(nn.Module):
    def __init__(self, size):
        super(Opt, self).__init__()
        self.l1 = nn.Linear(size, 2 * size)
        self.l2 = nn.Linear(2 * size, size)

    def forward(self, input):
        X = self.l1(input)
        return self.l2(X)

    def init_normal(self):
        # if type(m) == nn.Linear:
        nn.init.uniform_(self.l1.weight, a=0., b=0.5)
        nn.init.uniform_(self.l2.weight, a=0., b=0.5)
        nn.init.uniform_(self.l1.bias, a=0., b=0.1)
        nn.init.uniform_(self.l2.bias, a=0., b=0.1)


def bgfs(lossfn, iters=20, size=4):
    torch.randn(3)
    # model = nn.Linear(size, size)
    model = Opt(size)
    model.init_normal()
    optimizer = Adam(model.parameters())
    p = prob1()
    print(p)
    for i in range(iters):
        # def closure():
        print('\n')
        optimizer.zero_grad()
        x = model(p)
        # print(x)
        loss = lossfn2(x)

        loss.backward()
        optimizer.step()
        if lossfn(model(p)).item() < 0.005:
            print('end')
            print(model(p), lossfn(model(p)).item())
            break

        print(model(p), lossfn(model(p)).item())


def floating_trees(problem):
    """
    F(x_t) -> F(x_t+1):

        loss = M(x_t)
        g = grad(x_t, loss)
        d = optimizer.step(x_t, loss, g)
        x_t+1 = x_t + d

    a floating tree is a geometric object where nodes have coordinates:

    modification of top level geometric parameters effects lower level
    final positions

    Node parametrizations :
        [angle, distance]
        [x, y]

    Action parametrizations :
        action set size = 2 * num_tree_nodes

        if an action causes two boundaries to collide,
        need to calculate a split with own parameters.

        move := array.shape(3, dtype=int) [index, dx, dy]
        scale := <index, x>

        add_branches array[index,

        remove_branches

    Input:
        1) image of space
        2) plane for each covering
        3) plane for each covering's nodes.
        4) Encoded objectives:
            - vector of target areas
            - vector of min-max areas
            - adjacency matrix
            - non-adjacency matrix
            -
    Output:

        a) action type [ move, delete, add, done ], bit [ done ]
            geom:
            1) Move Action as: (point, point) 2 x 1-hot distributions over plane (from, to)
            2) Move Action as: (point, point) 4 x 1-hot distributions over (omega.x, omega.y) * (from, to)
            3) Add node a:  (point, None)
            4) Remove Node as:  (point, None)
            5) done as [0, 1]

        b) [Action, covering_idx, geom]

            I.
            1) Action onehot of size [ num_actions ]
            2) Covering onehot of size [ num_covers ]
            3) Geom as 2 (or 4) onehot of size [ omega ]

            II.
            1) Action is one_hot of size 8 ( two for each square side)
            If each 'room' has its own agent, thats it, otherwise 8 * num_rooms
            -or-
            Master Controller learns who moves and do one of these:

                slave_index <- Master(state)
                action.size(n) <- Slave(state, slave_index)

                -or-

                slave_index <- Master(state)
                action.size(n) <- slaves[slave_index](state)

                -or-

                slave_index <- Master(state)
                state_given_slave <- fn(state, slave_index)
                action.size(n) <- Slave(state_given_slave)

    Boundary:
        Tree will return a ccw list of coordinates

    ------------------
    given this object, a SearchTree of transformations can be created

    -- or --

    RL can be trained to do make the actions.

    """
    xy1 = np.asarray([2, 2], dtype=int)
    xy2 = np.asarray([7, 7], dtype=int)
    left = np.asarray([-1, 0], dtype=int)
    right = np.asarray([+1, 0], dtype=int)

    left_down  = np.asarray([-1, -1])
    left_up    = np.asarray([-1, +1])
    right_up   = np.asarray([+1, +1])
    right_down = np.asarray([+1, -1])

    roots = []
    for xy in [xy1, xy2]:
        root1 = PolyTree(xy)
        root1.left = FloatingNode(xy + left)
        root1.right = FloatingNode(xy + right)
        root1.left.left = FloatingNode(xy + left + left_down)
        root1.left.right = FloatingNode(xy + left + left_up)
        root1.right.left = FloatingNode(xy + right + right_up)
        root1.right.right = FloatingNode(xy + right + right_down)
        roots.append(root1)

    rooms = []
    for i, root in enumerate(roots):
        rooms.append(Room(list(root.get_boundary()), name=str(i)))

    from src.layout import BuildingLayout
    layout = BuildingLayout(None, rooms=rooms)
    X = torch.from_numpy(layout.to_vec4())
    print(X)
    loss2 = lossfn3(X)
    loss4 = lossfn4(X)
    print(loss2, loss4)

    return roots


def trees_to_layout(trees):
    return BuildingLayout(None, rooms=[Room(list(t.get_boundary())) for t in trees])


def all_angles(layout):
    thetas = []
    for room in layout:
        crd = np.asarray(list(room.exterior.coords))
        for i in range(len(crd)):
            v1 = crd[i - 2] - crd[i-1]
            v2 = crd[i - 1] - crd[i]
            ft = acos(np.dot(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2) ))
            thetas.append(ft if isinstance(ft, float) else 0)
    return thetas


class Annealer(object):
    def __init__(self, transition=None, tmax=20000, tmin=1):
        self._transition = transition
        self.engy_hist = []
        self.best_hist = []
        self.Tmax = tmax
        self.Tmin = tmin

    def __call__(self, layout, cost_fn, transition,
                 num_iter=10000,
                 log_every=10):
        """
        sim anneal
        Let s = s0
        For k = 0 through kmax (exclusive):
            T ← temperature(k ∕ kmax)
            Pick a random neighbour, snew ← neighbour(s)
            If P(E(s), E(snew), T) ≥ random(0, 1):
                s ← snew
        Output: the final state s

        nd exp ( − ( e′ − e ) / T ) {\exp(-(e'-e)/T)} \exp(-(e'-e)/T) otherwise.

        """
        t_factor = -np.log(self.Tmax / self.Tmin)

        # counters
        self.engy_hist = []
        self.best_hist = []
        e_tmp = []
        rejected = 0
        accepted = 0
        improved = 0
        st_hist = []
        # state
        energy = cost_fn(layout)
        state = layout

        best_state = layout
        best_energy = energy

        for k in range(1, num_iter):
            T = self.Tmax * np.exp(t_factor * k / num_iter)

            state_new = transition(state)
            # ng = [x for x in all_angles(state_new) if int(2 * x / np.pi) % 2 != 0 ]
            # if ng:
            #     print(k, ng)
            if len(st_hist) > 50:
                st_hist.pop(0)
            st_hist.append(state_new)
            try:
                energy_new = cost_fn(state_new)
            except:
                utils.plotpoly(st_hist)
                print(state_new)
                return
            delta_energy = energy_new - energy
            if delta_energy > 0.0 and np.exp(- delta_energy / T) < random.random():
                rejected += 1
            else:
                accepted += 1
                e_tmp.append(energy_new)
                energy = energy_new
                state = state_new
                # state_hist.append(state_new)
                if energy_new < best_energy:
                    improved += 1
                    best_energy = energy_new
                    best_state = state_new
                    self.best_hist.append([k, energy_new, state_new])

                if accepted % log_every == 0:
                    mean_energy = np.mean(e_tmp)
                    self.engy_hist.append(mean_energy)
                    print('step {} temp {}, mean E: {} energies {}, {}'.format(
                        k, T, mean_energy, energy, energy_new)
                    )

        print('completed, accepted {}, rejected: {}, improved, {} energy {}'.format(
            accepted, rejected, improved, energy)
        )
        utils.plotpoly([x for i, e, x in self.best_hist],
                       titles=['iter {}, E:{}'.format(i, round(e, 5)) for i, e, x in self.best_hist])
        return best_state


def layout_to_tensor(layout, *dim):
    if isinstance(layout, list):
        return torch.from_numpy(np.stack([x.to_tensor(*dim) for x in layout], 0)).float()
    else:
        return torch.from_numpy(layout.to_tensor(*dim)).unsqueeze(0).float()


class NNEncoder:
    def __init__(self, problem, num_ents=2, size=(20, 20)):
        self._problem = problem
        self._dx = size[0]
        self._dy = size[1]
        self._dim = size[0:2]
        self.num_ents = num_ents
        self._zsize = (1, 1, self._dx, self._dy)

    @staticmethod
    def _batch(np_arr):
        return torch.from_numpy(np_arr).unsqueeze(0).float()

    def room_features(self, layout):
        return self._batch(layout.to_tensor(self._dx, self._dy, self.num_ents))

    def rooms_mat(self, layout):
        footprint = layout.write_mat(layout.geoms, *self._dim)
        return self._batch(footprint).sum(0, keepdim=True)

    def footprint(self, layout):
        """ a plane with all """
        footprint = layout.write_mat([self._problem.footprint], *self._dim)
        return self._batch(footprint)

    def room_edge(self, layout, room_index, item_idx):
        """ a plane with just 'selected' edge """
        scale = layout.get_scale_for_size(self._dx, self._dy)
        coord = (np.asarray([list(xy) for i, xy in layout[room_index][item_idx]]) // scale).astype(int)
        masks = gcg.discretize_segment(coord[0].tolist(), coord[1].tolist())
        plane = torch.zeros(1, 1, self._dx, self._dy)
        for x, y in masks:
            plane[0, 0, min(x, self._dx - 1), min(y, self._dy - 1)] = 1
        return plane

    def edges(self, layout):
        """ a plane with all the edges = 1, else 0 """
        return

    def negative_space(self, layout):
        """ a plane with all unoccopied cells = 1, else 0"""
        return torch.ones(*self._zsize) - self.rooms_mat(layout)

    def rooms_line_feats(self, layout, room_index, item_index):
        room = self.room_features(layout)
        room_edge = self.room_edge(layout, room_index, item_index)

    def encode_constraints(self, layout):
        ents = len(layout)
        feat = [0] * (ents * 2)
        area = self._problem.footprint.area
        for i, r in enumerate(layout.rooms()):
            feat[i] = r.area / area



class ConstraintEncoder:
    def __init__(self, problem):
        self.problem = problem

    def eval(self, layout):
        for room in layout.rooms():
            pass


class JoyStick:
    def __init__(self, problem, size=(20, 20)):
        self.problem = problem
        self.state = torch.zeros(*size)
        self._selected_room = None
        self._selected_line = False

    def update(self, state, action):
        """
        actions are [Up, down, left right (maybe combo) ] * [sel, not-sel] = 8 (or 16)

        1) if selecte is
        """

        if self._selected_room is not None:
            return


class FloatingTreeExp1:
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10

    def __init__(self, env, num_moves=10, out_size=7,
                 size=(20, 20, 2),
                 title='',
                 debug=True,
                 mem_size=5000,
                 log_every=10,
                 mode='vanilla'):
        self.dim = size
        self._dx = size[0]
        self._dy = size[1]
        self.env = env
        self.mode = mode
        self._title = title

        self._debug = debug
        self.BATCH_SIZE = 64

        # [up, down, left, right] * num_room * [1, -1]
        self.num_move = num_moves
        self.num_ents = len(env.problem)
        if mode == 'line':
            self.out_size = out_size
        elif mode == 'room':
            self.out_size = num_moves * self.num_ents
        else:
            self.out_size = num_moves * self.num_ents

        self.problem = env.problem
        self.next_actns = 0
        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(size[0], size[1], self.out_size, in_size=size[2]).cuda()
        self.target_net = DQN(size[0], size[1], self.out_size, in_size=size[2]).cuda()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        self.memory = ReplayMemory(mem_size)
        # self.n_actions = 1
        self.log_every = log_every
        self.M = AllMeters(title)

    def step(self, layout, action, item, item_index=None):
        """ [up, down, left, right] * num_room * [1, -1]
        room = num_wall_moves * 2[-1,+1] + 4uplr
        room = 4 * 2 + 4 = 12
        """
        room_num = item // self.num_move
        # print('move: ', item, room_num, self.num_move, self.out_size)
        action_idx = item - room_num * self.num_move

        index = action_idx if action_idx < self.num_move // 2 else action_idx // 2
        sign = 1 if action_idx < self.num_move // 2 else -1

        room_name = layout[room_num].name
        params = dict(room=room_name, item=index, mag=1, sign=sign)
        next_layout = self.env.step(layout, **params)
        reward = self.env.reward(next_layout)
        return next_layout, reward, False

    @staticmethod
    def enclayout(layout, dx, dy, num_ents, room_index=None, item_idx=None):

        state = layout_to_tensor(layout, dx, dy, num_ents)
        if room_index is not None and item_idx is None:
            state = torch.cat((state, state[:, room_index, :, :].unsqueeze(1)), 1)

        elif room_index is not None and item_idx is not None:
            scale = layout.get_scale_for_size(dx, dy)
            coord = (np.asarray([list(xy) for i, xy in layout[room_index][item_idx]]) // scale).astype(int)
            masks = gcg.discretize_segment(coord[0].tolist(), coord[1].tolist())
            plane = torch.zeros(1, 1, dx, dy)
            for x, y in masks:
                plane[0, 0, min(max(x, 0), dx - 1), min(max(y, 0), dy - 1)] = 1
            state = torch.cat((state, plane), 1)
        else:
            state = torch.cat((state, torch.zeros(1, 1, dx, dy)), 1)
        return state

    def layout_to_tensor(self, layout, room_index=None, item_idx=None):
        """

        """
        state = layout_to_tensor(layout, self._dx, self._dy, self.num_ents)
        if room_index is not None and item_idx is None:
            state = torch.cat((state, state[:, room_index, :, :].unsqueeze(1)), 1)

        elif room_index is not None and item_idx is not None:
            scale = layout.get_scale_for_size(*self.dim[0:2])
            coord = (np.asarray([list(xy) for i, xy in layout[room_index][item_idx]]) // scale).astype(int)
            masks = gcg.discretize_segment(coord[0].tolist(), coord[1].tolist())
            plane = torch.zeros(1, 1, self._dx, self._dy)
            for x, y in masks:
                plane[0, 0, min(max(x, 0), self._dx -1), min(max(y, 0), self._dy -1)] = 1
            state = torch.cat((state, plane), 1)
        else:
            state = torch.cat((state, torch.zeros(1, 1, self._dx, self._dy)), 1)
        return state

    def select_action(self, state):
        eps = self.EPS_START - self.EPS_END
        eps_threshold = self.EPS_END + eps * np.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            self.explore_exploit.add(1)
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            self.explore_exploit.add(0)
            return torch.tensor([[random.randrange(self.out_size)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                      device=self.device,
                                      dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # print(self.steps_done, loss, state_action_values.size(), expected_state_action_values.unsqueeze(1).size())
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # record loss an Q values -
        self.loss_meter.add(loss.cpu().item())
        # self.qval_meter.add(torch.mean(next_state_values).cpu().item())
        self.qval_meter.add(torch.mean(expected_state_action_values).cpu().item())

    def train(self, episodes=100, steps=10):
        episode_ends = []
        episode_rewards = []
        actions = [0] * self.out_size
        steps_total = 0

        for i_episode in range(1, episodes):

            # Initialize the environment and state

            layout = self.env.initialize()
            room = 0
            item = np.random.randint(1, len(list(layout[room].exterior.coords)[0:-1]))

            state = self.layout_to_tensor(layout, room, item).cuda()
            reward = 0

            # bookkeeping
            first_reward = None
            best_reward = self.env.reward(layout)
            nstep = 0
            # for t in count():
            for t in range(steps):
                steps_total += 1
                nstep = t
                # tensor.size([batch_size, 1])
                action = self.select_action(state)
                actions[action.item()] += 1

                # apply action
                next_layout, reward, fail = self.step(layout, action.item(), room, item)
                # reward = 2 * reward - 1
                reward_tsr = torch.tensor([reward], device=self.device).float()

                best_reward = max(reward, best_reward)
                if first_reward is None:
                    first_reward = reward
                if fail is False:
                    # nexts
                    room = t % self.num_ents
                    # setting this randomly for now
                    item = np.random.randint(1, len(list(next_layout[room].exterior.coords)[0:-1]))
                    next_state = self.layout_to_tensor(next_layout, room, item).cuda()
                    layout = next_layout
                else:
                    next_state = state
                    # next_state = None

                self.memory.push(state, action, next_state, reward_tsr)
                state = next_state
                self.optimize_model()

                # if done is True:
                #    break
                self.reward_meter.add(reward)

            self.duration_meter.add(nstep)
            self.max_reward_meter.add(best_reward)
            # self.improved_meter.add(reward - first_reward)
            # Update the target network, copying all weights and biases in DQN

            if i_episode % self.log_every == 0:
                # logging reward,
                print('ep', i_episode, nstep, best_reward, reward)
                episode_ends.append(layout)
                self.reward_logger.log(i_episode, (self.max_reward_meter.value()[0],
                                                   self.reward_meter.value()[0],
                                                   self.qval_meter.value()[0]
                                                   ))
                self.duratn_logger.log(i_episode, [x / steps_total for x in actions])
                self.losses_logger.log(i_episode, self.loss_meter.value())
                steps_total = 0
                actions = [0] * self.out_size
                # self.explor_logger.log(i_episode, self.explore_exploit.value())
                self.reset_meters()

            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        print('Complete')
        utils.plotpoly(episode_ends, show=False, figsize=(10, 10))
        self.viz.matplot(plt)


class RL2(FloatingTreeExp1):
    def __init__(self, *args, **kwargs):
        FloatingTreeExp1.__init__(self, *args, **kwargs)

    def step(self, layout, action, room_index=None, item_index=None):
        """ case 1 - output is right or left or Noop (0) HA !  [0, 1, 3]"""
        size = self.out_size // 2

        sign = 1 if action == size else -1
        mag = -1*action if action < size else action - size

        room_name = layout[room_index].name
        params = dict(room=room_name, item=item_index, mag=mag, sign=sign)
        next_layout = self.env.step(layout, **params)
        reward = self.env.reward(next_layout)
        fail = True if next_layout is None else False
        return next_layout, reward, fail


class Reinforced:
    def __init__(self, env, num_actions=3, batch_size=1, title='', size=(20, 20, 2), log_every=10):
        self.env = env
        self.size = size
        self._title = title

        self.BATCH_SIZE = batch_size
        self.num_actions = num_actions
        self.num_ents = 2

        self.log_every = log_every
        self._debug = False
        self.eps = np.finfo(np.float32).eps.item()
        self._improve_thresh = 15

        self.GAMMA = 0.99
        self.policy_net = nns.DQNC(size[0], size[1], self.num_actions, in_size=size[2]).cuda()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.000075)
        self.M = AllMeters(title)
        self.__save_log_probs = []
        self.__policy_rewards = []
        self._episode_ends = []

    def optimize_model(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.__policy_rewards[::-1]:
            R = r + self.GAMMA * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.__save_log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # meters
        self.M.loss_meter.add(loss.cpu().item())
        self.M.qval_meter.add(torch.mean(torch.stack(self.__save_log_probs)).cpu().item())

        # cleanup
        del self.__save_log_probs[:]
        del self.__policy_rewards[:]

    def select_action(self, state):
        probs = self.policy_net(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        self.__save_log_probs.append(log_prob)
        return action.cpu().item()

    def layout_to_tensor(self, layout, room, item):
        return FloatingTreeExp1.enclayout(
            layout, self.size[0], self.size[1], self.num_ents, room, item
        )

    # def step(self, layout, action, room_index=None, item_index=None):
    #     """ case 1 - output is right or left or Noop (0) HA !  [0, 1, 3]"""
    #     size = self.num_actions // 2
    #
    #     sign = 1 if action == size else -1
    #     mag = -1*action if action < size else action - size
    #
    #     room_name = layout[room_index].name
    #     params = dict(room=room_name, item=item_index, mag=mag, sign=sign)
    #     next_layout = self.env.step(layout, **params)
    #     reward = self.env.reward2(next_layout)
    #     fail = True if next_layout is None else False
    #     return next_layout, reward, fail
    def wrap_feats(self, mat, imp=0):
        mat = torch.from_numpy(mat).float().view(-1)
        mat = torch.cat((mat, torch.tensor([imp / self._improve_thresh])))
        return mat.cuda()

    def train(self, episodes=100, steps=10, fig_size=(20, 20)):
        actions = [0] * self.num_actions
        steps_total = 0
        self._episode_ends = []
        episode_titles = []
        for i_episode in range(1, episodes):

            # Initialize the environment and state
            room = 0
            layout = self.env.initialize()
            item = np.random.randint(1, len(list(layout[room].exterior.coords)[0:-1]))

            reward, mat = self.env.reward2(layout, encode=True)
            state = self.layout_to_tensor(layout, room, item).cuda()
            state = (state, self.wrap_feats(mat))

            # bookkeeping
            first_reward, best_reward = reward, reward
            nstep = 0
            no_improvement = 0
            # for t in count():
            for t in range(steps):
                steps_total += 1
                nstep = t
                # tensor.size([batch_size, 1])
                action = self.select_action(state)
                actions[action] += 1

                # apply action -> layout(possibly None)
                state_dict = self.env.step_enc(layout, action, room, item)
                next_layout, next_reward = state_dict['layout'], state_dict['reward']

                if next_reward > reward:
                    no_improvement = 0
                elif reward >= next_reward:
                    no_improvement += 1

                if no_improvement > self._improve_thresh:
                    if abs(1 - next_reward) < 0.001:
                        self.__policy_rewards.append(1)
                    else:
                        self.__policy_rewards.append(-1)
                    break

                self.__policy_rewards.append(next_reward)
                # room = t % self.num_ents
                # item = np.random.randint(1, len(list(next_layout[room].exterior.coords)[0:-1]))
                if state_dict['fail'] is False:
                    room = t % self.num_ents
                    item = np.random.randint(1, len(list(next_layout[room].exterior.coords)[0:-1]))
                    mat = self.wrap_feats(state_dict['feats'], no_improvement)
                    state = (self.layout_to_tensor(next_layout, room, item).cuda(), mat)
                    layout = next_layout
                    self.M.fail_meter.add(0)
                else:
                    self.M.fail_meter.add(1)
                    # next_state = state
                # next object

                # meters
                self.M.advantage_meter.add(reward)
                self.M.reward_meter.add(next_reward)
                best_reward = max(next_reward, best_reward)
                if next_reward > 0:
                    reward = next_reward

            self.optimize_model()
            self.M.duration_meter.add(nstep)
            self.M.max_reward_meter.add(best_reward)
            # self.improved_meter.add(reward - first_reward)
            if i_episode % self.log_every == 0:
                print('ep', i_episode, nstep, best_reward, reward)
                self._episode_ends.append(layout)
                episode_titles.append('ep:{}'.format(i_episode))
                self.M.reward_logger.log(i_episode, (self.M.max_reward_meter.value()[0],
                                                     self.M.reward_meter.value()[0],
                                                     self.M.fail_meter.value()[0]
                                                     ))
                self.M.action_logger.log(i_episode, [x / steps_total for x in actions])
                self.M.losses_logger.log(i_episode, (self.M.loss_meter.value()[0],
                                                     self.M.loss_meter.value()[1],
                                                     self.M.qval_meter.value()[0],
                                                     )
                                         )
                self.M.duratn_logger.log(i_episode, self.M.duration_meter.value())
                steps_total, actions = 0, [0] * self.num_actions
                self.M.reset()

        print('Complete')
        utils.plotpoly(self._episode_ends, show=False, figsize=fig_size, titles=episode_titles)
        self.M.viz.matplot(plt)
        if not os.path.exists('./data/{}.pkl'.format(self._title)):
            torch.save(self.policy_net, './data/{}.pkl'.format(self._title))

    def log(self, i_episode, nstep, best_reward, reward, actions, steps_total, layout):
        print('ep', i_episode, nstep, best_reward, reward)
        self.episode_ends.append(layout)
        self.M.reward_logger.log(i_episode, (self.M.max_reward_meter.value()[0],
                                             self.M.reward_meter.value()[0],
                                             # self.M.qval_meter.value()[0]
                                             ))
        self.M.duratn_logger.log(i_episode, [x / steps_total for x in actions])
        self.M.losses_logger.log(i_episode, self.M.loss_meter.value())
        steps_total = 0
        actions = [0] * self.num_actions
        self.M.reset()

    def test(self, path=None, steps=10):
        self.policy_net = torch.load('./data/{}.pkl'.format(self._title))
        rooms, items, layouts, rewards, actions = [], [], [], [], []

        room = 0
        layout = self.env.initialize()
        item = np.random.randint(1, len(list(layout[room].exterior.coords)[0:-1]))
        state = self.layout_to_tensor(layout, room, item).cuda()
        reward = self.env.reward2(layout)

        for t in range(steps):
            action = self.select_action(state)

            actions.append(action)
            rewards.append(reward)
            items.append(item)
            rooms.append(room)
            layouts.append(layout)
            next_layout, reward, fail = self.step(layout, action, room, item)

            if fail is False:
                room = t % self.num_ents
                item = np.random.randint(1, len(list(next_layout[room].exterior.coords)[0:-1]))
                next_state = self.layout_to_tensor(next_layout, room, item).cuda()
                layout = next_layout
            else:
                next_state = state
            state = next_state

        for i in range(len(rewards)):
            title = 'a:{}'.format()



class RandomWalk:
    """ random actions benchmark """
    def __init__(self, env, num_actions=3, log_every=10, title=''):
        self.env = env
        self.num_actions = num_actions
        self.num_ents = len(env.problem)
        self.M = AllMeters(title)
        self.log_every = log_every

    def select_action(self, state):
        return random.choice([0, 1, 2])

    def step(self, layout, action, room_index=None, item_index=None):
        """ case 1 - output is right or left or Noop (0) HA !  [0, 1, 3]"""
        size = self.num_actions // 2

        sign = 1 if action == size else -1
        mag = -1*action if action < size else action - size

        room_name = layout[room_index].name
        params = dict(room=room_name, item=item_index, mag=mag, sign=sign)
        next_layout = self.env.step(layout, **params)
        reward = self.env.reward2(next_layout)
        fail = True if next_layout is None else False
        return next_layout, reward, fail

    def train(self, episodes=100, steps=10, fig_size=(20, 20)):
        self.episode_ends = []
        actions = [0] * self.num_actions
        for i_episode in range(1, episodes):
            room, steps_total  = 0, 0
            layout = self.env.initialize()
            item = np.random.randint(1, len(list(layout[room].exterior.coords)[0:-1]))
            reward = self.env.reward2(layout)
            first_reward, best_reward = reward, reward

            for t in range(steps):
                steps_total += 1
                action = self.select_action(layout)
                actions[action] += 1
                next_layout, reward, fail = self.step(layout, action, room, item)
                reward = self.env.reward2(layout)
                if fail is False:
                    room = t % self.num_ents
                    item = np.random.randint(1, len(list(next_layout[room].exterior.coords)[0:-1]))
                    layout = next_layout

                self.M.reward_meter.add(reward)

            best_reward = max(reward, best_reward)
            self.M.max_reward_meter.add(best_reward)

            if i_episode % self.log_every == 0:
                print('ep', i_episode, best_reward, reward)
                self.episode_ends.append(layout)
                self.M.reward_logger.log(i_episode, (self.M.max_reward_meter.value()[0],
                                                     self.M.reward_meter.value()[0],
                                                     # self.M.qval_meter.value()[0]
                                                     ))
                self.M.duratn_logger.log(i_episode, [x / steps_total for x in actions])
                steps_total, actions = 0, [0] * self.num_actions
                self.M.reset()
        print('Complete')
        # utils.plotpoly(self.episode_ends, show=False, figsize=fig_size)
        # self.M.viz.matplot(plt)



def grass_drass():
    """ algorithm from GRASS / DRASS papers """


def spatial_planner():
    """ algorithm from adsk Spatial Layout git repo

    """
    from scipy.spatial import KDTree
    # KDTree



def generalized_reduced_gradient():
    x1, x2, x3 = symbols('x1 x2 x3')
    xvars = [x1, x2, x3]

    fx = 4 * x1 - x2 ** 2 + x3 ** 2 - 12  # Function to be minimized
    hxs = [20 - x1 ** 2 - x2 ** 2, x1 + x3 - 7]  # Constraints to be obeyed
    alpha_0 = 1  # Parameter initializations
    gamma = 0.4
    max_iter = 100
    max_outer_iter = 50
    eps_1, eps_2, eps_3 = 0.001, 0.001, 0.001

    xcurr = np.array([2, 4, 5])  # Starting solution

    dfx = np.array([diff(fx, xvar) for xvar in xvars])
    dhxs = np.array([[diff(hx, xvar) for xvar in xvars] for hx in hxs])
    nonbasic_vars = len(xvars) - len(hxs)
    opt_sols = []

    for outer_iter in range(max_outer_iter):

        print('\n\nOuter loop iteration: {0}, optimal solution: {1}'.format(outer_iter + 1, xcurr))
        opt_sols.append(fx.subs(zip(xvars, xcurr)))

        # Step 1

        delta_f = np.array([df.subs(zip(xvars, xcurr)) for df in dfx])
        delta_h = np.array(
            [[dh.subs(zip(xvars, xcurr)) for dh in dhx] for dhx in dhxs])  # Value of h'_i(xcurr) for all i
        J = np.array([dhx[nonbasic_vars:] for dhx in delta_h])  # Computation of J and C matrices
        C = np.array([dhx[:nonbasic_vars] for dhx in delta_h])
        delta_f_bar = delta_f[nonbasic_vars:]
        delta_f_cap = delta_f[:nonbasic_vars]

        J_inv = np.linalg.inv(np.array(J, dtype=float))
        delta_f_tilde = delta_f_cap - delta_f_bar.dot(J_inv.dot(C))

        # Step 2

        if abs(delta_f_tilde[0]) <= eps_1:
            break

        d_bar = - delta_f_tilde.T  # Direction of search in current iteration
        d_cap = - J_inv.dot(C.dot(d_bar))
        d = np.concatenate((d_bar, d_cap)).T

        # Step 3

        alpha = alpha_0

        while alpha > 0.001:

            print('\nAlpha value: {0}\n'.format(alpha))

            # Step 3(a)

            v = xcurr.T + alpha * d
            v_bar = v[:nonbasic_vars]
            v_cap = v[nonbasic_vars:]
            flag = False

            for iter in range(max_iter):
                print('Iteration: {0}, optimal solution obtained at x = {1}'.format(iter + 1, v))
                h = np.array([hx.subs(zip(xvars, v)) for hx in hxs])
                if all([abs(h_i) < eps_2 for h_i in h]):  # Check if candidate satisfies all constraints
                    if fx.subs(zip(xvars, xcurr)) <= fx.subs(zip(xvars, v)):
                        alpha = alpha * gamma
                        break
                    else:
                        xcurr = v  # Obtained a candidate better than the current optimal solution
                        flag = True
                        break

                # Step 3(b)

                delta_h_v = np.array([[dh.subs(zip(xvars, v)) for dh in dhx] for dhx in dhxs])
                J_inv_v = np.linalg.inv(np.array([dhx[nonbasic_vars:] for dhx in delta_h_v], dtype=float))
                v_next_cap = v_cap - J_inv_v.dot(h)

                # Step 3(c)

                if abs(np.linalg.norm(np.array(v_cap - v_next_cap, dtype=float), 1)) > eps_3:
                    v_cap = v_next_cap
                    v = np.concatenate((v_bar, v_cap))
                else:
                    v_cap = v_next_cap
                    v = np.concatenate((v_bar, v_cap))
                    h = np.array([hx.subs(zip(xvars, v)) for hx in hxs])
                    if all([abs(h_i) < eps_2 for h_i in h]):

                        # Step 3(d)

                        if fx.subs(zip(xvars, xcurr)) <= fx.subs(zip(xvars, v)):
                            alpha = alpha * gamma  # Search for lower values of alpha
                            break
                        else:
                            xcurr = v
                            flag = True
                            break
                    else:
                        alpha = alpha * gamma
                        break

            if flag == True:
                break

    print('\n\nFinal solution obtained is: {0}'.format(xcurr))
    print('Value of the function at this point: {0}\n'.format(fx.subs(zip(xvars, xcurr))))

    plt.plot(opt_sols, 'ro')  # Plot the solutions obtained after every iteration
    plt.show()


if __name__ == '__main__':
    generalized_reduced_gradient()

