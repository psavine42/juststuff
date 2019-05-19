
from .layout import *
from .objectives import *
import numpy as np
from .actions.basic import *
import math
import src.utils as utils
from random import randint
from src.problem.base import Problem, ProgramEntity
from src.problem.terminal_conditions import *
import pprint

class EnvBase(ABC):
    @abstractmethod
    def get_actions(self, *args):
        pass


class ConstraintEnv(EnvBase):
    def __init__(self, costfn):
        self._cost_fn = costfn
        self._state_cls = None

    def random_state(self):
        return


class WrappedAction(object):
    pass


class MetroLayoutEnv(EnvBase):
    def __init__(self,  weights={}, swap_prob=0.2):
        self._weights = weights  # dict of weights for cost_fn
        self._swap_prob = swap_prob
        self._hist = []

    def get_actions(self, layout: Layout):
        return

    def apply_action(self, layout: Layout, action=None) -> Layout:
        if action is None:
            pass
        return layout

    def transition_model(self, layout):
        new_layout = None
        while not new_layout:
            if np.random.random() < self._swap_prob:
                new_layout = swap_rooms(layout)
            else:
                param = slide_wall_params(layout)
                new_layout = slide_wall(layout, **param)

            if not new_layout.is_valid:
                new_layout = None
        return new_layout


class MockEnv(object):
    def __init__(self, problem, init):
        self.initialize = init
        self.problem = problem
        self.objective = ConstraintsHeur(
            problem, wmap=dict(AreaConstraint=5, FootPrintConstraint=0)
        )
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
            normed = self.objective.reward(layout)
            if (normed - 1.) > 1e-2:
                return normed
            return normed - 1
            # return self.loss.reward(layout)
            # return (3 -  self.loss(layout)) / 3
        except:
            return -1

    def reward2(self, layout, encode=False, **kwargs):
        """(-1, 1)"""
        if layout is None:
            if encode is True:
                return -.5, np.zeros(self.objective.size)
            return -.5
        return self.objective.reward(layout, encode=encode)

    def _transition(self, layout, action, room_index=None, item_index=None):
        size = self.num_actions // 2
        if action == size:
            # assumes middle action is 0
            # save some time by not computing this
            return layout

        sign = 1 if action == size else -1
        mag = -1 * action if action < size else action - size

        room_name = layout[room_index].name
        params = dict(room=room_name, item=item_index, mag=mag, sign=sign)
        next_layout = self.action.forward(layout, **params)
        return next_layout

    def step_enc(self, layout, action, room_index=None, item_index=None):
        """ case 1 - output is right or left or Noop (0) HA !  [0, 1, 3]"""
        next_layout = self._transition(layout, action, room_index, item_index)

        # would like to not do this either - if the action is to do nothing,
        # there should be no reward, otherwise it will sit on 0 actions and
        # keep getting rewarded ...
        reward, mat = self.reward2(next_layout, encode=True)
        if action == self.num_actions // 2:
            reward = -0.01
        num_sat = len(np.where(mat == 1)[0])
        fail = True if next_layout is None else False
        return dict(layout=next_layout,
                    reward=reward,
                    fail=fail,
                    feats=mat,
                    num_sat=num_sat)

    def step_norm(self, layout, action, room_index=None, item_index=None):
        """ case 1 - output is right or left or Noop (0) HA !  [0, 1, 3]"""
        next_layout = self._transition(layout, action, room_index, item_index)
        reward = self.reward2(next_layout)
        fail = True if next_layout is None else False
        return dict(layout=next_layout, reward=reward, fail=fail)


class DiscreteEnv:
    def __init__(self,
                 problem,
                 objective,
                 state_size,
                 actor=None,
                 random_init=False,
                 random_objective=False,
                 encode_step=True,
                 invalid_penalty=0.5):
        # params
        self._size = state_size if isinstance(state_size, (list, tuple)) else (state_size, state_size)
        self._encode = encode_step
        self._invalid_action_penalty = invalid_penalty
        self._random_objective = random_objective
        self._random_init = random_init
        # objects
        self._problem = problem
        self._action = actor
        self._objective = objective

    @property
    def objective(self):
        return self._objective

    @property
    def problem(self):
        return self._problem

    @property
    def action_size(self):
        return self._action.num_actions

    @property
    def state_shape(self):
        return self._size

    @property
    def goals(self):
        return self._objective.goal

    def initialize(self, origin=(0, 0)):
        if self._random_init:
            origin = (randint(0, self._size[0]-1), randint(0, self._size[1]-1))

        if self._random_objective:
            # objective is randomized
            tg = self._objective._target
            sg = self._objective.sum_goals

        layout = CellComplexLayout(self._problem, size=self._size, origin=origin)
        reward, code = self._objective.reward(layout, True)
        # set the initial state on the action
        self._action.set_prev(origin)
        return layout, dict(prev_state=None,
                            state=layout.state,
                            action=None,
                            fail=False,
                            solved=False,
                            reward=reward,
                            feats=self.goals)

    def step(self, layout, action):
        prev_state = layout.state
        layout, legal = self._action.forward(layout, action)
        reward, code = self._objective.reward(layout, encode=True, avg=True)
        if legal is False:
            # if invalid
            reward -= self._invalid_action_penalty

        solved = True if 0.01 > abs(1 - reward) else False

        if reward == 1 or layout.num_rooms == self._objective.num_goals:
            done = True
        elif len(layout._G) < self._objective.sum_goals:
            done = True
            reward -= 1.
        else:
            done = False

        if solved is True:
            print('solved')
            # utils.layout_disc_to_viz(layout)
        return dict(prev_state=prev_state,
                    state=layout.state,
                    done=done,
                    action=action,
                    solved=solved,
                    fail=legal,
                    feats=self.goals,
                    reward=reward)


class DiscreteEnv2(DiscreteEnv):
    """
    Action [index, [x1, y1, x2, t2]]
    """
    def __init__(self, *args,
                 num_spaces=5,
                 random_reset=True,
                 inst_args=None,
                 terminal=TerminalCondition(),
                 state_cls=StackedRooms,
                 problem_gen=generate_constraint_dict,
                 unsolved_problem_reward=None,
                 objective_args={},
                 **kwargs):
        DiscreteEnv.__init__(self, *args, **kwargs)
        self.num_spaces = num_spaces
        self._inst_args = inst_args
        self._state_cls = state_cls
        self._objective_args = objective_args
        self._problem_gen = problem_gen

        self._target_code = None
        self._instance = None

        # self._max_repeat = 4
        self._unsolved_problem_reward = unsolved_problem_reward
        self._terminal_fn = terminal
        self._random_reset = random_reset
        self.__prev_data = self.initialize()

    def dataset(self, batch_size=1):
        pass

    def initialize(self):
        # if self._instance is not None
        self._problem = Problem()
        for i in range(self.num_spaces):
            self._problem.add_program(ProgramEntity(i, 'room'))

        problem_dict = self._problem_gen(num_spaces=self.num_spaces,
                                         x=self._size[0],
                                         y=self._size[1])
        self._problem.footprint = generate_footprint_constr(self._size[0])
        self._objective = DiscProbDim(self._problem, problem_dict, **self._objective_args)

        self._instance = self._state_cls(self._problem,
                                         size=self._size,
                                         problem_dict=problem_dict,
                                         **self._inst_args)
        self._target_code = self._objective.to_input()
        reward, code = self._objective.reward(self._instance, True)

        assert code.shape == self._target_code.shape, str(code.shape) + ' ' + str(self._target_code.shape)
        self._terminal_fn.reset()
        self.__prev_data = dict(prev_state=None,
                                action=None,
                                state=self._instance.to_input(),
                                reward=reward,  # zeros
                                done=False,
                                legal=True,
                                solved=False,
                                target_code=self._target_code,
                                feats=code ) # np.stack([code, self._target_code]))
        return self.__prev_data

    def to_image(self, input_state):
        return self._instance.to_image(input_state)

    @property
    def input_state_size(self):
        return self._instance.input_state_size

    @property
    def state(self):
        return self._instance.state

    def step(self, action, **kwargs):
        reward, moved = True, -1
        # legal = action[0] < len(self._problem)
        prev_state = self._instance.to_input()
        moved, legal = self._instance.add_step(action)

        if moved is True:
            reward, code = self._objective.reward(self._instance,
                                                  action=action[0],
                                                  encode=True, avg=True)

        elif moved is False:
            # if no movement can be made, then ...
            self._terminal_fn.inc()
            data = self.__prev_data.copy()
            data['reward'] = -1
            return self.__prev_data

        # enviornment specific terminal conditions
        done, solved = self._terminal_fn(reward, legal)

        # goose the reward
        # if not legal:
        #    reward = max(-1, reward - self._invalid_action_penalty)
        if done and not solved:
            reward = self._unsolved_problem_reward # -1

        self.__prev_data = dict(prev_state=prev_state,
                                action=action,
                                state=self._instance.to_input(),
                                reward=reward,
                                done=done,
                                solved=solved,
                                legal=legal,
                                target_code=self._target_code,
                                feats=code)
        return self.__prev_data


class ModularLayoutEnv(EnvBase):
    def __init__(self):
        pass

    def cost(self, layout):
        pass

# import core.env
# class RLEnv(core.env.)