from abc import ABC, abstractmethod
from .interfaces import Layout
from .objectives import *
import src.layout_ops as lops
import numpy as np
from .actions.basic import *


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


class ArchiBuildEnv(EnvBase):
    def __init__(self, problem):
        self._problem = problem
        self._initializer = None
        self._solver = None

    def expand(self):
        pass

    def transition_model(self, layout):
        pass



class GridBasedEnv(EnvBase):
    def __init__(self, size=(200, 200)):
        self._size = size

    def encode(self, layout):
        mat = np.zeros(self._size)



class ModularLayoutEnv(EnvBase):
    def __init__(self):
        pass

    def cost(self, layout):
        pass

# import core.env
# class RLEnv(core.env.)