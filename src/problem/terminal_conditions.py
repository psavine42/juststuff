import numpy as np


class TerminalCondition(object):
    def __init__(self, tol=0.01):
        self._tol = tol
        self.steps = 0

    def reset(self):
        self.steps = 0

    def inc(self):
        self.steps += 1

    def __call__(self, reward, legal):
        """ return tuple( done, solved ) """
        if self._tol > abs(1 - reward):
            return True, True
        return False, False


class SameScore(TerminalCondition):
    """ if 'limit' steps happened without change in score """
    def __init__(self, limit=4):
        TerminalCondition.__init__(self)
        self._limit = limit

        self.steps = 0
        self.prev = -np.inf

    def __str__(self):
        return '<{}>:limit:{}'.format(self.__class__.__name__ , self._limit)

    def reset(self):
        self.steps = 0
        self.prev = -np.inf

    def __call__(self, reward, legal):
        """ return ( done, solved ) """
        if self._tol > abs(1 - reward):
            return True, True

        elif self.prev == round(reward, 5):
            if self.steps < self._limit:
                self.steps += 1
                self.prev = round(reward, 5)
                return False, False
            return True, False

        self.steps = 0
        self.prev = round(reward, 5)
        return False, False


class NoImprovement(TerminalCondition):
    """ if 'limit' steps happened without improvement """
    def __init__(self, limit=4):
        TerminalCondition.__init__(self)
        self._limit = limit

        self.steps = 0
        self.best = -np.inf

    def __str__(self):
        return '<{}>:limit:{}'.format(self.__class__.__name__, self._limit)

    def reset(self):
        self.steps = 0
        self.best = -np.inf

    def __call__(self, reward, legal):
        if self._tol > abs(1 - reward):
            return True, True

        elif self.best >= round(reward, 6):
            # if no improvement
            if self.steps < self._limit:
                self.steps += 1
                return False, False
            return True, False

        self.steps = 0
        self.best = round(reward, 6)
        return False, False

