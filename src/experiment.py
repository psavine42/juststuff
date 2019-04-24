from src.envs import MetroLayoutEnv
from abc import ABC, abstractmethod


class _Expirement(ABC):
    pass


class SimpleMH(_Expirement):
    def __init__(self,
                 env,
                 problem,
                 num_iter=100,
                 cost_fn=None,
                 model=None,
                 priors=[],
                 initializer=None):
        self._priors = priors
        #
        self._problem = problem
        self._env = env
        self._initializer = initializer
        self._model = model
        self._num_iter = num_iter
        self._cost_fn = cost_fn

    @property
    def model(self):
        return self._model

    @property
    def problem(self):
        return self._problem

    @property
    def cost(self):
        return self._cost_fn

    def run(self):
        init_state = self._initializer()

        accepted = [init_state]
        rejected = []
        prev = init_state
        for i in range(self._num_iter):
            X = self.model(prev)
            if X == accepted[-1]:
                accepted.append(X)
                prev = X
            else:
                rejected.append(X)

        # accepted, _ = self._model(init_state)
        # plot the state history
        print(self.cost(init_state), self.cost(accepted[-1]))
        # print(accepted[-1])
        return accepted

    def __repr__(self):
        st = ''
        st += str(self._env)
        st += str(self._env)



