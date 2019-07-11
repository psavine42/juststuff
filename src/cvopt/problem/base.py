from cvxpy import Problem


class FPProbllem(object):
    def __init__(self):
        self._problem = None
        self._formulations = []
        self._placements = []

    def own_constraints(self, **kwargs):
        raise NotImplemented('not implemented in base class')

    def display(self, problem, **kwargs):
        return

    def action_eliminators(self):
        return []

    def objective(self, **kwargs):
        raise NotImplemented('not implemented in base class')

    def print(self, problem):
        print('Problem:----------')
        # print(problem)
        print('----------')
        print(problem.solution.status)
        print(problem.solution.opt_val)
        print(problem.solution.attr)

    @property
    def solution(self):
        return

    def run(self, obj_args={},
            const_args={},
            verbose=False,
            show=True,
            save=None):
        constraints = self.own_constraints(**const_args)
        if verbose is True:
            for c in constraints:
                print(c)
        objective = self.objective(**obj_args)
        self._problem = Problem(objective, constraints)
        print(self._problem.objective)
        print('problem ready')
        self._problem.solve(verbose=verbose)
        print('solution created')
        print(self._problem.solution)
        if show:
            self.display(self._problem, save=save, constraints=constraints)
        return self.solution

    def solve(self, **kwargs):
        return self.run(**kwargs)

    @property
    def formulations(self):
        return self._formulations

    @property
    def placements(self):
        return self._placements


