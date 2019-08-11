from cvxpy import Problem
import dccp
import dmcp


def describe_problem(problem: Problem):
    """
    DMCP https://github.com/cvxgrp/dmcp
    dccp
    """
    st = 'Curvature {}'.format(problem.objective.expr.curvature)
    st += '\nis disciplined quasiconvex    {}'.format(problem.is_dqcp())
    st += '\nis disciplined geometric      {}'.format(problem.is_dgp())
    st += '\nis disciplined quadratic      {}'.format(problem.is_qp())
    st += '\nis disciplined convex         {}'.format(problem.is_dcp())
    st += '\nis disciplined concave-convex {}'.format(dccp.is_dccp(problem))
    st += '\nis disciplined multi-convex   {}'.format(dmcp.is_dmcp(problem))
    # todo
    # SQP sequential quadratic program
    # SCP seperable convex program
    #
    return st


class FPProbllem(object):
    def __init__(self):
        self._problem = None
        self._constraints = None
        self.G = None
        self._formulations = []
        self._placements = []
        self._meta = {}

    def action_eliminators(self):
        return []

    def print(self, problem):
        print('Problem:----------')
        print('----------')
        print(problem.solution.status)
        print(problem.solution.opt_val)
        print(problem.solution.attr)

    def make(self,
             verbose=False,
             obj_args={},
             const_args={}):
        constraints = self.own_constraints(**const_args)

        objective = self.objective(**obj_args)
        self._problem = Problem(objective, constraints)
        if verbose is True:
            print('Constraints')
            print('---------------------------')
            for c in constraints:
                print(c)
            print('Objective')
            print('---------------------------')
            print(self._problem.objective)
            print('problem ready')
        return self._problem

    def run(self, obj_args={},
            const_args={},
            solve_args={},
            verbose=False,
            show=True,
            save=None):
        if self._problem is None:
            self.make(verbose=verbose,
                      const_args=const_args,
                      obj_args=obj_args)
        self._problem.solve(verbose=verbose, **solve_args)
        if self._problem.solution.status == 'infeasible':
            print(self._problem._solver_stats.__dict__)
            for x in self._problem.constraints:
                print(x)

        print('solution created')
        print(self._problem.solution)
        #if show:
        #    self.display(self._problem, save=save)
        return self.solution

    def solve(self, **kwargs):
        return self.run(**kwargs)

    @property
    def formulations(self):
        return self._formulations

    @property
    def placements(self):
        return self._placements

    # implement in supercalss ----------------
    @property
    def solution(self):
        return

    @property
    def domain(self):
        return self.G

    @property
    def meta(self):
        return self._meta

    @property
    def problem(self):
        return self._problem

    def _anchor(self, *args):
        return

    def _pre_compute_placements(self):
        return

    def own_constraints(self, **kwargs):
        raise NotImplemented('not implemented in base class')

    def objective(self, **kwargs):
        raise NotImplemented('not implemented in base class')

    def display(self, problem, **kwargs):
        return

