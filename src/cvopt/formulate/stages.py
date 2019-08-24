from .formulations import Formulation, sum_objectives
from .cont_base import FormulationR2
from .fp_disc import FormulationDisc
from ..problem import FPProbllem
from cvxpy import Variable, Parameter, Minimize
import src.cvopt.utils as du


class Stage(FPProbllem):
    def __init__(self, inputs, outputs=None, forms=None, **kwargs):
        FPProbllem.__init__(self)
        self._inputs = None
        self._outputs = None

        if isinstance(inputs, (list, tuple)):
            self._inputs = inputs
        else:
            self._inputs = [inputs]

        if outputs is None:
            self._outputs = self._inputs
        elif isinstance(outputs, (list, tuple)):
            self._outputs = outputs
        else:
            self._outputs = [outputs]
        if forms is not None:
            self._formulations.extend(forms)

    @property
    def is_solved(self):
        if self._problem is None:
            return False
        elif self._problem.solution is None:
            return False
        elif self._problem.solution.status == 'optimal':
            return True
        return False

    @property
    def serializable(self):
        return self._formulations + self._inputs

    def solve(self, verbose=False, solve_args={}):
        method = None
        for x in self._formulations:
            nargs = x.solver_args
            if method is not None and nargs.get('method', None) != method:
                print('SOLVER WARNING - METHOD OVERRIDE')
            elif nargs.get('method', None) is not None:
                method = nargs['method']

        if method is not None:
            solve_args['method'] = method
        print(solve_args)
        FPProbllem.solve(self, verbose=verbose, solve_args=solve_args)

    @property
    def outputs(self):
        """ once the initial problem is solved """
        return self._outputs

    @property
    def solution(self):
        return self._outputs

    def own_constraints(self, **kwargs):
        C = []
        for x in self._formulations:
            C += x.constraints()
        return C

    def objective(self, **kwargs):
        return sum_objectives(self._formulations)

    def display(self, save=None, extents=None, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, figsize=(7, 7))
        for f in self._formulations:
            du.draw_form(f, ax, **kwargs)
        for f in self.outputs:
            du.draw_form(f, ax, **kwargs)
        du.finalize(ax=ax, save=save, extents=extents)




