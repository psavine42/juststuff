from .formulations import Formulation
from src.cvopt.shape import R2
from cvxpy import Variable, Minimize
from cvxpy.expressions.expression import Expression


class FormulationR2(Formulation):
    DOMAIN = {'R2'}

    def __init__(self, inputs, domain=None, **kwargs):
        """
        Formulations in rn do not require an explicit domain per-say,
        as they do not need to reference the discretizations of it

        """
        if domain is None:
            domain = R2()
        Formulation.__init__(self, domain, **kwargs)
        self._inputs = inputs

    @property
    def inputs(self):
        """ todo rename this -need a distinction between lists of stuff and formulations
            todo also, need to return some datastrucre,

        """
        return self._inputs

    def display(self):
        """ returns geometry in r2 """
        return {
            'points':   [],
            'segments': [],
            'polygons': [],
            'boxes':    [],
            'spheres':  []
        }

    def as_constraint(self, **kwargs):
        return []

    def as_objective(self, **kwargs):
        return None


class ConstrLambda(FormulationR2):
    META = {'constraint': True, 'objective': False}

    def as_constraint(self):
        if isinstance(self._inputs, (tuple, list)):
            return self._inputs
        elif isinstance(self._inputs, Expression):
            return [self._inputs]
        return []


class NumericBound(FormulationR2):
    META = {'constraint': True, 'objective': False}

    def __init__(self, child_var, low=None, high=None, domain=None, **kwargs):
        if domain is None:
            domain = R2()
        assert isinstance(child_var, Variable)
        FormulationR2.__init__(self, domain, [], **kwargs)
        self._inputs = child_var if child_var else []
        self._high = high
        self._low = low

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._inputs

    def as_constraint(self, *args):
        C = []
        if self._high is not None:
            C += [self._inputs <= self._high]
        if self._low is not None:
            C += [self._inputs >= self._low]
        return C

    def as_objective(self, **kwargs):
        return None

    def describe(self):
        return
