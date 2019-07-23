from .formulations import Formulation
from src.cvopt.shape import R2
from cvxpy import Variable

class FormulationR2(Formulation):
    DOMAIN = {'R2'}

    def __init__(self, domain, children, **kwargs):
        """
        Formulations in rn do not require an explicit domain per-say,
        as they do not need to reference the discretizations of it

        """
        Formulation.__init__(self, domain, **kwargs)
        self._inputs = children if children else []

    def display(self):
        """ returns geometry in r2 """
        return {
            'points':   [],
            'segments': [],
            'polygons': [],
            'boxes':    [],
            'spheres':  []
        }

    def as_objective(self, **kwargs):
        return None

    def as_constraint(self, *args):
        return None


class NumericBound(FormulationR2):
    META = {'constraint': True, 'objective': False}

    def __init__(self, child_vars, low=None, high=None, domain=None, **kwargs):
        if domain is None:
            domain = R2()
        assert isinstance(child_vars, Variable)
        FormulationR2.__init__(self, domain, [], **kwargs)
        self._inputs = child_vars if child_vars else []
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
