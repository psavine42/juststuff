from .formulations import Formulation
from cvxpy import Variable, Parameter, Minimize


class Stage(Formulation):
    def __init__(self, domain, **kwargs):
        Formulation.__init__(self, domain, **kwargs)

    def _pull_pred_vars(self):
        for v in self.vars:
            yield v
        for input in self.inputs:
            for vars_i in input._gather_vars():
                yield vars_i

    def _gather_dict(self):
        for v in self._pull_pred_vars():
            if v.name in self._in_dict and self._in_dict[v.name] is None:
                self._in_dict[v.name] = v

