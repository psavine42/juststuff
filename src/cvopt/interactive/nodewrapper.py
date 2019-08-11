import cvxpy as cvx


class FormulationWrapper(object):
    def __init__(self, klass):
        self._base = klass()
        self._in_slots = {}
        self._out_slots = {}

    def __setitem__(self, key, value):
        self._in_slots[key] = value

    def run(self):
        return

    def inputs(self):
        return

    def outputs(self):
        return


class ParameterWrapper(object):
    pass


class VariableWrapper(object):
    pass


class ObjectiveWrapper(object):
    pass


class ProblemWrapper(object):
    pass


_klass_dict = {

}
