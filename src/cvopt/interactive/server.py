from .nodewrapper import *


class Venv(object):
    """ virtual enviornment for a session """
    def __init__(self):
        self._nodes = []
        self._links = []

    def connect(self, from_, to, sfrom, sto):
        """ setting a value in a node class """
        nfrom = self._nodes[from_]
        nto = self._nodes[to]
        output_obj = nfrom.outputs[sfrom]
        nto.inputs[sto] = output_obj
        return 'ok'

    def create(self, cls_id):
        """ initialize a class """
        cls = klass_dict.get(cls_id, None)
        new = cls()
        self._nodes.append(new)
        return 'ok'

    def destroy(self, uid):

        return

    def disconnect(self, uid):
        return

    def cmd_solve(self):
        """ solve the problem,
            return variable values to be populated """
        return {}


