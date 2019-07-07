import numpy as np


class VarMat(object):
    """
    Constructor for objects to reference a common matrix for linear solver

    save indicies that each object uses.

    constructors should obtain a reference instead of directly doing

    usage:
        vm = Varmat()
        cobj = Constructor()

        cobj.get_ref(vm)
        # -- or --
        vm.add_ref(cobj)

        # something like this:
        cobj.set_M(indicies, value) -> calls vm[indicies] = value



    """
    def __init__(self, shape=None):
        self._M = None
        self.refs = {}  # { object_id : reference_block_size }
        if shape:
            self._M = np.zeros(shape)

    def __len__(self):
        """ dimension 0 of matrix """
        return

    def __getitem__(self, item):
        return

    def __setitem__(self, key, value):
        return

    def add_ref(self, obj):
        return


