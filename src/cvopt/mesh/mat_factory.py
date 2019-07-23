from .mesh2d import Mesh2d


class MatrixFactory(object):
    def __init__(self, mesh: Mesh2d):
        """ create and paramater-ify incidents matricies of mesh domain """
        self._mesh = mesh



