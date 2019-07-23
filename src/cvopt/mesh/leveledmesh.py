from .mesh2d import Mesh2d, MeshBase
from cvxpy.utilities.performance_utils import compute_once, lazyprop


class StackedMesh(MeshBase):
    def __init__(self, *meshes):
        """ store """
        self._meshs = list(meshes)

    @lazyprop
    def edges(self):
        pass

    @lazyprop
    def boundary(self):
        pass

    @lazyprop
    def half_edges(self):
        pass

    @lazyprop
    def faces(self):
        pass

    @lazyprop
    def vertices(self):
        pass

    def __getitem__(self, item):
        return self._meshs[item]



