from .mesh2d import Mesh2d
import numpy as np


class MatrixFactory(object):
    def __init__(self, mesh: Mesh2d):
        """ create and paramater-ify incidents matricies of mesh domain """
        self._mesh = mesh

    @property
    def space(self):
        return self._mesh

    def adj_face_to_face(self):
        """ faces adjacency matrix ... if M[i, j] = 1, faces are adjacent"""
        N = len(self.space.faces)
        M = np.zeros((N, N), dtype=int)
        for k, vs in self.space.faces.to_faces.items():
            M[k, list(vs)] = 1
        return M

    @classmethod
    def face_adj(cls, space):
        return cls(space).adj_face_to_face()



