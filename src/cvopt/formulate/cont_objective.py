import numpy as np
import cvxpy as cvx
from cvxpy import Variable, Minimize, Maximize, Parameter
from .cont_base import FormulationR2, NumericBound
import cvxpy.lin_ops.lin_utils as lu
import operator


class ObjectiveR2(FormulationR2):
    META = {'constraint': True, 'objective': True}

    def __init__(self, inputs, weight=None, **kwargs):
        """

        :param inputs:
        :param w: optional weight matrix
        :param kwargs:
        """
        kwargs['is_objective'] = True
        FormulationR2.__init__(self, inputs, **kwargs)
        shape = self.w_shape
        self._w = None
        if weight is None:
            self._w = np.ones(shape)
        elif isinstance(weight, (list, tuple, np.ndarray)):
            w = np.asarray(weight)
            assert w.shape[0] == shape, \
                'required, {} got {}'.format(w.shape, shape)
            self._w = w

    @property
    def w_shape(self):
        return len(self.inputs)

    @property
    def obj_klass(self):
        if self._obj_type in (Minimize, Maximize):
            return self._obj_type
        else:
            return Minimize


class PointDistObj(ObjectiveR2):
    """
    based on Novel Convex Optimization Approaches for VLSI Floorplanning  2008 SDP
    for a floorplanning problem with constraints on the area of cells,
    minimize distances between

    Arguments:
        inputs: Boxlist

    """

    def __init__(self, inputs, weight=None, **kwargs):
        ObjectiveR2.__init__(self, inputs, weight=weight, **kwargs)

    @property
    def w_shape(self):

        # triu = np.triu_indices(len(self.inputs), 1)
        # print(len(self.inputs))
        shp = np.triu_indices(len(self.inputs), 1)[0].shape[0]
        return shp

    def as_constraint(self):
        """
            C = [
                U <= X[tj] + M * bx,
                X[ti] <= U + M * (1 - bx),

                # -- AND --
                U >= X[ti] + M * bx,
                U >= X[tj] + M * (1 - bx),

                # U >= X[ti] - X[tj] * bx,
                # U >= X[tj] - X[ti],
                # V >= Y[ti] - Y[tj],
                # V >= Y[tj] - Y[ti]
            ]
            C = [
                0 <= -U + X[tj] + bx_eq_1,
                0 <=  U - X[ti] + bx_eq_1,

                # -- AND --
                0 <= -U + X[ti] + bx_eq_0,
                0 <=  U - X[tj] + bx_eq_0,

                # -----------------------
                0 <= -V + Y[tj] + by_eq_1,
                0 <=  V - Y[ti] + by_eq_1,

                # -- AND --
                0 <= -V + Y[ti] + by_eq_0,
                0 <=  V - Y[tj] + by_eq_0,
            ]
        """
        N = len(self.inputs)
        M = 199
        ti, tj = [x.tolist() for x in np.triu_indices(N, 1)]
        C = []
        X, Y, W, H = self.inputs.vars
        U = Variable(shape=len(tj), name='U.{}'.format(self.name))
        V = Variable(shape=len(tj), name='V.{}'.format(self.name))
        if self.obj_klass == Maximize:
            bx = Variable(shape=len(tj), boolean=True)
            by = Variable(shape=len(tj), boolean=True)

            bx_eq_1, bx_eq_0 = M * bx, M * (1 - bx)
            by_eq_1, by_eq_0 = M * by, M * (1 - by)
            C = [
                # pos, 0
                # neg, 1
                0 <= X[ti] - X[tj] + bx_eq_1,
                0 <= X[tj] - X[ti] + bx_eq_0,
                0 <= U,
                U <= X[ti] - X[tj] + bx_eq_1,
                U <= X[tj] - X[ti] + bx_eq_0,

                0 <= Y[ti] - Y[tj] + by_eq_1,
                0 <= Y[tj] - Y[ti] + by_eq_0,
                0 <= V,
                V <= Y[ti] - Y[tj] + by_eq_1,
                V <= Y[tj] - Y[ti] + by_eq_0,
            ]
            self.uv = (U, V)
            expr = U + V
            self._obj = Maximize(cvx.sum(cvx.multiply(self._w, expr)))
            # self._obj = Maximize(cvx.sum(cvx.sqrt(expr)))
        else:
            C = [
                # linearized absolute value constraints - i guess its faster
                # | x_i - x_j |  | y_i - y_j |
                U >= X[ti] - X[tj],
                U >= X[tj] - X[ti],
                V >= Y[ti] - Y[tj],
                V >= Y[tj] - Y[ti]
            ]
            self._obj = Minimize(cvx.sum(cvx.multiply(self._w, (U + V))))
        return C

    def as_objective(self, **kwargs):
        return self._obj

    def describe(self):
        return


class DistToSet(ObjectiveR2):
    def __init__(self, inputs, cvx_set, **kwargs):
        ObjectiveR2.__init__(self, inputs, **kwargs)
        shp = self.inputs.point_vars.shape
        self._cvx_set = Parameter(shape=shp, value=np.tile(cvx_set, (shp[0], 1)))

    def as_objective(self, **kwargs):
        XY = self.inputs.point_vars
        if self.obj_klass == Maximize:
            return Maximize(cvx.sum(cvx.sqrt(XY - self._cvx_set)))
        else:
            return Minimize(cvx.sum(cvx.norm(XY - self._cvx_set, 2, axis=1)))


class MaxPerimeter(ObjectiveR2):
    def as_objective(self, **kwargs):
        X, Y, W, H = self.inputs.vars
        return cvx.Maximize(cvx.sum(W+H))
