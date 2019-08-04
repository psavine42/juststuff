from .formulations import Formulation
import cvxpy as cvx
from .cont_base import FormulationR2
from cvxpy import Variable, Minimize\
    , Constant, Parameter
import numpy as np


def segment_index(n):
    # res = np.asarray([[i, i - 1] for i in range(1, n)])
    res = np.asarray([[i - 1, i] for i in range(1, n)])
    return res


class PathFormulationR2(FormulationR2):
    """
    a single p-link path on Affine sets from source to target

    -----------------------
    If C is an affine set, x1 , ... , xk ∈ C,
    and θ1 + ... + θk = 1,
    then the point θ1 * x1 + ... + θk*xk also belongs to C.
    -----------------------
    """
    def __init__(self, inputs, tgt, src=None, **kwargs):
        """
        Convert a PointList to a segment list
        """
        from src.cvopt.formulate.input_structs import PointList
        if isinstance(inputs, PointList):
            pass
        elif isinstance(inputs, int):
            inputs = PointList(inputs)
        FormulationR2.__init__(self, inputs, **kwargs)

        self._tgt_x = Parameter(value=tgt[0])
        self._tgt_y = Parameter(value=tgt[1])
        # source may not be an expression
        if src is None:
            self._src_x = Variable(pos=True, name='path_start_X')
            self._src_y = Variable(pos=True, name='path_start_Y')
        else:
            self._src_x = Parameter(value=src[0])
            self._src_y = Parameter(value=src[1])

    @property
    def vars(self):
        X, Y = self.inputs.X, self.inputs.Y
        nx = cvx.hstack([self._src_x] + X + [self._tgt_x])
        ny = cvx.hstack([self._src_y] + Y + [self._tgt_y])
        return nx, ny

    @property
    def segments(self):
        ixs = segment_index(len(self))
        s = []
        X = [self._src_x] + self.inputs.X + [self._tgt_x]
        Y = [self._src_y] + self.inputs.Y + [self._tgt_y]
        for i in range(len(self) -1):
            s.append(cvx.hstack([X[i-1], Y[i-1], X[i], Y[i]]))
        return s

    def __len__(self):
        return self.vars[0].shape[0]

    def display(self):
        display = FormulationR2.display(self)
        X, Y = [x.value for x in self.vars]
        cnt = 0
        ix = segment_index(len(self)).T.tolist()
        for i, j in zip(ix[0], ix[1]):
            datum = dict(x0=X[i], y0=Y[i], x1=X[j], y1=Y[j],
                         index=cnt)
            print(i, j, datum)
            display['segments'].append(datum)
            cnt += 1
        for i in range(len(self)):
            datum = dict(x0=X[i], y0=Y[i], index=i)
            display['points'].append(datum)
            cnt += 1
        return display


class ShortestPathCont(FormulationR2):
    META = {'constraint': False, 'objective': True}

    def __init__(self, inputs, **kwargs):
        """
        segments are represented as

        y = x2 + θ(x1 - x2)
        ------
            if θ = 1
            y = x2 - x1

        Non-neg homogenous cone
        θ1*x1 + θ2*x2 ∈ C
        if
        θ1, θ2 ≥ 0

        """
        assert isinstance(inputs, PathFormulationR2)
        FormulationR2.__init__(self, inputs, **kwargs)

    def as_constraint(self, **kwargs):
        """ todo = distance(origin, X_i,j) < distance(origin, X_i+1,j+1 )"""
        return

    def as_objective(self, **kwargs):
        """
        minimize norm2 of p-link path
        """
        X, Y = self.inputs.vars
        iseg = segment_index(len(self.inputs))
        o = Minimize(
            cvx.norm1(X[iseg[:, 0]] - X[iseg[:, 1]]) +
            cvx.norm1(Y[iseg[:, 0]] - Y[iseg[:, 1]])
        )
        return o


class Branch(FormulationR2):
    META = {'constraint': False, 'objective': True}

    def __init__(self, inputs, target, branch=None, bsize=None,  **kwargs):
        """
        Assign segment s of inputs to be a branch.

        Transforms input list to a group of input lists.
        Each list should be
        """
        FormulationR2.__init__(self, inputs, **kwargs)
        self._size = bsize if bsize else len(inputs)
        self._target = target
        self._branch = branch
        if branch is None:
            from src.cvopt.formulate.input_structs import PointList
            self._branch = PathFormulationR2(PointList(self._size), self._target)

    @property
    def vars(self):
        return

    def as_constraint(self, **kwargs):
        # branchix = Variable(shape=len(self.inputs) - 2, boolean=True,
        #                     name='branch.{}'.format(self.name))
        # iseg = segment_index(len(self.inputs))
        # X, Y = self._branch.vars
        # start_x, start_y = X[0], Y[0]
        # C = [
        #     cvx.sum(branchix) <= 1,
        #
        #
        # ]
        return []

    def as_objective(self, **kwargs):
        """
        main[selected] - branch[0]

        selection can be written as Minimize(minimum(main[:] - branch[0]))

        """
        Xmain, Ymain = self.inputs.vars
        Xb, Yb = self._branch.vars
        start_x, start_y = Xb[0], Yb[0]
        o = Minimize(
            cvx.sum_smallest(
                cvx.norm1(Xmain - start_x) +
                cvx.norm1(Ymain - start_y), 1)
        )
        return o


class SegmentsHV(FormulationR2):
    META = {'constraint': True, 'objective': False}

    def __init__(self, inputs, N=100, **kwargs):
        """
        vertical or horizontal segments only
        """
        assert isinstance(inputs, PathFormulationR2)
        FormulationR2.__init__(self, inputs, **kwargs)
        self.N = N

    def as_constraint(self, **kwargs):
        """ MIP formulation - either
            Xi - Xj < 0 or
            Yi - Yj < 0
        """
        X, Y = self.inputs.vars
        iseg = segment_index(len(self.inputs))

        vars1 = Variable(shape=len(self.inputs)-1, boolean=True,
                         name='X_seg.{}'.format(self.name))
        mag_x = cvx.abs(X[iseg[:, 0]] - X[iseg[:, 1]])
        mag_y = cvx.abs(Y[iseg[:, 0]] - Y[iseg[:, 1]])
        C = [
            # chose minimum one of indicators
            mag_x <= self.N * vars1,
            mag_y <= self.N * (1 - vars1)
        ]
        return C


class FixedVertex(FormulationR2):
    def __init__(self, inputs, index=None, xy=None, **kwargs):
        """

        """
        FormulationR2.__init__(self, inputs, **kwargs)
        self.index = index
        self.xy = xy

    def as_constraint(self, **kwargs):
        return

    def as_objective(self, **kwargs):
        return []


class UnusableZoneSeg(FormulationR2):
    META = {'constraint': True, 'objective': False}

    def __init__(self, inputs, x, y, h, w, **kwargs):
        """
        Generally intersection of a line and a box is

        intersection of set.half_spaces and line.points is subset of set.halfspace

        """
        FormulationR2.__init__(self, inputs, **kwargs)
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self.bins = None

    def _as_mip_constraint(self, **kwargs):
        """ Mixed integer program
            where x1 and x2 must fall into one
        """
        X, Y = self.inputs.vars
        iseg = np.asarray([[i -1, i] for i in range(1, len(self.inputs))])

        ip1, ip2 = iseg[:, 0], iseg[:, 1]
        lhs, rhs = self._x - self._w / 2, self._x + self._w / 2
        btm, top = self._y - self._h / 2, self._y + self._h / 2
        vxr, vxl, vyu, vyb = [Variable(shape=len(self.inputs), boolean=True,  name='{}.{}'.format(self.name, i))
                              for i in range(4)]

        orxr, orxl, oryu, oryb = [Variable(shape=iseg.shape[0], boolean=True, name='{}.{}'.format(self.name, i))
                                  for i in range(4)]
        M = 100
        # constraint active implies Xi is within the half space H_i
        # if below:  x_i <= Hi * var
        # var = 1, T -> x_i <= 0   |
        # var = 0, T -> x_i <= 0    | FAIL
        # if above:  x_i <= Hi * (1 - var)
        # var = 1, T -> x_i <= 0    |
        # var = 0, T -> x_i <= Hi   |
        # --------------------------------
        C = [
            0 <= -X + lhs + M * (1 - vxl),
            0 <=  X - lhs + M * vxl,

            0 <= -X + rhs + M * vxr,
            0 <=  X - rhs + M * (1 - vxr),

            0 <= -Y + top + M * vyu,
            0 <=  Y - top + M * (1 - vyu),

            0 <=  Y - btm + M * vyb,
            0 <= -Y + btm + M * (1 - vyb),

            # ---------------------
            0 <= vxr[ip1] + vxr[ip2] - 2 * orxr,
            1 >= vxr[ip1] + vxr[ip2] - 2 * orxr,

            0 <= vxl[ip1] + vxl[ip2] - 2 * orxl,
            1 >= vxl[ip1] + vxl[ip2] - 2 * orxl,

            0 <= vyu[ip1] + vyu[ip2] - 2 * oryu,
            1 >= vyu[ip1] + vyu[ip2] - 2 * oryu,

            0 <= vyb[ip1] + vyb[ip2] - 2 * oryb,
            1 >= vyb[ip1] + vyb[ip2] - 2 * oryb,

            orxr + orxl + oryu + oryb >= 1
        ]
        self.bins = [vxl, vxr, vyu, vyb]
        return C

    def as_constraint(self, **kwargs):
        return self._as_mip_constraint(**kwargs)

    @classmethod
    def from_box(cls, inputs, x, y, h, w, **kwargs):
        """ base case"""
        return cls(inputs, x, y, h, w, **kwargs)

    @classmethod
    def from_segment(cls, inputs, p0, p1, **kwargs):
        """ """
        return

    @classmethod
    def from_boundary(cls, inputs, *points, **kwargs):
        """ convert the boundary into boxes"""
        return

    def as_objective(self, **kwargs):
        """ unusable zones can also be interpreted as maximizing distance to set
            since the objective is a Min, max is used.
            Maximize distance to self.set
            --or --
            Minimize the distance of points to free space
            (manually compute those from self.set)
            this would result in routing which prefers open space - but is not
            necessarily constrained to it. To make this stronger, add a large weight on it.
            self is described as a convex set.

        """
        X, Y = self.inputs.vars
        #Minimize(cvx.max(cvx.norm1(X - X) +
        #                 cvx.norm1(X - Y)))
        return

    def display(self):
        display = FormulationR2.display(self)
        datum = dict(x=self._x - 0.5 * self._w,
                     y=self._y - 0.5 * self._h,
                     w=self._w,
                     h=self._h,
                     color='grey',
                     index=0)
        display['boxes'].append(datum)
        return display


class PolyhedronConstraint(FormulationR2):
    def __init__(self, inputs, A, b, **kwargs):
        """
        if the discrete routing problem is solved, input[i] is restricted to
        the polyhedron Ax < b

        inputs: shape n, m
        A:      shape k, n
        B:      shape k
        """
        FormulationR2.__init__(self, inputs, **kwargs)
        self.A = A
        self.b = b

    def as_constraint(self, **kwargs):
        return [self.A @ self.inputs.mat <= self.b]

    def display(self):
        return


class TreePathCont(FormulationR2):
    def __init__(self, inputs, tgts, src=None, **kwargs):
        """

        """
        FormulationR2.__init__(self, inputs, **kwargs)



