from .fp_cont import FormulationR2
from cvxpy import Variable, Minimize, Parameter, Constant
import cvxpy as cvx
import numpy as np


class BoxInputList(FormulationR2):
    def __init__(self, children, **kwargs):
        """
        second stage optimization of - Fixed Outline or classical FP problem
        incorporate the RPM, aspect ratio, and bounds

        implementations are:
        1)
            Large-Scale Fixed-Outline Floorplanning Design
            Using Convex Optimization Techniques            2008    SOC
        2)  Novel Convex Optimization Approaches
                for VLSI Floorplanning                      2008    SDP
        3)
            An Efficient Multiple-stage Mathematical
            Programming Method for Advanced Single and
            Multi-Floor Facility Layout Problems                    LP

        X: x center locations of boxes
        Y: y center locations of boxes
        W:
        H:

        """
        FormulationR2.__init__(self, children, **kwargs)
        num_in = len(children)
        self.X = Variable(shape=num_in, pos=True, name='Stage2.x')
        self.Y = Variable(shape=num_in, pos=True, name='Stage2.y')
        self.W = Variable(shape=num_in, pos=True, name='Stage2.w')
        self.H = Variable(shape=num_in, pos=True, name='Stage2.h')

    @property
    def inputs(self):
        return self._inputs

    @property
    def vars(self):
        return self.X, self.Y, self.W, self.H

    @property
    def outputs(self):
        return self.X, self.Y, self.W, self.H

    @property
    def num_actions(self):
        return self.X.shape[0]

    @property   # todo compute once
    def points_in_box(self):
        """ NOT canonical expressions for an arbitrary point within each box"""
        u = Variable(shape=len(self), pos=True, name='U.x')
        v = Variable(shape=len(self), pos=True, name='V.x')
        return [u, v], self.within(u, v)

    @property
    def top(self):
        return self.Y + self.H/2

    @property
    def bottom(self):
        return self.Y - self.H / 2

    @property
    def left(self):
        return self.X - self.W / 2

    @property
    def right(self):
        return self.X + self.W / 2

    def within(self, u, v):
        return [self.X - self.W / 2 <= u,
                self.X + self.W / 2 >= u,
                self.Y - self.H / 2 <= v,
                self.Y + self.H / 2 >= v]

    def __len__(self):
        return self.X.shape[0]

    def display(self):
        display = FormulationR2.display(self)
        X, Y, W, H = self.X.value, self.Y.value, self.W.value, self.H.value
        for i in range(self.num_actions):
            datum = dict(x=X[i] - 0.5 * W[i], y=Y[i] - 0.5 * H[i], w=W[i], h=H[i], index=i)
            display['boxes'].append(datum)
        return display


class PointList(FormulationR2):
    def __init__(self, children, **kwargs):
        """
        second stage optimization of - Fixed Outline or classical FP problem
        incorporate the RPM, aspect ratio, and bounds

        implementations are:
        1)
            Large-Scale Fixed-Outline Floorplanning Design
            Using Convex Optimization Techniques            2008    SOC
        2)  Novel Convex Optimization Approaches
                for VLSI Floorplanning                      2008    SDP
        3)
            An Efficient Multiple-stage Mathematical
            Programming Method for Advanced Single and
            Multi-Floor Facility Layout Problems                    LP

        todo - test which is better with this solver ?? cvx calls an SDP solver irregarless so...

        X: x center locations of boxes
        Y: y center locations of boxes
        W:
        H:

        """
        if isinstance(children, int):
            children = [None for i in range(children)]
        num_in = len(children)
        self._shape = num_in
        FormulationR2.__init__(self, children, **kwargs)
        self.X = []
        self.Y = []

    # def __getitem__(self, item):
    #     return self.mat[item]

    @property
    def inputs(self):
        return self._inputs

    @property
    def vars(self):
        """ Variable(shape=len(self)) """
        return cvx.hstack(self.X), cvx.hstack(self.Y)

    @property
    def mat(self):
        """ Variable(shape=(len(self), 2)) """
        return cvx.vstack(self.vars)

    # ---------------------------------------
    @property
    def top(self):
        return self.Y

    @property
    def bottom(self):
        return self.Y

    @property
    def left(self):
        return self.X

    @property
    def right(self):
        return self.X

    @property
    def outputs(self):
        return self.X, self.Y

    @property
    def num_actions(self):
        return self._shape

    def __len__(self):
        return self._shape

    def __setitem__(self, key, value):
        self.X[key] = Parameter(value[0])
        self.Y[key] = Parameter(value[1])

    def display(self):
        display = FormulationR2.display(self)
        X, Y = [x.value for x in self.vars]
        for i in range(self.num_actions):
            datum = dict(x=X[i], y=Y[i], index=i)
            display['segments'].append(datum)
        return display

    @classmethod
    def _var_list(cls, n, name):
        return [Variable(pos=True, name='{}.{}.{}'.format(cls.__name__, name, i)) for i in range(n)]


class PointList2d(PointList):
    def __init__(self, children, **kwargs):
        """
        second stage optimization of - Fixed Outline or classical FP problem
        incorporate the RPM, aspect ratio, and bounds

        implementations are:
        1)
            Large-Scale Fixed-Outline Floorplanning Design
            Using Convex Optimization Techniques            2008    SOC
        2)  Novel Convex Optimization Approaches
                for VLSI Floorplanning                      2008    SDP
        3)
            An Efficient Multiple-stage Mathematical
            Programming Method for Advanced Single and
            Multi-Floor Facility Layout Problems                    LP

        todo - test which is better with this solver ?? cvx calls an SDP solver irregarless so...

        X: x center locations of boxes
        Y: y center locations of boxes
        W:
        H:

        """
        PointList.__init__(self, children, **kwargs)
        self.X = [Variable(pos=True, name='Stage2.x{}'.format(i)) for i in range(self._shape)]
        self.Y = [Variable(pos=True, name='Stage2.y{}'.format(i)) for i in range(self._shape)]

    # ---------------------------------------
    @property
    def vars(self):
        """ Variable(shape=len(self, 2)) """
        return cvx.hstack(self.X), cvx.hstack(self.Y)

    @property
    def outputs(self):
        return self.X, self.Y


class PointList3d(PointList):
    def __init__(self, children, **kwargs):
        PointList.__init__(self, children, **kwargs)
        self.X = [Variable(pos=True, name='Stage2.x{}'.format(i)) for i in range(self._shape)]
        self.Y = [Variable(pos=True, name='Stage2.y{}'.format(i)) for i in range(self._shape)]
        self.Z = [Variable(pos=True, name='Stage2.z{}'.format(i)) for i in range(self._shape)]

    @property
    def vars(self):
        """ Variable(shape=len(self)) """
        return cvx.hstack(self.X), cvx.hstack(self.Y), cvx.hstack(self.Z)

    @property
    def mat(self):
        """ Variable(shape=(len(self), 2)) """
        return cvx.vstack(self.vars)

    @property
    def outputs(self):
        return self.X, self.Y, self.Z


class CircleList(PointList):
    def __init__(self, n, x=None, y=None, r=None, dim=2, **kwargs):
        PointList.__init__(self, n, **kwargs)
        # self.X = self._var_list(self._shape, 'x') if x is None else x
        # self.Y = self._var_list(self._shape, 'y') if y is None else y
        self.X = Variable(shape=(self._shape, dim), name=self.name) if x is None else x
        self.R = self._var_list(self._shape, 'r') if r is None else r

    @property
    def vars(self):
        """ Variable(shape=len(self)) """
        return self.X, cvx.hstack(self.R)

    @property
    def areas(self):
        return np.pi * self.R

    @property
    def radii(self):
        return self.R

    def _min_dist(self):
        # todo
        return Minimize( cvx.max(cvx.max(cvx.abs(c), axis=1) + self.R))




class Group(object):
    pass

