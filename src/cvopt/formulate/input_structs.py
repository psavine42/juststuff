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
        c = [
            self.X - self.W / 2 <= u,
            self.X + self.W / 2 >= u,
            self.Y - self.H / 2 <= v,
            self.Y + self.H / 2 >= v,
        ]
        return [u, v], c

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
        """ Variable(shape=len(self)) """
        return cvx.hstack(self.X), cvx.hstack(self.Y)

    @property
    def outputs(self):
        return self.X, self.Y


class PointList3d(PointList):
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

