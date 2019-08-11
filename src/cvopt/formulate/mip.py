from .formulations import Formulation
import cvxpy as cvx
import numpy as np
from cvxpy import Variable, Minimize
from .cont_base import FormulationR2, NumericBound


def emul(lh, rh):
    return cvx.multiply(lh, rh)


class MIPConstraint(FormulationR2):
    def __init__(self, inputs, indices=None, **kwargs):
        """
        continuous X_i
        x_i = values[i, 0] or x_i = values[i, 1]

        indices
        """
        FormulationR2.__init__(self, inputs, **kwargs)
        self._internal_vars = None
        self._indices = None

        # set indices to use when applicable
        if indices is None:
            self._indices = list(range(len(inputs)))
        else:
            self._indices = indices

    def _input_vars(self):
        if isinstance(self.inputs, (list, tuple, Variable)):
            return self.inputs
        return self.inputs.vars

    @property
    def indicators(self):
        """ accessor for indicator variables generated by MIP formulation
        """
        return self._internal_vars

    @indicators.setter
    def indicators(self, value):
        self._internal_vars = value


# -------------------------------------------------------
class DiscreteValueChoice(MIPConstraint):
    def __init__(self, inputs, values, **kwargs):
        """
        continuous X_i
        x_i = values[i, 0] or x_i = values[i, 1]
        todo - currently
        """
        MIPConstraint.__init__(self, inputs, **kwargs)
        self._choices = values
        self._internal_vars = Variable(shape=(len(values), len(self._indices)), boolean=True)

    def as_constraint(self, **kwargs):
        X = self._input_vars()
        choices = self._choices
        N = len(choices)

        ind = self.indicators
        exprs = []
        for i in range(N):
            exprs.append(cvx.multiply(ind[i, :], choices[i]))

        # discrete sizes --or-- 2 /4 constraints
        C = [
            X == cvx.sum(cvx.vstack(exprs), axis=1),
            cvx.sum(ind, axis=1) == 1,
        ]
        return C


# todo Generalize! ---------------------------------------------------

class OrientationConstr(MIPConstraint):
    def __init__(self, inputs, indices=None, eq=True, **kwargs):
        """
        for orientable shapes, restrict shapes to have same orientation
        see BoxInputList.orientation for details
        """
        MIPConstraint.__init__(self, inputs, indices=indices, **kwargs)
        self._eq = eq

    @property
    def indicators(self):
        """ see BoxInputList.orientation for details """
        ovars, _ = self.inputs.orientation
        return ovars

    def as_constraint(self, **kwargs):
        ovars, C = self.inputs.orientation
        inds = self._indices
        for i in range(len(inds) - 1):
            if self._eq is True:
                C += [ovars[inds[i]] == ovars[inds[i + 1]]]
            else:
                # todo - this is wrong FIXME
                C += [ovars[inds[i]] == 1 - ovars[inds[i + 1]]]
        return C


class FixedDimension(MIPConstraint):
    def __init__(self, inputs, values, **kwargs):
        """
        box.W = values[0] and  box.H = values[1]
        or
        box.W = values[1] and  box.H = values[0]
        """
        MIPConstraint.__init__(self, inputs, **kwargs)
        self._choices = values
        self._internal_vars = Variable(shape=(len(values), len(self._indices)), boolean=True)

    def as_constraint(self, **kwargs):
        """ constrain W and H to mutually exclusive discrete values """
        W, H = self.inputs.W, self.inputs.H
        v1, v2 = self._choices
        ind = self._internal_vars
        i = self._indices
        C = [
            W[i] == cvx.multiply(v1[i], ind[0]) + cvx.multiply(v2[i], ind[1]),
            H[i] == cvx.multiply(v1[i], ind[1]) + cvx.multiply(v2[i], ind[0]),
            ind[0] + ind[1] == 1,
        ]
        return C


# todo FINISH THEN Generalize! ---------------------------------------------------
class OneEdgeMustTouchBoundary(MIPConstraint):
    def __init__(self, inputs, values, indicies=None, **kwargs):
        MIPConstraint.__init__(self, inputs, **kwargs)
        self._choices = values
        self._indices = indicies
        self._internal_vars = Variable(shape=(4, len(inputs)), boolean=True)

    @property
    def indicators(self):
        """ when indicators[j, i] == 1, boundry[j] is touched by input[i]"""
        return (1 - self._internal_vars)

    def _alt(self, **kwargs):
        """ atleast one of the edges must touch a wall
            if it must be in a corner, then >= 2!

        Indicators
        if ind[0, i] == 1, then
        """
        bx = self.inputs
        ymin,   ymax,   xmin,   xmax = bx.bottom, bx.top, bx.left, bx.right
        by_min, by_max, bx_min, bx_max = self._choices

        M = np.max(self._choices) * 2
        ind = self._internal_vars # (1 - self._internal_vars)

        C = [
            by_max <=  ymin + emul(by_min, ind[0]),
            0 <= -by_max + ymax + ind[1],
            0 <=  bx_min - xmin + ind[2],
            0 <= -bx_max + xmax + ind[3],
            cvx.sum(self._internal_vars, axis=0) <= 3,
            # cvx.sum(self._internal_vars, axis=0) <= 1,
        ]
        return C

    def as_constraint(self, **kwargs):
        """ atleast one of the edges must touch a wall
            if it must be in a corner, then >= 2!

        Indicators
        if ind[0, i] == 1, then
        """
        bx = self.inputs
        ymin,   ymax,   xmin,   xmax = bx.bottom, bx.top, bx.left, bx.right
        by_min, by_max, bx_min, bx_max = self._choices

        M = np.max(self._choices) * 2
        ind = M * self._internal_vars # (1 - self._internal_vars)

        C = [
            0 <=  by_min - ymin + ind[0],
            0 <= -by_max + ymax + ind[1],
            0 <=  bx_min - xmin + ind[2],
            0 <= -bx_max + xmax + ind[3],
            cvx.sum(self._internal_vars, axis=0) <= 3,
            # cvx.sum(self._internal_vars, axis=0) <= 1,
        ]
        return C


class DimFromEdge(MIPConstraint):
    def __init__(self, inputs, values, **kwargs):
        """ THis may be more efficent in the discrete space,
            as with fixed dimensions, this becomes
            sum(edge_i @ X) == min_dim
        """
        MIPConstraint.__init__(self, inputs, **kwargs)
        self._choices = values
        self._internal_vars = Variable(shape=(4, len(inputs)), boolean=True)

    def as_constraint(self, **kwargs):
        bx = self.inputs
        bottom, top, left, right = bx.bottom, bx.top, bx.left, bx.right
        c1, c2, c3, c4 = self._choices
        M = np.max(self._choices) * 2
        ind = self._internal_vars
        C = [
            0 <=  c1 - bottom + M * ind[0],
            0 <= -c2 + top + M * ind[1],
            0 <=  c3 - left + M * ind[2],
            0 <= -c4 + right + M * ind[3],
            cvx.sum(ind, axis=0) >= 1,
        ]
        # self.indicators = ind
        return C


def fixed_dimension(inputs, values):
    return DiscreteValueChoice(inputs, values)






