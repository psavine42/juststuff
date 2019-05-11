from abc import ABC, abstractmethod
from copy import deepcopy, copy
from shapely import ops
from shapely.geometry import Polygon, LineString, MultiPolygon
# from shapely.affinity import translate
from src.layout import BuildingLayout
from src.building import Room
import torch
import numpy as np
import random

random.seed(1)


# Constraints -------------------------------------------
class Constraint(ABC):
    """
    Constraints that are functions of Layout (L)

    forward: L -> (0, 1)
    reward:  L -> (-1, 1)

    stubs for to LP, IP, QP for the constraint
    """
    def __init__(self, ent):
        self._ent = ent

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '{} on {}'.format(self.__class__.__name__, self._ent)

    def _from_layout(self, layout):
        return layout[self._ent]

    def gather_attrib(self, ent):
        pass

    @abstractmethod
    def is_satisfied(self, layout=None):
        pass

    @abstractmethod
    def _forward(self, layout):
        pass

    @abstractmethod
    def _fwd_discrete(self, layout):
        pass

    def forward(self, x):
        if torch.is_tensor(x):
            return self._fwd_discrete(x)
        return self._forward(x)

    def to_qp(self, problem):
        return []

    @property
    def ent(self):
        return self._ent

    @property
    def upper_bound(self):
        return 1.


class _BoundConstraint(Constraint):
    """ min <= ent[property] <= max  """
    def __init__(self, ent, *args, min=0, max=None, tol=None):
        Constraint.__init__(self, ent)
        if len(args) == 1:
            self._min = args[0]
            self._max = None
        elif len(args) == 0:
            self._min = min
            self._max = max

    @abstractmethod
    def gather_attrib(self, ent):
        pass

    def is_satisfied(self, layout=None):
        x = self.forward(layout)
        if self._min <= x <= self._max:
            return True

    def _forward(self, layout):
        """ domain is  """
        ent = self._from_layout(layout)
        x = self.gather_attrib(ent)
        if self._max:
            if self._min > x:
                z = self._min - x
            elif self._max < x:
                z = x - self._max
            else:
                z = 0
        else:
            z = abs(x - self._min) / self._min
        return z

    def _fwd_discrete(self, state_mat):
        # todo ------------------ can default be 1 ???
        pass

    def reward(self, layout):
        return 1 - self.forward(layout)


class ConvexityConstraint(Constraint):
    def __init__(self, ent, tol=0.1):
        Constraint.__init__(self, ent)
        self._tol = 0.1

    def to_qp(self, problem):
        return []

    def is_satisfied(self, layout=None):
        return self._tol > self.forward(layout)

    def _forward(self, layout):
        ent = self._from_layout(layout)
        area = ent.area
        z = (ent.convex_hull.area - area) / area
        return z

    def _fwd_discrete(self, state_mat):
        """ given a single plane
        domain is ( 1, inf ]
        """
        return torch.sum(state_mat)

    def show(self, ax=None):
        return np.arange(1, 5, 20)

    def reward(self, layout):
        return 1 - self.forward(layout)


class AreaConstraint(_BoundConstraint):
    def __init__(self,  ent, *args, **kwargs):
        _BoundConstraint.__init__(self, ent, *args, **kwargs)

    def to_qp(self, problem):
        fn = []
        ub, lb = self._min, self._max
        ix = problem.index_of(self.ent)
        if ub is not None:
            fn.append(lambda X: ub - 4 * X[ix + 2] * X[ix + 3])
        if lb is not None:
            fn.append(lambda X: 4 * X[ix + 2] * X[ix + 3] - lb )
        return [{'type': 'ineq', 'fun': x} for x in fn]

    def gather_attrib(self, ent):
        return ent.area

    def _fwd_discrete(self, state_mat):
        """ given a single plane"""
        return torch.sum(state_mat)


class AspectConstraint(_BoundConstraint):
    def __init__(self,  ent, *args, **kwargs):
        _BoundConstraint.__init__(self, ent, *args, **kwargs)

    def to_qp(self, problem):
        """
        min < (x_u / x_v) < max
        F(x) - min > 0
        max - F(x) > 0
        """
        fn = []
        ub, lb = self._min, self._max
        ix = 4 * problem.index_of(self.ent)
        if ub is not None:
            fn.append(lambda X: ub - X[ix + 2] / X[ix + 3])
            fn.append(lambda X: ub - X[ix + 3] / X[ix + 2])
        if lb is not None:
            fn.append(lambda X: X[ix + 2] / X[ix + 3] - lb)
            fn.append(lambda X: X[ix + 3] / X[ix + 2] - lb)
        return [{'type': 'ineq', 'fun': x} for x in fn]

    def gather_attrib(self, ent):
        minx, miny, maxx, maxy = ent.exterior.bounds
        aspx = maxx - minx
        aspy = maxy - miny
        return min(aspx, aspy) / max(aspx, aspy)

    def show(self, ax=None, samples=20):
        return np.arange(0, 1, samples)


class MinDimConstraint(_BoundConstraint):
    def __init__(self, ent, **kwargs):
        _BoundConstraint.__init__(self, ent, **kwargs)

    def gather_attrib(self, ent):
        minx, miny, maxx, maxy = ent.envelope.bounds
        return min([maxx - minx, maxy - miny])

    def show(self, ax=None, samples=20):
        return np.arange(0, 1, samples)


class MaxDimConstraint(_BoundConstraint):
    def __init__(self, ent, **kwargs):
        _BoundConstraint.__init__(self, ent, **kwargs)

    def gather_attrib(self, ent):
        minx, miny, maxx, maxy = ent.envelope.bounds
        return max([maxx - minx, maxy - miny])

    def _fwd_discrete(self, state_mat):
        """
            ~ dimensions of boundingbox( where(state == 1s) )

        """
        return

    def show(self, ax=None, samples=20):
        # normalize to dimension of footprint
        return np.arange(0, 1, samples)


class AdjacencyConstraint(Constraint):
    """
    requires that ent1 and ent2 are adjacent in the layout
    if dim is provided, the overlap must be of size dim
    """
    def __init__(self, ent1, ent2, dim=0.1):
        Constraint.__init__(self, ent1)
        self._ent2 = ent2
        self._dim = dim

    def __str__(self):
        return '{}: {}, {}'.format(
            self.__class__.__name__, self._ent, self._ent2
        )

    def gather_attrib(self, ent):
        pass

    @property
    def upper_bound(self):
        return 1.

    def _forward(self, layout: BuildingLayout):
        return int(not self.is_satisfied(layout))

    def qp_fn(self, problem):
        """
        min < F(x) < max
        F(x) - min > 0
        max - F(x) > 0
        """
        ix1 = 4 * problem.index_of(self._ent)
        ix2 = 4 * problem.index_of(self._ent)

        def constraint_max(X):
            val = (X[ix1+1] - X[ix2+1]) ** 2 + (X[ix1] - X[ix2]) ** 2
            lim = (X[ix1+2] + X[ix2+2])
            return lim - val ** 0.5

        def constraint_min(X):
            dx = (X[ix1] - X[ix2]) ** 2
            dy = (X[ix1+1] - X[ix2+1]) ** 2
            return (dx + dy) ** 0.5 - (X[ix1+2] + X[ix2+2])

        return [{'type': 'ineq', 'fun': constraint_max},
                {'type': 'ineq', 'fun': constraint_min}]

    def is_satisfied(self, layout=None):
        if self._ent2 in layout._adj[self._ent]:
            if self._dim is None:
                return True
            r1 = layout[self._ent].exterior
            r2 = layout[self._ent2].exterior
            adj = ops.shared_paths(r1, r2)
            if adj.is_empty:
                return False
            t1, t2 = adj
            if not t1.is_empty and t1.length >= self._dim:
                return True
            if not t2.is_empty and t2.length >= self._dim:
                return True
        return False

    def _fwd_discrete(self, state_tnsr):
        """ where tensor[0] is next to tensor[1]

        torch.where( dilate 1's on tensor[0] == tensor[1] )

        """
        # todo-filter which gets conv'd with tensor to get dilation
        mat = torch.zeros(5, 5)
        mat[0, :2, :] = 1
        mat[1, :, :2] = 1
        mat[2, 2:, :] = 1
        mat[3, :, 2:] = 1
        # corners
        for i in [2, 3]:
            mat[3, i:, i:] = 1
            mat[3, :i, i:] = 1
            mat[3, i:, :i] = 1
            mat[3, :i, :i] = 1
        import scipy.ndimage
        # scipy.ndimage.convolve
        # zeros = state_tnsr
        # torch.where(state_tnsr =)
        return

    def reward(self, layout):
        return 1. - self.forward(layout)


class AdjacencySoft(AdjacencyConstraint):
    def _forward(self, layout: BuildingLayout):
        """
        distance between lines == 0,
        distance between points > threshold

        """
        r1 = layout[self._ent]
        r2 = layout[self._ent2]
        ds = r1.distance(r2)


class HasLineConstraint(Constraint):
    """
    A constraint that an existing line must overlap a wall of 'ent'
    in otherwords - must touch this wall for at least N feet

    if dmin and dmax are specified,
    """
    def __init__(self, ent, geom: LineString, dmin=None, dmax=None):
        Constraint.__init__(self, ent)
        self._geom = geom
        if dmin is None:
            dmin = geom.length()
        if dmax is None:
            dmax = geom.length()
        self._min = dmin
        self._max = dmax

    def _forward(self, layout):
        ent = self._from_layout(layout)
        sp = ops.shared_paths(ent.exterior, self._geom)
        if sp.is_empty:
            return False
        seg1, seg2 = sp
        shared_len = 0.
        if not seg1.is_empty:
            shared_len += seg1.length
        if not seg2.is_empty:
            shared_len += seg2.length
        return shared_len

    def is_satisfied(self, layout=None):
        shared_len = self.forward(layout)
        if self._min <= shared_len <= self._max:
            return True


class FootPrintConstraint(Constraint):
    """ apply to whole layout"""
    def __init__(self, ent, geom):
        Constraint.__init__(self, ent)
        self._geom = geom
        self._area = geom.area

    def to_qp(self, problem):
        """ """
        fns = []
        fp = problem.footprint
        for ent in problem.program:
            ix = 4 * problem.index_of(ent.name)
            def constraint(X):
                # X[ix], X[ix+1]
                return

            fns.append(constraint)
        return []

    def tolinear(self, layout):
        """

        containment by generating all of:

            x_i - x_j <= y
        st. x_i is and element of exterior coords

        complexity = O( num_corners * num_footprint_cords )
        """
        for room in layout:
            coord = room.exterior.coords

    def _forward(self, layout: BuildingLayout):
        """ """
        assert layout is not None, 'layout required'
        merged = layout[self.ent].union(self._geom)  # shapely polygon
        merged_area = merged.area

        # if the footprint area is __ge__ fp.union(ent), no loss
        if self._area >= merged_area:
            return 0
        return abs(merged_area - self._area) / self._area

    def reward(self, layout):
        return 1. - self.forward(layout)

    @property
    def upper_bound(self):
        return 1.

    def is_satisfied(self, layout=None):
        return self._tol > self.forward(layout)

    def _fwd_discrete(self, state_tnsr):
        """
        Does not apply, unless there is a 'not allowed' plane
        """
        not_allowed = state_tnsr[0]
        # zeros = state_tnsr
        # torch.where(state_tnsr =)
        return

    def encode(self, layout):
        pass




class GlobalConstraint(Constraint):
    """
    Apply To everything in problem
    """
    pass


class ProgramConstraint(Constraint):
    pass


