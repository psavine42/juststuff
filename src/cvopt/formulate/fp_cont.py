from .formulations import Formulation
import cvxpy as cvx
from cvxpy import Variable, Minimize
import numpy as np
import dccp
from . import form_utils as fu
from .cont_base import FormulationR2, NumericBound
from src.cvopt.shape.base import R2
from .positioning import *
from .input_structs import BoxInputList

"""
Generally, the inputs for a floorplanning problem are given as follows:
• a set of n rectangular modules S = {1, 2, . . . , n} with a list of areas a i , 1 ≤ i ≤ n;
• a partition of S into sets S 1 and S 2 representing the modules with fixed and
    free orientations respectively;
• an interconnection matrix C n×n = [c ij ] , 1 ≤ i, j ≤ n, where c ij captures the
    connectivity between modules i and j 
    (we assume C is symmetric, i.e., c ij = c ji,given by netlist)
• values a_i for the area of each module i;
• bounds R_i low and R i up on the aspect ratio R i of each module i;
• bounds wF low , wF_up , h lowF, and h F on the width and height respectively of the
    floorplan, for an instance of outline-free floorplanning
• values w F and h F for the width and height of the floorplan, for an instance of
    fixed-outline floorplanning.

The required outputs are as follows. The floorplanning problem is to determine the
location, width, and height of each module on the floorplan so that:
• there is no overlap between the modules;
• coordinates (x i , y i ), height h i and width w i for each module such that w i × h i =
    a i , 1 ≤ i ≤ n;
• R i low ≤ h i w i
• R i low ≤ h i w i ≤ R i up for every module i with fixed orientation (i ∈ S 1 );
    ≤ R i up or 1 R up_i ≤ h i w i ≤ R low i
    for every module i with free orientation (i ∈ S 2 );
• all the modules are enveloped in the floorplan and the total wirelength is minimized.

An optimum floorplan is achieved by optimizing the desired objective function.
Possible objectives are (Lu et al., 2008):
• minimize area (bus area);
• minimize wirelength; 20
• maximize routability;
• minimize power dissipation;
• minimize timing/delays; or
• any combination of the above.
"""


def tris(n):
    tri_i, tri_j = np.triu_indices(n, 1)
    return zip(tri_i.tolist(), tri_j.tolist())


# usable/unusable space ------------------------------------------------------------
class BoundsXYWH(FormulationR2):
    META = {'constraint': True, 'objective': False}

    def __init__(self, inputs=None, w=None, h=None, wmin=0, hmin=0, **kwargs):
        """ the bounding box for a fixed outline floor plan """
        FormulationR2.__init__(self, inputs, **kwargs)
        self.w_max = w
        self.h_max = h
        self.w_min = wmin
        self.h_min = hmin

    def as_constraint(self, **kwargs):
        fp = self.inputs
        C = [
            fp.X + 0.5 * fp.W <= self.w_max,   # 0.5 * self.w,
            fp.Y + 0.5 * fp.H <= self.h_max,   # 0.5 * self.h,
            fp.X - 0.5 * fp.W >= 0.,        # 0.5 * self.w,
            fp.Y - 0.5 * fp.H >= 0.,        # 0.5 * self.h,
        ]
        return C

    def as_objective(self, **kwargs):
        return None

    def describe(self):
        return


class UnusableZone(FormulationR2):
    def __init__(self, inputs, x=None, y=None, w=None, h=None,
                 rpm=None,
                 pts=None,
                 **kwargs):
        """
        unusable zones within the fixed outline floor plan bounding box

        represented as entries in an RPM which is built on the fly.
        """
        FormulationR2.__init__(self, inputs=inputs, **kwargs)
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.rpm = rpm
        self._points = pts.tolist()
        if rpm is not None:
            self._points = rpm.points

    def _as_mip_constraint(self, **kwargs):
        X, Y, W, H = self.inputs.vars
        C = []

        g = [x is None for x in [self.w_max, self.h_max, self.w_min, self.h_min]]
        if sum(g) > 2:
            raise Exception('invalid deadzone constraints ')

        if sum(g) == 2:
            vx = Variable(shape=X.shape, boolean=True)
            vy = Variable(shape=X.shape, boolean=True)
            C += [vx + vy >= 1]

            if self.w_min is not None:
                C += [X + W / 2 <= self.w_min * vx]
            elif self.w_max is not None:
                C += [X - W / 2 >= self.w_max * vx]

            if self.h_min is not None:
                C += [Y + H / 2 <= self.h_min * vy]
            elif self.h_max is not None:
                C += [Y - H / 2 >= self.h_max * vy]
        return C

    def as_constraint(self, **kwargs):
        """
        SDP
            create a fake RPM with an entry for the box defined by self.x,y,w,h
            the last entry of rpm' is self.bounds
            for each of these entries, make a new RPM constraint

        MLP (no RPM available and problem is not SDP)
            boolean decision variables

        """
        C = []
        X, Y, W, H = self.inputs.vars

        #
        base = self._points
        base.append([self.x, self.y])
        new_rpm = RPM.points_to_mat(base)
        #
        for i in range(new_rpm.shape[0]-1):
            C += RPM.expr(0, new_rpm[i, -1, 0], X[i], self.x, W[i], self.w)
            C += RPM.expr(1, new_rpm[i, -1, 1], Y[i], self.y, H[i], self.h)
        return C

    def display(self):
        display = FormulationR2.display(self)
        datum = dict(x=self.x - 0.5 * self.w,
                     y=self.y - 0.5 * self.h,
                     w=self.w, h=self.h,
                     color='black', index=0)
        display['boxes'].append(datum)
        return display


# ------------------------------------------------------------
class BoxAspect(FormulationR2):
    def __init__(self, inputs=None, high=None, low=None, **kwargs):
        """
        """
        FormulationR2.__init__(self, inputs, **kwargs)
        self.B = Variable(shape=len(inputs), pos=True, name=self.name)
        self._bnd = NumericBound(self.B, high=high, low=low)

    def vars(self):
        return self.B

    def as_constraint(self, **kwargs):
        """
        Generates SDP constraints for box aspect 'B'

        """
        in_size = len(self.inputs)
        A_i = np.asarray([x.area for x in self.inputs.inputs])

        # SDP to enforce aspect constraints
        _, _, W, H = self.inputs.vars
        C = [cvx.PSD(cvx.bmat([[self.B[i], W[i]],
                               [W[i], A_i[i]]])) for i in range(in_size)]

        C += [cvx.PSD(cvx.bmat([[self.B[i], H[i]],
                                [H[i], A_i[i]]])) for i in range(in_size)]

        C += self._bnd.as_constraint(self)
        return C

    def as_objective(self, **kwargs):
        """ minimize the aspects """
        return cvx.Minimize(cvx.sum(self.B))


class BoxAspectLinear(FormulationR2):
    def __init__(self, inputs=None, high=None, low=None, **kwargs):
        """
        """
        FormulationR2.__init__(self, inputs, **kwargs)
        self._bnd_high = high
        self._bnd_low = low

    def as_constraint(self, **kwargs):
        """
        li < h/w < u_i
        """
        _, _, W, H = self.inputs.vars
        C = [
            self._bnd_high * W >= H
        ]
        return C


# todo ---------------------
class MustTouchEdge(FormulationR2):
    pass


class MaximizeDistance(FormulationR2):
    def __init__(self, inputs, max_dists=None, **kwargs):
        """
        Sometimes there are two program elements that want to be as far away
        from each other as possible (on center)

        'motivation'
            for example - a building with egress requirements would want
            to have staircases adjacent to a hallway. But they should be on opposite
            sides of that hallway!
            Formulated as a constraint, this could say the dist(I[i], I[j]) >= 100ft

            Formulated as an objective (Ie - them as far away as possible from each other)


        Note - if there is as an RPM associated with the problem,
        todo maximization of this will cause a Concave-Convex objective ...
        :param inputs:
        :param kwargs:
        """
        FormulationR2.__init__(self, inputs, **kwargs)

    def as_constraint(self, **kwargs):

        return

    def as_objective(self, **kwargs):
        return

    def display(self):
        return []


class SplitTree(FormulationR2):
    def __init__(self, inputs, adj=None, **kwargs):
        FormulationR2.__init__(self, inputs, **kwargs)
        self.adj_matrix = adj

    def as_constraint(self, **kwargs):
        """
        given an adjacency matrix,


        WH = max{wA + wB , wC + wD }(max{hA , hB } + max{hC , hD }).

        Take max of all possible chains left-right
        * max of all possible chains up-down

        number of chains in any direction is (box_dim /min_unit_dim)!
        """
        X, Y, W, H = self.inputs.vars
        num_children = len(self.inputs)
        A = Variable(shape=num_children, boolean=True)
        C = []

        def constraint_node(wi, wj, hi, hj):
            wa = Variable(pos=True)
            ha = Variable(pos=True)
            ha = Variable(pos=True)
            c = [
                wa <= wi + wj,
                ha <= cvx.max(hi + hj),
            ]

            return
        return


class DisputedZone(FormulationR2):
    def __init__(self, inputs, offset, adj, **kwargs):
        FormulationR2.__init__(self, inputs, **kwargs)
        self.offset = offset
        self.adj = adj

    def as_constraint(self, **kwargs):
        X, Y, W, H = self.inputs.vars
        for (i, j) in self.adj:
            b1 = [Variable(boolean=True) for i in range(4)]
            cutvar = Variable()
            # vert
            # if x and
            vert_ij = cutvar * self.offset * X[i]
            vert_ji = X[j] - vert_ij
            # horizantal
            # horz_ij = cutvar *
            # horz_ji = cutvar *


class StructuralConstraint(FormulationR2):
    def __init__(self, inputs):
        FormulationR2.__init__(self, inputs)


# --------------------------------------------------------------
# Area
# --------------------------------------------------------------
def linearize_abs(X, Y, name=None):
    """
    constraints for absolute values of distance matrix of X, Y (only upper tri entries)

    """
    n = X.shape[0]
    tri_i, tri_j = np.triu_indices(n, 1)
    U = Variable(shape=tri_j.shape[0], pos=True, name='U.{}'.format(name))
    V = Variable(shape=tri_j.shape[0], pos=True, name='V.{}'.format(name))
    tri_i, tri_j = tri_i.tolist(), tri_j.tolist()
    C = [
        # linearized absolute value constraints
        # | x_i - x_j |  | y_i - y_j |
        U >= X[tri_i] - X[tri_j],
        U >= X[tri_j] - X[tri_i],

        V >= Y[tri_i] - Y[tri_j],
        V >= Y[tri_j] - Y[tri_i]
    ]
    return (U, V), C


# Area as Constant
class MinFixedPerimeters(FormulationR2):
    def __init__(self, inputs, **kwargs):
        """
        based on Novel Convex Optimization Approaches for VLSI Floorplanning  2008 SDP
        for a floorplanning problem with constraints on the area of cells,
        minimize distances between

        Arguments:
            inputs: Boxlist

        """
        FormulationR2.__init__(self, inputs, **kwargs)
        num_in = len(inputs)
        tris = np.triu(np.ones((num_in, num_in), dtype=int), 1).sum()
        self.U = Variable(shape=tris, pos=True, name='U.{}'.format(self.name))
        self.V = Variable(shape=tris, pos=True, name='V.{}'.format(self.name))

    def as_constraint(self):
        """
        1) generate SDP constraints for Area .
            w * h >= a      (min w, h)
        2) transform absolute value distance of X,Y to Variables
        # todo
        """
        X, Y, W, H = self.inputs.vars
        num_in = len(self.inputs)
        A_i = np.asarray([x.area for x in self.inputs.inputs])
        a2s = np.sqrt(A_i)
        C = [cvx.PSD(cvx.bmat([[W[i], a2s[i]],
                               [a2s[i], H[i]]])) for i in range(num_in)]

        if self.is_objective is True:
            tri_i, tri_j = np.triu_indices(num_in, 1)
            tri_i, tri_j = tri_i.tolist(), tri_j.tolist()
            C += [
                # linearized absolute value constraints
                # | x_i - x_j |  | y_i - y_j |
                self.U >= X[tri_i] - X[tri_j],
                self.U >= X[tri_j] - X[tri_i],

                self.V >= Y[tri_i] - Y[tri_j],
                self.V >= Y[tri_j] - Y[tri_i]
            ]
        return C

    def as_objective(self, **kwargs):
        o1 = Minimize(cvx.sum(self.U + self.V))
        return o1

    def describe(self):
        return


# Area as function of some variables F(w,h)
class MinArea(FormulationR2):
    def __init__(self, inputs, area, method='log', **kwargs):
        """ Areas as function of 2 scalars """
        FormulationR2.__init__(self, inputs, **kwargs)
        self._min_area = area
        self._fn = self.log_area if method == 'log' else self.mean_area

    @classmethod
    def log_area(cls, w, h, area):
        return cvx.log(w) + cvx.log(h) >= np.log(area)

    @classmethod
    def mean_area(cls, w, h, area):
        return cvx.geo_mean(cvx.hstack([w, h])) >= np.sqrt(area)

    def as_constraint(self, **kwargs):
        _, _, W, H = self.inputs.vars
        return [self._fn(W, H, self._min_area)]


# Area as Variable
class MinAreaGP(FormulationR2):
    pass


# Stage 1 ----------------------------------------------------------------


# stage2 --------------------------------------------------------------


# stage2 --------------------------------------------------------------
class PlaceLayoutSDP(BoxInputList):
    def __init__(self, children,
                 rpm=None,
                 aspect=4,
                 width=None,
                 height=None,
                 lims=None):
        """
        based on Novel Convex Optimization Approaches for VLSI Floorplanning  2008 SDP

        Arguments:
         - Paramaters:
            children : len(n) objects with 'area' interface
            width:  (int) width of layout
            height: (int) height of layout

         - child formulations:
            RPM: upper tri tensor [n x n x 2]
                todo if RPM is not Given, generate N(n - 1) options per
            B: Aspect: min/max aspect ratio of boxes

        """
        BoxInputList.__init__(self, children)
        num_in = self.X.shape[0]
        tris = np.triu(np.ones((num_in, num_in), dtype=int), 1).sum()
        self.B = Variable(shape=self.X.shape[0], pos=True, name='aspect')
        self.U = Variable(shape=tris, pos=True)
        self.V = Variable(shape=tris, pos=True)

        # children
        self.aspect = NumericBound(self.B, high=aspect)
        self.rpm = RPM(self, rpm)
        self.bnds = BoundsXYWH(self, width, height)

    def as_constraint(self):
        """
        1) generate SDP constraints for

        Area and Aspect.
            w * h >= a      (min w, h)

        Aspect
            a * B >= h^2
            => a * B >= h * h
            => [ B, h ]
               [ h, a ]

        2) transform absolute value distance of X,Y to Variables
        """
        num_in = self.X.shape[0]
        tri_i, tri_j = np.triu_indices(num_in, 1)
        tri_i, tri_j = tri_i.tolist(), tri_j.tolist()

        A_i = np.asarray([x.area for x in self._inputs])
        a2s = np.sqrt(A_i)
        C = [
            # linearized absolute value constraints
            self.U >= self.X[tri_i] - self.X[tri_j],
            self.U >= self.X[tri_j] - self.X[tri_i],
            self.V >= self.Y[tri_i] - self.Y[tri_j],
            self.V >= self.Y[tri_j] - self.Y[tri_i]
        ]

        # SDP constraints
        # a <= w * h
        C += [cvx.PSD(cvx.bmat([[self.W[i], a2s[i]],
                                [a2s[i], self.H[i]]])) for i in range(num_in)]

        # aspect
        # a * B >= h^2
        C += [cvx.PSD(cvx.bmat([[self.B[i], self.W[i]],
                                [self.W[i], A_i[i]]])) for i in range(num_in)]

        C += [cvx.PSD(cvx.bmat([[self.B[i], self.H[i]],
                                [self.H[i], A_i[i]]])) for i in range(num_in)]

        # child constraints
        C += self.aspect.as_constraint()
        # RPM adjacency constraints
        C += self.rpm.as_constraint()
        # within bounds
        C += self.bnds.as_constraint()
        return C

    def as_objective(self, **kwargs):
        o1 = Minimize(cvx.sum(self.U + self.V))
        return o1

    def describe(self):
        return


class Nstage(FormulationR2):
    """
    todo [citation]
    Algorithm 1 Algorithm for Interchange-free Local Improvement
    Input: SRPM
    Output: Aspect ratios, module dimensions
    1. Solve SOCP model without aspect ratio constraints;
    2. If all the aspect ratios are satisfactory, goto Step 9; otherwise, goto Step 3;
    3. Select all the Central Modules M i ;
    4. Select and compute all the First Layer Modules M ij ;
    5. Select and compute all the Second Layer Modules M ijk ;
    6. Set up the relaxed SRPM;
    7. Solve the SOCP model with relaxed SRPM without aspect
    ratio constraints to obtain a layout with overlaps; based on
    this result to update the SRPM;
    8. Re-solve SOCP model with aspect ratio constraints;
    9. End.

    this is ok since a ~30 dep problem takes like 0.2 seconds - i can do 10
    """
    pass


class HallwayEntity(FormulationR2):
    """
    Engineering problem:
        this thing would have to 'rebuild' adjacency constraints if strict


    Math problem:
        how to express w/o all the extra shyt
    """
    pass


class Motion(FormulationR2):
    pass



