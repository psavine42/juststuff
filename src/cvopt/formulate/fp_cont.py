from .formulations import Formulation
import cvxpy as cvx
from cvxpy import Variable, Minimize
import numpy as np
import dccp
from . import form_utils as fu
from .cont_base import FormulationR2, NumericBound

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


# Utilities ----------------------------------------------------------------
class RPM(FormulationR2):
    META = {'constraint': True, 'objective': False}

    def __init__(self, domain, layout, mat):
        """
        todo - add overlap term
        todo - minimize overlap as objective

        Relative Position Matrix
        for children

        upper triangular matrix
            M[i, j, 0] = 0  -> no constraint
            M[i, j, 0] = 1  -> i to left of j
            M[i, j, 0] = 2  -> i to right of j

            M[i, j, 1] = 0  ->  no constraint
            M[i, j, 1] = 1  ->  i above j
            M[i, j, 1] = 2  ->  i below j

        2 -> to above of ...
        """
        FormulationR2.__init__(self, domain, [layout])
        self.mat = mat

    @property
    def num_actions(self):
        return self.mat.shape[0]

    def as_constraint(self):
        C = []
        fp = self._inputs[0]
        x, y, w, h = fp.X, fp.Y, fp.W, fp.H
        for i, j in tris(self.mat.shape[0]):
            lr, ud = self.mat[i, j]
            if lr == 1:     # i left of j
                C += [x[j] - x[i] >= 0.5 * (w[i] + w[j])]
            elif lr == 2:   # i right of j
                C += [x[i] - x[j] >= 0.5 * (w[i] + w[j])]

            if ud == 1:     # i above j
                C += [y[i] - y[j] >= 0.5 * (h[i] + h[j])]
            elif ud == 2:   # i below j
                C += [y[j] - y[i] >= 0.5 * (h[i] + h[j])]
        return C

    def as_objective(self, **kwargs):
        return None

    def describe(self, arr=None, text=None):
        """ text of relations between RPM entries """
        txt = ''
        num = self.num_actions
        arr = [[''] * num for i in range(num)]
        for i, j in tris(num):
            lr, ud = self.mat[i, j]
            s = ''
            if lr == 1:
                txt += '\n{} left of {}'.format(i, j)
                s += '1'
            elif lr == 2:
                txt += '\n{} rght of {}'.format(i, j)
                s += '2'
            else:
                s += '0'

            if ud == 1:
                txt += '\n{} above   {}'.format(i, j)
                s += '1'
            elif ud == 2:
                txt += '\n{} below   {}'.format(i, j)
                s += '2'
            else:
                s += '0'
            arr[i][j] = s
        if text:
            return txt
        return [' '.join(x) for x in arr]


class SRPM(RPM):
    def __init__(self, domain, layout, mat):
        """
        Sparse Relative Position Matrix
        before calling self.as_constraint(), removes redundant entries
        """
        FormulationR2.__init__(self, domain, [layout])
        self.base = mat     # keep a copy of the base matrix

    def _sparsify(self, mat):
        # todo implement
        return mat

    def as_constraint(self):
        self.mat = self._sparsify(self.mat)
        return RPM.as_constraint(self)


class BoundsXYWH(FormulationR2):
    META = {'constraint': True}

    def __init__(self, space, layout, w, h, **kwargs):
        FormulationR2.__init__(self, space, [layout], **kwargs)
        self.w = w
        self.h = h

    def as_constraint(self):
        fp = self._inputs[0]
        C = [
            fp.X + 0.5 * fp.W <= self.w,   # 0.5 * self.w,
            fp.Y + 0.5 * fp.H <= self.h,   # 0.5 * self.h,
            fp.X - 0.5 * fp.W >= 0.,        # 0.5 * self.w,
            fp.Y - 0.5 * fp.H >= 0.,        # 0.5 * self.h,
        ]
        return C

    def as_objective(self, **kwargs):
        return None

    def describe(self):
        return


class BoxAspect(FormulationR2):
    def __init__(self, space, inputs=[], aspect=4):
        """
        """
        FormulationR2.__init__(self, space, inputs)
        self.B = Variable(shape=len(inputs), pos=True, name='aspect')

    def as_constraint(self, *args):
        fp = self._inputs[0]
        num_in = len(self._inputs)
        A_i = np.asarray([x.area for x in self._inputs])

        # SDP to enforce aspect constraints
        W, H = fp.W, fp.H
        C = [cvx.PSD(cvx.bmat([[self.B[i], W[i]],
                               [W[i], A_i[i]]])) for i in range(num_in)]

        C += [cvx.PSD(cvx.bmat([[self.B[i], H[i]],
                                [H[i], A_i[i]]])) for i in range(num_in)]
        return C

    def as_objective(self, **kwargs):
        """ minimize the aspects """
        return cvx.Minimize(cvx.sum(self.B))


# Stage 1 ----------------------------------------------------------------
class PlaceCirclesAR(FormulationR2):
    creates_var = True

    def __init__(self, domain, children,
                 cij=None,
                 epsilon=1,
                 width=None,
                 height=None,
                 min_edge=None):
        """
        todo - this is returning non-dcp - since this is its own step - maybe implement in str8 CVX
        statement
        ''

        Attractor-Repellor placement Stage 1.

        reference:
            Large-Scale Fixed-Outline Floorplanning Design Using Convex Optimization Techniques

        children list of objects with interfaces to
            - Parameter: Area
            - Variables: x, y, w, h
        """
        FormulationR2.__init__(self, domain, children)
        self._epsilon = epsilon
        self._sigma = 1e6
        self._k = 1
        num_in = len(self._inputs)
        print('num_in', num_in)
        self.Cij = cij if cij is not None else np.ones((num_in, num_in))
        print(self.Cij)
        self.WF = width if width else Variable(pos=True, name='Stage1.W')
        self.HF = height if height else Variable(pos=True, name='Stage1.H')
        self.X = Variable(shape=num_in, pos=True, name='Stage1.X')
        self.Y = Variable(shape=num_in, pos=True, name='Stage1.Y')

        # additional
        self._smallest_edge = min_edge if min_edge else 2
        self.Dij = None
        self.Tij = None
        self.obj = None

    @property
    def num_actions(self):
        return self.X.shape[0]

    def _area_to_rad(self, area, min_a):
        """"""
        phi = self._smallest_edge
        return np.sqrt(area / np.pi) * np.log(1 + area / (min_a - phi ** 2))

    def _target_dist_i(self, ):
        return

    def as_constraint(self, *args):
        """

        """
        areas = np.asarray([x.area for x in self._inputs])
        R_i = self._area_to_rad(areas, areas.min())

        Dij = []    # actual distance
        Tij = []    # target distance
        Cij = []
        ij = 0
        for i in range(self.num_actions):
            for j in range(i + 1, self.num_actions):
                tij = (R_i[i] + R_i[j]) ** 2
                tgt_sq = np.sqrt(tij / (self.Cij[i, j] + self._epsilon))
                # print(tij, tgt_sq)
                if tij >= tgt_sq:
                    Tij.append(tij)
                else:
                    Tij.append(tgt_sq)
                Cij.append(self.Cij[i, j])
                dij = cvx.square(self.X[i] - self.X[j]) + cvx.square(self.Y[i] - self.Y[j])
                Dij.append(dij)
                ij += 1
        # t_ij = sigma ( a_i^0.5 + a_j ^ 0.5 )
        # c_ij = connectivity   (input)
        C = [
            self.X + R_i <= 0.5 * self.WF,
            self.X + R_i <= 0.5 * self.WF,

            R_i - self.Y <= 0.5 * self.HF,
            R_i - self.Y <= 0.5 * self.HF,
        ]
        # T_ij = sqrt( t_ij / (c_ij + eps))
        # D_ij = (x i − x j ) 2 + (y i − y j ) 2
        Dij = cvx.hstack(Dij)
        Tij = np.asarray(Tij)
        Cij = np.asarray(Cij)

        # print(Tij)
        # F(xi, xj, yi, yj)
        f1 = Cij * Dij + Tij / Dij - 1 # Tij / Dij is not convex ?
        f2 = 2 * np.sqrt(Cij * Tij) - 1
        # print('\nf2\n', f2, '\n--')
        # Kln(Dij / t_ij)
        f3 = self._k * cvx.log(Dij / Tij)
        # f3 = self._k * Dij / Tij

        z1 = Variable(shape=self.num_actions, boolean=True)
        z2 = Variable(shape=self.num_actions, boolean=True)
        # e = self.WF
        # F = z1 * f1 + z2 * f2
        print('-----------------------------')
        print(Dij.is_convex())
        print(cvx.inv_pos(Dij).is_atom_convex())
        print(cvx.inv_pos(Dij).is_convex())
        print(cvx.inv_pos(Dij).is_concave())
        print(f1.is_convex())
        print(f1.is_concave())
        print('-----------------------------')
        C += [
            z1 + z2 <= 1,
            z1 + z2 >= 0,
            # Dij >=  Tij then z1 goes to 1
            # if Dij - Tij >= 0  then z1 = 1
            # if Dij - Tij  < 0  then z1 = 0
            # 0 <= Dij - Tij + 1e4 * (1 - z1),
            # F <= Dij - Tij + 1e4 * (1 - z1)
            # f3 <= f2
        ]
        from cvxpy import linearize

        self.obj = Minimize(cvx.sum(f1))
        # self.obj = Minimize(cvx.sum(f2 -  linearize(f3)))
        # print(self.obj)
        # self.obj = Minimize(cvx.sum(F - f3))
        self.Dij = Dij
        self.Tij = Tij
        return C

    def as_objective(self, **kwargs):
        """
        b_0 + f_0(x)                    for 0 <= x < 10,
        g(x) = b_0 + f_0(10) + f_1(x)   for 10 <= x <= 100

        ----------------
        minimize     b_0 + f_0(z_0) + f_1(z_1)

        subject to   10y_1  <= z_0
                     z_0    <= 10y_0

                     0     <= z_1 <= 90y_1
                     x      = z_0 + z_1

                     y_0, y_1 in {0,1}
        """
        return self.obj

    def compute_rpm(self):
        return

    def display(self):
        data = FormulationR2.display(self)
        return

    @property
    def action(self):
        """ output is RPM """
        return cvx.hstack([self.X, self.Y])


# stage2 --------------------------------------------------------------
class FPStage2(FormulationR2):
    def __init__(self, domain, children, **kwargs):
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
        FormulationR2.__init__(self, domain, children)
        num_in = len(children)
        self.X = Variable(shape=num_in, pos=True, name='Stage2.x')
        self.Y = Variable(shape=num_in, pos=True, name='Stage2.y')
        self.W = Variable(shape=num_in, pos=True, name='Stage2.w')
        self.H = Variable(shape=num_in, pos=True, name='Stage2.h')

    @property
    def num_actions(self):
        return self.X.shape[0]

    @property
    def outputs(self):
        return self.X, self.Y, self.W, self.H

    def display(self):
        display = FormulationR2.display(self)
        X, Y, W, H = self.X.value, self.Y.value, self.W.value, self.H.value
        for i in range(self.num_actions):
            datum = dict(x=X[i] - 0.5 * W[i], y=Y[i] - 0.5 * H[i], w=W[i], h=H[i], index=i)
            display['boxes'].append(datum)
        return display


class PlaceLayoutSDP(FPStage2):
    def __init__(self, space, children, rpm,
                 adj_mat=None,
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
        FPStage2.__init__(self, space, children)
        num_in = self.X.shape[0]
        tris = np.triu(np.ones((num_in, num_in), dtype=int), 1).sum()
        self.B = Variable(shape=self.X.shape[0], pos=True, name='aspect')
        self.U = Variable(shape=tris, pos=True)
        self.V = Variable(shape=tris, pos=True)

        # children
        self.aspect = NumericBound(self.B, high=aspect)
        self.rpm = RPM(self.space, self, rpm)
        self.bnds = BoundsXYWH(self.space, self, width, height)

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


# deprecated / not needed ----------------------------------------
class _BoundsXYWH(FormulationR2):
    META = {'constraint': True}

    def __init__(self, space, layout, w, h, **kwargs):
        FormulationR2.__init__(self, space, [layout])
        self.w = w
        self.h = h

    def as_constraint(self):
        fp = self._inputs[0]
        C = [fp.X + 0.5 * fp.W <= 0.5 * self.w,   # 0.5 * self.w,
             fp.Y + 0.5 * fp.H <= 0.5 * self.h,   # 0.5 * self.h,

             0.5 * fp.W - fp.X <= 0.5 * self.w,        # 0.5 * self.w,
             0.5 * fp.H - fp.Y <= 0.5 * self.h,        # 0.5 * self.h,
             # fp.H >= 0.2,
             # fp.W >= 0.2
             # fp.X - 0.5 * fp.W >= 0,
             # fp.Y - 0.5 * fp.H >= 0,
             ]
        return C

    def as_objective(self, **kwargs):
        return None


class PlaceLayout(FPStage2):
    def __init__(self, space, children,
                 adj_mat,
                 rpm,
                 aspect=4,
                 width=None,
                 height=None,
                 lims=None):
        """ Stage 2 of layout problem
        from
        An Efficient Multiple-stage Mathematical Programming Method for
        Advanced Single and Multi-Floor Facility Layout Problems

            - Fixed Outline (can work for classical as well)
            -

        """
        FPStage2.__init__(self, space, children)
        self._rpm = rpm
        self._adj_mat = adj_mat
        self._lims = lims
        self._aspect = aspect
        num_in = len(self._inputs)

        self.WF = width if width else Variable(nonneg=True, name='Stage2.W')
        self.HF = height if height else Variable(nonneg=True, name='Stage2.H')

        tris = np.triu(np.ones((num_in, num_in)), 1).sum()
        self.U = Variable(shape=tris, nonneg=True)
        self.V = Variable(shape=tris, nonneg=True)

    def as_constraint(self, *args):
        """

        """
        areas = np.asarray([x.area for x in self._inputs])

        # Area constraints -> can be SDP relaxed
        C = [self.X * self.Y == areas ]                 # 28

        # aspect constraints -> can be SDP relaxed
        C += [
            self.W - self._aspect * self.H <= 0,
            self.H - self._aspect * self.W <= 0,  # 29
        ]

        # common within facility half-bounds
        C += [
            self.X + 0.5 * self.W <= 0.5 * self.WF,
            self.Y + 0.5 * self.H <= 0.5 * self.HF,
            0.5 * self.W - self.X <= 0.5 * self.WF,
            0.5 * self.H - self.Y <= 0.5 * self.HF,     # 26, 27
        ]
        if self._lims:
            C += [
                # within bounds - todo factor out
                self._lims[0] <= self.W,
                self._lims[2] >= self.W,
                self._lims[1] <= self.H,
                self._lims[3] >= self.H,  # 30
            ]

        ij = 0
        for i in range(len(self._inputs)):
            for j in range(i+1, len(self._inputs)):
                C += [
                    self.U[ij] >= self.X[i] - self.X[j],    # 22
                    self.U[ij] >= self.X[j] - self.X[i],    # 23
                    self.V[ij] >= self.Y[i] - self.Y[j],    # 24
                    self.V[ij] >= self.Y[j] - self.Y[i],    # 25
                ]
                ij += 1
        return C

    def as_objective(self, **kwargs):
        O = []
        ij = 0
        for i in range(len(self._inputs)):
            for j in range(i+1, len(self._inputs)):
                O += [
                    self._adj_mat[i, j] * (self.U[ij] + self.V[ij]),  # 21
                ]
                ij += 1
        return Minimize(cvx.sum(cvx.hstack(O)))

    @property
    def action(self):
        geom = cvx.hstack([self.X, self.Y, self.W, self.H])
        return geom


class PlaceLayoutSOC(FPStage2):
    def __init__(self, space, children,
                 adj_mat,
                 rpm,
                 aspect=4,
                 width=None,
                 height=None,
                 lims=None):
        """
        RPM: upper tri trensor [n x n x 2]
            [0, 0] -> no relation

        """
        FPStage2.__init__(self, space, children)
        self.rpm = rpm
        num_in = self.X.shape[0]
        tris = np.triu(np.ones((num_in, num_in)), 1).sum()

        self.U = Variable(shape=tris, nonneg=True)
        self.V = Variable(shape=tris, nonneg=True)
        self.B = Variable(shape=self.X.shape[0], nonneg=True)

    def as_constraint(self):
        """
        """
        num_in = self.X.shape[0]
        tri_i, tri_j = np.triu_indices(num_in, 1)
        tri_i, tri_j = tri_i.tolist(), tri_j.tolist()

        A_i = np.asarray([x.area for x in self._inputs])
        a2s = 2 * np.sqrt(A_i)
        C = [
            # lineaerized absolute value constraints
            self.U >= self.X[tri_i] - self.X[tri_j],
            self.U >= self.X[tri_j] - self.X[tri_i],
            self.V >= self.Y[tri_i] - self.Y[tri_j],
            self.V >= self.Y[tri_j] - self.Y[tri_i],

            # SOC constraints on B
            cvx.SOC(self.H - self.W + a2s,      self.H + self.W),
            cvx.SOC(A_i - self.B + 2 * self.H,  self.B + A_i),
            cvx.SOC(A_i - self.B + 2 * self.W,  self.B + A_i),
        ]
        # RPM adjacency constraints
        C += RPM(self.space, self, self.rpm).as_constraint()
        return C

    def as_objective(self, **kwargs):
        return Minimize(cvx.sum(self.U + self.V))


class PlaceLayoutGM(FPStage2):
    def __init__(self, space, children, rpm,
                 adj_mat=None,
                 aspect=4,
                 width=None,
                 height=None,
                 lims=None):
        """
        RPM: upper tri trensor [n x n x 2]
            [0, 0] -> no relation
        Novel Convex Optimization Approaches
                for VLSI Floorplanning                      2008    SDP
        """
        FPStage2.__init__(self, space, children)
        num_in = self.X.shape[0]
        tris = np.triu(np.ones((num_in, num_in), dtype=int), 1).sum()
        self.B = Variable(shape=self.X.shape[0], pos=True)
        self.U = Variable(shape=tris, pos=True)
        self.V = Variable(shape=tris, pos=True)
        self.WF = width
        self.HF = height
        self.rpm = RPM(self.space, self, rpm)

    def as_constraint(self):
        """
        """
        num_in = self.X.shape[0]
        A_i = np.asarray([x.area for x in self._inputs])
        a2s = np.sqrt(A_i)
        tris = np.triu(np.ones((num_in, num_in), dtype=int), 1).sum()
        tri_i, tri_j = np.triu_indices(num_in, 1)
        tri_i, tri_j = tri_i.tolist(), tri_j.tolist()
        # A_i = np.asarray([x.area for x in self._inputs])
        a2s = np.sqrt(A_i)
        C = [
            # lineaerized absolute value constraints
            self.U >= self.X[tri_i] - self.X[tri_j],
            self.U >= self.X[tri_j] - self.X[tri_i],
            self.V >= self.Y[tri_i] - self.Y[tri_j],
            self.V >= self.Y[tri_j] - self.Y[tri_i]
        ]
        # SDP constraints
        # a = w * h
        C += [cvx.geo_mean(cvx.hstack([self.W[i], self.H[i]])) >= a2s[i] for i in range(num_in)]

        # a = w * h
        # C += [cvx.PSD(cvx.bmat([[self.B[i], self.W[i]],
        #                        [self.W[i], A_i[i]]]), constr_id='bw{}'.format(i)) for i in range(num_in)]

        # C += [cvx.PSD(cvx.bmat([[self.B[i], self.H[i]],
        #                         [self.H[i], A_i[i]]]), constr_id='hw{}'.format(i)) for i in range(num_in)]

        # C += [self.B <= 4]
        # RPM adjacency constraints
        C += self.rpm.as_constraint()

        # within bounds
        # C += BoundsXYWH(self.space, self, self.WF, self.HF).as_constraint()
        return C

    def as_objective(self, **kwargs):
        tri_i, tri_j = np.triu_indices(self.X.shape[0], 1)
        tri_i, tri_j = tri_i.tolist(), tri_j.tolist()
        o1 = Minimize(cvx.sum(self.U + self.V))
        # o2 = Minimize(cvx.sum(cvx.abs(self.X[tri_i] - self.X[tri_j])
        #                      + cvx.abs(self.Y[tri_i] - self.Y[tri_j])
        #                      ))
        return o1

