from .fp_cont import FormulationR2
from cvxpy import Variable, Minimize, Parameter
import numpy as np
import networkx as nx
import cvxpy as cvx



class AdjacencyMatrix(FormulationR2):
    def __init__(self, mat, inputs=None, constants=None, min_overlap=None, points=None, **kwargs):
        """
        todo - not sure about this in SDP context - aka its fucked.
        overlap has to added directly into the sdp constraint
        MIP Based -
        the overlaps between rooms must be of size c

        x i       ≤ x j + w j − c · θ i, j
        x i + w i ≥ x j + c · θ_i,j
        y i       ≤ y j + d j − c · (1 − θ i, j )
        y i + d i ≥ y j + c · (1 − θ i, j ),
        """
        FormulationR2.__init__(self, inputs, **kwargs)
        self.mat = mat
        self.min_overlap = min_overlap
        self.constants = constants

    def as_constraint(self, **kwargs):
        X, Y, W, H = self.inputs.vars
        n = self.mat.shape[0]
        theta_ij = Variable()
        for r, (i, j) in enumerate(zip(*np.triu_indices(n, 1))):
            pass
        return


class RPM(FormulationR2):
    META = {'constraint': True, 'objective': False}

    def __init__(self, mat, inputs=None, constants=None, min_overlap=None, points=None, **kwargs):
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
        FormulationR2.__init__(self, inputs, **kwargs)
        self.mat = mat
        self.min_overlap = min_overlap
        self.constants = constants
        self.points = points.tolist()

    @property
    def num_actions(self):
        return self.mat.shape[0]

    @classmethod
    def points_to_mat(cls, points):
        """

        :param points:
        :return: tensor[node_i, node_j, [lr, ud, voronoi_distance]]
        """
        # from scipy.spatial import Voronoi
        n = len(points)
        if isinstance(points, (list, tuple)):
            points = np.asarray(points)
        # vor = Voronoi(points)
        # dm = np.zeros((n, n), dtype=int)
        rpm = np.zeros((n, n, 2), dtype=int)
        for r, (i, j) in enumerate(zip(*np.triu_indices(n, 1))):
            pi, pj = points[i], points[j]
            dx, dy = np.abs(pi - pj)
            if dx >= dy:
                # Left/right
                if pi[0] >= pj[0]:
                    rpm[i, j, 0] = 2
                else:
                    rpm[i, j, 0] = 1    # i rght_of j
            else:
                # Up/down
                if pi[1] >= pj[1]:
                    rpm[i, j, 1] = 1    # i above j
                else:
                    rpm[i, j, 1] = 2    # i below j
        return rpm

    @classmethod
    def from_points(cls, pnts, **kwargs):
        if isinstance(pnts, (list, tuple)):
            pnts = np.asarray(pnts)
        return cls(cls.points_to_mat(pnts), points=pnts, **kwargs)

    @classmethod
    def to_graph(cls, mat):
        # import networkx as nx
        # mat = self.mat.shape[0]
        Gs = []
        for dim in range(mat.shape[2]):
            G = nx.DiGraph()
            for i, j in np.stack(np.where(mat[:, :, dim] == 1)).T.tolist():
                G.add_edge(i, j)
            for i, j in np.stack(np.where(mat[:, :, dim] == 2)).T.tolist():
                G.add_edge(j, i)
            Gs.append(G)
        return Gs

    @classmethod
    def expr(cls, index, value, xi, xj, di, dj, offs=0):
        """ generate expression for a single RPM entry

            one of {xi, xj, di, dj} must be a Variable

            - index (int) {0, 1} - lh/ud axis of rpm
            - value (int) {0, 1, 2} - value of rpm at [i, j, index]
            - xi: Variable or float corresponding to X_i or Y_i
            - xj: Variable or float corresponding to X_i or Y_i
            - di: Variable or float corresponding to W_i or H_i
            - dj: Variable or float corresponding to W_i or H_i
        """
        if index == 0:
            if value == 1:
                return [xj - xi >= 0.5 * (di + dj)]
            elif value == 2:
                return [xi - xj >= 0.5 * (di + dj)]
        elif index == 1:
            if value == 1:
                return [xi - xj >= 0.5 * (di + dj)]
            elif value == 2:
                return [xj - xi >= 0.5 * (di + dj)]
        return []

    def as_constraint(self):
        """
        todo - add offsets . Offset limits imply 'doors' otherwise
        """
        in_size, mat = len(self.inputs), self.mat
        assert in_size == mat.shape[0]
        assert in_size == mat.shape[1]

        C = []
        x, y, w, h = self.inputs.vars
        for i, j in np.stack(np.where(mat[:, :, 0] == 1)).T.tolist():
            # i left_of j
            C += [x[j] - x[i] >= 0.5 * (w[i] + w[j])]

        for i, j in np.stack(np.where(mat[:, :, 0] == 2)).T.tolist():
            C += [x[i] - x[j] >= 0.5 * (w[i] + w[j])]

        for i, j in np.stack(np.where(mat[:, :, 1] == 1)).T.tolist():
            C += [y[i] - y[j] >= 0.5 * (h[i] + h[j])]

        for i, j in np.stack(np.where(mat[:, :, 1] == 2)).T.tolist():
            C += [y[j] - y[i] >= 0.5 * (h[i] + h[j])]

        return C

    def as_objective(self, **kwargs):
        return None

    def describe(self, arr=None, text=None):
        """ text of relations between RPM entries """
        if arr is None and text is None:
            text = True
        txtlr = ''
        txtud = ''
        cnt = 0
        num = self.num_actions
        arr = [[''] * num for i in range(num)]
        for i, j in zip(*np.triu_indices(num, 1)):
            lr, ud = self.mat[i, j]
            s = ''
            if lr in [1, 2]:
                cnt += 1
                s += str(lr)
            if lr == 1:
                txtlr += '\n{} left of {}'.format(i, j)
            elif lr == 2:
                txtlr += '\n{} rght of {}'.format(i, j)
            else:
                s += '0'
            if ud in [1, 2]:
                cnt += 1
                s += str(ud)
            if ud == 1:
                txtud += '\n{} above   {}'.format(i, j)
            elif ud == 2:
                txtud += '\n{} below   {}'.format(i, j)
            else:
                s += '0'
            arr[i][j] = s
        if text:
            return 'num_{}\n{}\n{}'.format(cnt, txtlr, txtud)
        return [' '.join(x) for x in arr]


class ShiftMatrix(FormulationR2):
    META = {'constraint': True, 'objective': False}

    def __init__(self, mat, inputs=None, zdim=None, **kwargs):
        """
        Transform the problem -> if
        """
        FormulationR2.__init__(self, inputs, **kwargs)
        # assert mat.sum(axis=(-1)) <= 1, 'matrix contains non-mutex entries'
        self.mat = mat
        self.zdim = zdim

    @property
    def num_actions(self):
        return self.mat.shape[0]

    @classmethod
    def update_matrix(cls, mat):
        """
        for each adjacency relationship i,j replace it with two new relationships: i,k, k,j
        (removing original relationship)
        modifies the size of matrix

        Returns:
            np.array.shape[ mat.shape[0] + (mat > 0).sum(), mat.shape[0] + (mat > 0).sum(), 2]
        """
        in_size = mat.shape[0]
        num_adj = (mat > 0).sum()
        new_mat = np.zeros((in_size + num_adj, in_size + num_adj, 2))
        new_mat[:in_size, :in_size, :] = np.copy(mat)
        cnt = num_adj
        for dim in range(mat.shape[-1]):
            for i, j in zip(*np.triu_indices(in_size, 1)):
                value = new_mat[i, j, dim]
                if value == 0:
                    continue
                new_mat[i, cnt, dim] = value
                new_mat[cnt, j, dim] = value
                new_mat[i, j, dim] = 0
                cnt += 1
        return new_mat

    def as_constraint(self):
        mat = self.mat
        in_size = len(self.inputs)
        C = []
        x, y, w, h = self.inputs.vars
        num_adj = (self.mat > 0).sum()
        print('num_adj', num_adj)
        new_mat = np.zeros((in_size + num_adj, in_size+ num_adj, 2))
        new_mat[:in_size, :in_size, :] = np.copy(mat)

        cnt = num_adj
        for i, j in np.stack(np.where(mat[:, :, 0] == 1)).T.tolist():
            # define Z_ij
            Xz_ij = Variable(pos=True)
            Yz_ij = Variable(pos=True)
            Wz_ij = self.zdim
            Yz_ij = Variable(pos=True)

            C += [
                #
            ]
            pass

        for i, j in np.stack(np.where(mat[:, :, 0] == 2)).T.tolist():
            pass

        for i, j in np.stack(np.where(mat[:, :, 1] == 1)).T.tolist():
            pass

        for i, j in np.stack(np.where(mat[:, :, 1] == 2)).T.tolist():
            pass
        return C

    def as_objective(self, **kwargs):
        return None

    def describe(self, arr=None, text=None):
        """ text of relations between RPM entries """
        txt = ''
        num = self.num_actions
        arr = [[''] * num for i in range(num)]
        for i, j in zip(*np.triu_indices(num, 1)):
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
    def __init__(self, mat, is_sparse=False, **kwargs):
        """
        Sparse Relative Position Matrix
        before calling self.as_constraint(), removes redundant entries
        """
        RPM.__init__(self, mat, **kwargs)
        self.base = mat     # keep a copy of the base matrix
        if is_sparse is True:
            self.mat = self.sparsify(mat)
        else:
            self.mat = mat

    @classmethod
    def sparsify(cls, old_mat):
        """ if A is left of B and B is to left of C -> A is left of C
        => i,j = left
           j,k = left
           i,k => 0

        i,j = left
        j,k = right :  k,j = left
        ------
         - i,k = right =>
         - i,k = left =>
         - i,k = None =>

        scan matrix
        """
        mat = np.copy(old_mat)
        size = mat.shape[0]
        # ds = [(0, 1, 2), (0, 2, 1), (1, 1, 2), (1, 2, 1)]
        # for i in range(mat.shape[0]):
        #     for ax, rel_num, rel_inv in ds:
        #         # find all instances of
        #         rel_i = np.where(old_mat[i, :, ax] == rel_num)[0].tolist()
        #         for j in rel_i:
        #             rel_j1 = np.where(old_mat[j, :, ax] == rel_num)[0].tolist()
        #             # rel_j2 = np.where(old_mat[j, :, ax] == rel_inv)[0].tolist()
        #             mat[i, list(set(rel_i).intersection(rel_j1)), ax] = 0
        #             # mat[i, list(set(rel_i).intersection(rel_j2)), ax] = 0
        GS = cls.to_graph(mat)
        # todo nx.dag_longest_path()
        for dim in [0, 1]:
            G = GS[dim]
            # nx.pri
            for i in range(size):
                # paths exit from i to {j, k}
                # if there a path to from j to k -> eliminate k
                for j in range(i+1, size):
                    v_ij = mat[i, j, dim]
                    if v_ij == 0:
                        continue
                    for k in range(j+1, size):
                        v_ik = mat[i, j, dim]
                        if v_ik == 0 or v_ij != v_ik:
                            # if i above j and i below k, no need to check
                            continue
                        if nx.has_path(G, k, j) is True:
                            # i above j
                            mat[i, k, dim] = 0
                            G.remove_edge(i, k)
                            break
                        elif nx.has_path(G, j, k) is True:
                            mat[i, j, dim] = 0
                            G.remove_edge(i, j)
                            break
        return mat


class PackCircles(FormulationR2):
    META = {'constraint': True, 'objective': True}

    def __init__(self, circle_list,
                 cij=None,
                 obj='ar0',
                 overlap=False,
                 min_edge=2,
                 verbose=False, **kwargs):
        """
        Weighted Circle Packing Problem
            k -
            eps -
            min_edge -
        """
        FormulationR2.__init__(self, circle_list, **kwargs)

        # solver and record args
        self._solve_args = {'method': 'dccp'}
        self._in_dict = {'k': 1, 'min_edge': min_edge, 'eps': 1e-2}

        # gather inputs
        X = circle_list.point_vars
        n = len(circle_list)
        cij = cij if cij is not None else np.ones((n, n))

        # compute radii
        areas = np.asarray([x.area for x in circle_list.inputs])
        r = np.sqrt(areas / np.pi) # * np.log(1 + areas / (min_area - min_edge ** 2))
        self.r = r

        # indices of upper tris
        inds = np.triu_indices(n, 1)
        xi, xj = [x.tolist() for x in inds]

        # gather inputs
        weights = Parameter(shape=len(xi), value=cij[inds], name='cij', nonneg=True)
        radii = Parameter(shape=len(xi), value=r[xi] + r[xj], name='radii', nonneg=True)
        dists = cvx.norm(X[xi, :] - X[xj, :], 2, axis=1)

        # constraints
        self._constr.append(dists >= radii)

        # objective
        self._obj = Minimize(cvx.sum(cvx.multiply(weights, dists)))

    def _xxx(self, obj):
        if obj == 'rect':
            self._obj = Minimize(
                    cvx.max(cvx.abs(X[:, 0]) + r) +
                    cvx.max(cvx.abs(X[:, 1]) + r)
                )
        elif obj == 'sqr':
            self._obj = Minimize(cvx.max(cvx.max(cvx.abs(X), axis=1) + r))
        else:
            # f0 =
            # f1 = Tij / Dij #  - 1 # Tij / Dij is not convex ?
            # f1 = 1 / Dij     # , -1)  # - 1
            # vbar = Variable(n, boolean=True)
            # f2 = 2 * cvx.sqrt(Cij * Tij) - 1
            # f3 = Dij / Tij  # * self._k
            # if verbose is True:
            #     print('dij', Dij.curvature)
            #     print('f0', f0.curvature)
            #     print('f1', f1.curvature)
            #     print('f3', f3.curvature)

            if obj == 'ar0':
                print('ar0', f0.shape, cvx.sum(f0).curvature)


            # elif obj == 'ar1':
            #     self._obj = Minimize(cvx.sum(f1))
            #
            # elif obj == 'ar3':
            #     self._obj = Minimize(cvx.sum(f3))
            #
            # elif obj == 'ar01':
            #
            #     self._obj = Minimize(cvx.sum(f0)) + Minimize(cvx.sum(f1))
            # elif obj == 'ar03':
            #     self._obj = Minimize(cvx.sum(f0 - f3))
            # elif obj == 'arf':
            #     self._obj = Minimize(cvx.sum(f0 + f1 + f2 - f3))
            # else:
            #     raise Exception('unknown objective ! ')

    @property
    def vars(self):
        cx, _ = self.inputs.vars
        return cx

    @property
    def outputs(self):
        from .input_structs import CircleList
        cx, _ = self.inputs.vars
        inputs = self.inputs.inputs
        return CircleList(inputs, x=cx, r=self.r)

    def as_constraint(self, **kwargs):
        return self._constr

    def as_objective(self, **kwargs):
        return self._obj


class EuclideanDistanceMatrix(FormulationR2):
    META = {'constraint': True, 'objective': True}

    def __init__(self, point_list, norm=2, **kwargs):
        FormulationR2.__init__(self, point_list, **kwargs)
        n = len(point_list)
        X = point_list.point_vars
        print(X.shape )
        ti, tj = [x.tolist() for x in np.triu_indices(n, 1)]
        self._expr = cvx.norm(X[ti] - X[tj], norm, axis=1)

    def as_constraint(self, **kwargs):
        pl = self._inputs
        return [pl.point_vars >= 0, pl.point_vars <= 100]

    def as_objective(self):
        if self._obj_type is not None:
            if self._obj_type == cvx.Maximize:
                return self._obj_type(cvx.sum(self._expr))
            else:
                return self._obj_type(cvx.sum(self._expr))
        else:
            Exception('no objective type specified')


class NoOvelapMIP(FormulationR2):
    META = {'constraint': True, 'objective': False}

    def __init__(self, box_list, others=None, m=100, **kwargs):
        """

        if others is specified then, items in box_list will not overlap items in others list
        """
        FormulationR2.__init__(self, box_list, **kwargs)
        self._m = m
        self._others = others

    def from_2lists(self):
        # to do
        in1 = self.inputs
        in2 = self._others

        xi, xj = len(in1), len(in2)
        ntril = len(xi)
        trili = list(range(ntril))
        or_vars = Variable(shape=(ntril, 4), boolean=True, name='overlap_or({},{})')
        C = [
            in1.right <= in2.left[xj] + self._m * or_vars[trili, 0],
            in2.right[xj] <= in1.left + self._m * or_vars[trili, 1],
            X.top[xi] <= X.bottom[xj] + self._m * or_vars[trili, 2],
            X.top[xj] <= X.bottom[xi] + self._m * or_vars[trili, 3],
            cvx.sum(or_vars, axis=1) <= 3
        ]
        return

    def as_constraint(self, **kwargs):
        """
           http://yetanothermathprogrammingconsultant.blogspot.com/2017/07/rectangles-no-overlap-constraints.html
           xi+wi ≤ xj or
           xj+wj ≤ xi or
           yi+hi ≤ yj or
           yj+hj ≤ yi

           is transfromed to linear inequalities
        """
        X = self.inputs
        N = len(X)

        xi, xj = [x.tolist() for x in np.triu_indices(N, 1)]
        ntril = len(xi)
        trili = list(range(ntril))
        or_vars = Variable(shape=(ntril, 4), boolean=True, name='overlap_or({},{})')
        C = [
            X.right[xi] <= X.left[xj]   + self._m * or_vars[trili, 0],
            X.right[xj] <= X.left[xi]   + self._m * or_vars[trili, 1],
            X.top[xi]   <= X.bottom[xj] + self._m * or_vars[trili, 2],
            X.top[xj]   <= X.bottom[xi] + self._m * or_vars[trili, 3],
            cvx.sum(or_vars, axis=1) <= 3
        ]
        return C


# ------------------------------------------------------------------------
# NOT USED / FAIL
# ------------------------------------------------------------------------
class GramMatrix(FormulationR2):
    def __init__(self, point_list, **kwargs):
        """
        pg 419 8.3.1 Gram matrix and realizability
        d ij =

        = ka i − a j k 2
        = (l i 2 + l j 2 − 2a Ti a j ) 1/2
        = (l i 2 + l j 2 − 2G ij ) 1/2 .
        :param point_list:
        :param kwargs:
        """
        FormulationR2.__init__(self, point_list, **kwargs)
        n = len(point_list)
        X = point_list.X
        z = np.asarray([x.area for x in point_list.inputs])
        xi, xj = [x.tolist() for x in np.triu_indices(n, 1)]
        # print(X.shape)
        # print(xi, xj)
        # dij = cvx.norm(X[xi] - X[xj], 2, axis=1)
        G = Variable(shape=(n, n), PSD=True)
        C = [  ]




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
        f1 = Cij * Dij + Tij / Dij - 1      # Tij / Dij is not convex ?
        f2 = 2 * np.sqrt(Cij * Tij) - 1

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

    def display(self):
        data = FormulationR2.display(self)
        return

    @property
    def action(self):
        """ output is RPM """
        return cvx.hstack([self.X, self.Y])


class QuadraticPlacement(FormulationR2):
    creates_var = True

    def __init__(self, domain, children,
                 cij=None,
                 epsilon=1,
                 width=None,
                 height=None,
                 min_edge=None):
        """
        Adjacency Matrix -> Relative Placement Matix

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
