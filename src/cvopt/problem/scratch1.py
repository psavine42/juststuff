from .base import FPProbllem
from cvxpy import Variable, Minimize, Problem, Maximize
import numpy as np
import cvxpy as cvx
import itertools


class MIPFloorPlan(FPProbllem):
    """
    polyomino problem
    http://yetanothermathprogrammingconsultant.blogspot.com/2017/12/filling-rectangles-with-polyominos.html

    https://nbviewer.jupyter.org/github/MOSEK/Tutorials/blob/master/exact-planar-cover/exactcover.ipynb

    PLACEMENT ON HALF VERTICES

    Attributes
    ------------
        - n: (int) width of mesh    -> rows
        - m: (int) height of mesh   -> cols
        - t: (int) number of templates

        - templates: lsit of shapes that can be used

    Variables
    -------------
        - X:
        - areaas:
    """
    def __init__(self, templates, n, m, limits=None):
        FPProbllem.__init__(self)
        self.templates = templates
        self.n = n
        self.m = m
        self.t = len(templates)

        self.num_var = n * m * self.t
        self.num_con = n * m

        # limits
        self.limits = _pre_process_tile_limit(limits, self.t, self.num_con)
        print(self.limits)
        self.sinks = []

        # Variables
        self.X = Variable(shape=self.num_var, integer=True)

        # constraint definitiosn of problem
        self.A = np.zeros((self.num_con, self.num_var), dtype=int)
        self.areas = np.zeros(self.num_var, dtype=int)

    @property
    def solution(self):
        return self.X.value

    def anchorShape(self, shp, p, q, no_cov=[]):
        pts = [(p + x, q + y) for x, y in shp]
        if all(0 <= x and x < self.n and 0 <= y and y < self.m
               and (x, y) not in no_cov for x, y in pts):
            return pts
        else:
            return None

    def encode_nmt(self, x, y, l):
        return x * self.m * self.t + y * self.t + l

    def encode_nm(self, x, y):
        return x * self.m + y

    def vars_template(self, template_index):
        ixs = [i * self.m + template_index for i in range(self.num_con)]
        return self.X[ixs]

    def own_constraints(self):
        noncov = []
        B = np.ones(self.num_var)
        # for each shape set constraint matrix A
        for x, y, t in itertools.product(range(self.n), range(self.m), range(self.t)):
            pts = self.anchorShape(self.templates[t], x, y)
            bcode = self.encode_nmt(x, y, t)
            if pts is None:
                noncov.append(bcode)
            else:
                ar = np.asarray([self.encode_nm(x, y) for x, y in pts])
                self.A[ar, bcode] = 1           # when X_ijk = 1, points in A are set by mul
                self.areas[bcode] = len(pts)    # when X_ijk = 1 areas_ijk is part of objective
        #
        B[noncov] = 0

        # constraints
        base = [
            cvx.sum(cvx.reshape(self.X, (self.num_con, self.t)), axis=1) <= 1,
            self.A @ self.X <= 1,
            self.X <= B,
            self.X >= 0
        ]
        if self.limits is not None:
            vr = cvx.sum(cvx.reshape(self.X, (self.num_con, self.t)), axis=0) <= self.limits
            base.append(vr)
        return base

    def objective(self):
        objective = Maximize(cvx.sum(self.areas * self.X))
        return objective

    def display(self, problem, constraints=None):
        self.print(problem)
        colors = ['blue', 'yellow', 'green', 'red', 'violet', 'orange']
        sol = [(p, q, k) for p, q, k in itertools.product(range(self.n), range(self.m), range(self.t))
               if self.X.value[self.encode_nmt(p, q, k)] > 0.8]
        info = dict(edgecolor='black', linewidth=1)
        fig, ax = plt.subplots(1)
        # Plot all small squares for each brick
        for p, q, k in sol:
            print(p, q, k)
            for x, y in self.anchorShape(self.templates[k], p, q):
                ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor=colors[k], **info))

        ax.axis([0, self.n, 0, self.m])
        ax.axis('off')
        ax.set_aspect('equal')
        plt.show()


def _pre_process_tile_limit(limits, num_con, mx=40):
    if limits is None:
        return None
    elif isinstance(limits, (list, tuple)) and len(limits) == num_con:
        for i, l in enumerate(limits):
            if l is None:
                limits[i] = mx
        return np.asarray(limits, dtype=int)
    elif isinstance(limits, dict):
        for i, l in limits.items():
            if l is None:
                limits[i] = mx
        return np.asarray(limits, dtype=int)
    elif isinstance(limits, int):
        return np.asarray([limits for i in range(num_con)])
    raise Exception('limits must be same length as t')


class AdjPlan(FPProbllem):
    """
    PLACEMENT ON HALF EDGES

    modes:
    ------------------
        -
    """
    def __init__(self,
                 templates, mesh,
                 max_tiles=None,
                 min_tiles=None,
                 edges_forbidden=None,
                 faces_forbidden=None,
                 verts_forbidden=None,
                 ):
        FPProbllem.__init__(self)
        # prepare
        for i, t in enumerate(templates):
            if t.color is None:
                t.color = i

        self.T = templates
        self.t = len(templates)
        self.G = mesh

        self.num_vert = len(self.G.vertices()) * self.t
        self.num_hedg = len(self.G.half_edges()) * self.t
        self.num_edge = len(self.G.edges) # * self.t
        self.num_face = len(self.G.faces_to_vertices()) * self.t

        # number of potential actions is number of half edges * number of templates
        self.num_act = len(self.G.half_edges())
        self.num_var = self.num_act * self.t
        self.num_ihe = len(self.G.interior_half_edge_index()[0])
        self.num_color = len(set().union(*[x.colors for x in self.T]))

        # limits on how many times a tile can be used
        self.max_tile_uses = _pre_process_tile_limit(max_tiles, self.t, self.num_hedg)
        self.min_tile_uses = _pre_process_tile_limit(min_tiles, self.t, 0)

        # todo constraints
        # points where shape[i] cannot touch
        self.verts_forbidden = verts_forbidden if verts_forbidden else []
        self.edges_forbidden = edges_forbidden if edges_forbidden else []
        self.faces_forbidden = faces_forbidden if faces_forbidden else []

        # Variables
        self.X = Variable(shape=self.num_var, integer=True)
        self.J = Variable(shape=(self.num_edge, self.num_color), boolean=True)
        #
        # effect transformations
        # ------------------------------------------------
        # [num_(thing), num_var]
        self.Edges = np.zeros((self.num_edge, self.num_var), dtype=int)
        self.hE_ij = np.zeros((self.num_hedg, self.num_var), dtype=int)
        self.Verts = np.zeros((self.num_vert, self.num_var), dtype=int)
        self.Faces = np.zeros((self.num_face, self.num_var), dtype=int)

        self.Vert_Joint1 = np.zeros((self.num_hedg, self.num_var), dtype=int)
        self.Vert_Joint2 = np.zeros((self.num_hedg, self.num_var), dtype=int)

        # points where shape[i] must touch
        self.J_ij = np.zeros((self.num_color * self.num_edge, self.num_var), dtype=int)
        self.HE_Joint1 = np.zeros((self.num_act, self.num_var), dtype=float)
        self.HE_Joint2 = np.zeros((self.num_act, self.num_var), dtype=float)

        # areas
        self.areas = np.zeros(self.num_var, dtype=int)

        # -------------------------------------------------------------------
        self.FC2X = np.zeros((self.num_face, self.num_color, self.num_var), dtype=int)
        self.E2FC = np.zeros((self.num_edge, self.num_color, self.num_color, self.num_face), dtype=int)
        self.EC2X = np.zeros((self.num_edge, self.num_color, self.num_var), dtype=int)
        for k, vs in self.G.edges_to_faces().items():
            self.E2FC[k, :, :, list(vs)] = 1

        # computed during building constraints
        self._non_cov = []
        self._code_to_x = {}

    @property
    def solution(self):
        return self.X.value

    def encode_pos(self, *args):
        """ encode """
        if len(args) == 2:
            x, y = args
            return x * self.num_act + y

    def decode_pos(self, pos_index, m=2):
        if m == 2:
            return divmod(self.num_act, pos_index)
        elif m == 3:
            return

    def place_shape(self, t, half_edge_ix):
        result = []
        us, vs = self.G.get_half_edge(half_edge_ix).u
        for i, (p, q) in enumerate(self.T[t].half_edges()):

            # get the face bottom left for the half edge
            face_index = self.T[t].half_edges_to_faces()[(p, q)]
            face = self.T[t].get_face(face_index)
            bottom_left = sorted(face.verts)[0]
            result.append((us + bottom_left[0], vs + bottom_left[1]))
        return result

    def _add_joint_mapping(self, code, mapping, template):
        """
        see: Computing Layouts with Deformable Templates []

        ---------
        """
        indx = self.G.interior_half_edge_index()
        for k, v in mapping.items():

            ix = template.index_of_half_edge(k)
            _, _, he_data = template.half_edges_data()[ix]
            color = he_data.get('color', None)

            if color is None:
                continue
            this_ix = self.G.index_of_half_edge(v)
            edge_ix = self.G.half_edges_to_edges()[v]
            #
            self.J_ij[edge_ix * self.num_color + color, code] = 1

            if this_ix in indx[0]:
                self.HE_Joint1[this_ix, code] = color
                # self.HE_Joint1[this_ix, code, he_data['color']] = 1
            elif this_ix in indx[1]:
                self.HE_Joint2[this_ix, code] = color

            # self._add_face_color(edge_ix, color, code)
            # self.HE_Joint2[this_ix, code, he_data['color']] = 1
            # raise Exception('half edge ')
            # do not need an exception edge can be
            # """ if edge[i] has color C[c] then face[i] or face[j] should have color[c]"""
            self.EC2X[edge_ix, color, code] = 1
            self.FC2X[list(self.G.edges_to_faces()[edge_ix]), color, code] = 1

    def anchor_shape(self, code, t, half_edge_ix):
        """
            calculate if template of index (t)
            can be placed at half_edge of index (half_edge_ix)
        """
        us, vs = self.G.get_half_edge(half_edge_ix).u
        template = self.T[t]
        mapping = {}
        verts = []

        for p, q in template.half_edges():
            f1, f2 = ((us + p[0], vs + p[1]), (us + q[0], vs + q[1]))
            if (f1, f2) in self.G.half_edges():
                mapping[(p, q)] = (f1, f2)
                verts.append(f1)
                verts.append(f2)
            else:
                return False

        if all((start, end) not in self.edges_forbidden
               and start not in self.verts_forbidden
               and end not in self.verts_forbidden
               for start, end in mapping.values()):

            # if shape can be mapped here:
            # effects on constraints groups
            # note - these can contain duplicate indexes - numpy doesnt care
            he_ij_ixs = np.asarray([self.G.index_of_half_edge(x) for x in mapping.values()])
            edges_ixs = np.asarray([self.G.half_edges_to_edges()[x] for x in mapping.values()])
            faces_ixs = np.asarray([self.G.half_edges_to_faces()[x] for x in mapping.values()])
            verts_ixs = np.asarray([self.G.vertices_to_index()[x] for x in verts])

            self.Edges[edges_ixs, code] = 1
            self.Faces[faces_ixs, code] = 1
            self.hE_ij[he_ij_ixs, code] = 1
            self.Verts[verts_ixs, code] = 1
            self.areas[code] = template.weight * template.area()    #

            self._add_joint_mapping(code, mapping, template)
            return True
        return False

    def objective(self):
        """ """
        objective = Maximize(
            sum(self.areas @ self.X)
            # + sum(self.HE_Joint1 @ self.X + self.HE_Joint2 @ self.X)
            + sum(self.J)
        )
        return objective

    def joint_constraints(self):
        """
        if

        """
        ec2fc = np.reshape(self.E2FC, (self.num_edge * self.num_color, self.num_color * self.num_face))
        fc2x = np.reshape(self.FC2X, (self.num_face * self.num_color, self.num_var))
        Xvar_to_edge = ec2fc @ fc2x
        ec2x2 = np.reshape(self.EC2X, (self.num_edge * self.num_color, self.num_var))

        base = [

            # c1 - for each edge - select 0 to 1 colors for opt -OK
            cvx.sum(self.J, axis=1) <= 1,

            # c2 -
            # self.M_ec_ec @ self.J <= self.M_ec_x @ self.X,
            cvx.sum(self.J, axis=1) <= cvx.reshape(Xvar_to_edge, (self.num_edge, self.num_color)),

            # c3 -
            cvx.reshape(ec2x2 @ self.X, (self.num_edge, self.num_color))
               <= cvx.reshape(Xvar_to_edge @ self.X, (self.num_edge, self.num_color)) + 100 * (1 - self.J)
            # self.E2FC @ reshape(self.FC2X @ self.X, (self.num_face, self.num_color)) \
            #   <= reshape(self.EC2X @ self.X, (self.num_edge, self.num_color)) + 100 * (1 - self.J)
        ]
        return base

    def own_constraints(self):
        code = 0
        for t, half_edge_ix in itertools.product(range(self.t), range(self.num_act)):
            self._code_to_x[code] = (half_edge_ix, t)
            res = self.anchor_shape(code, t, half_edge_ix)
            if res is False:
                self._non_cov.append(code)
            code += 1

        # set bounds on invalid actions
        B = np.ones(self.num_var)
        B[self._non_cov] = 0

        # todo - is this right ???
        var_by_t = cvx.reshape(self.X, (self.num_act, self.t))

        base = [
            cvx.sum(var_by_t, axis=1) <= 1,         # ok in OO
            self.Faces @ self.X <= 1,           # todo OO
            # self.X <= B,                        # ok in OO
            # self.X >= 0                         # ok in OO ??
        ]

        # max - min tile usages - Note - these are mutex
        if self.max_tile_uses is not None:
            base += [cvx.sum(var_by_t, axis=0) <= self.max_tile_uses]
        if self.min_tile_uses is not None:
            base += [cvx.sum(var_by_t, axis=0) >= self.min_tile_uses]

        # joint constraint - for each Edge - Joint1 = Joint2
        # self.ind_j1 = Variable(shape=self.num_act, boolean=True)
        # sumz = ind_j1 >= 1 # all of these should be true, but this can be tuned??

        # M = 100
        # J_ij is supposed be [num_interior_half_edge x num_color]
        # J1 @ J2_T (symmetric) -> J_ij
        #
        #
        # if half edge on side 1 is set be color c1 (in HE_Joint), then
        #  - corresponding half edge is 0       -> ( J1 - J2 = -  +
        #  - corresponding half edge is c1==c2  -> ( J1 - J2 = 0  +
        #  - constraint not met if c1 != c2     -> ( J1 - J2 = ?
        # exps1 = self.HE_Joint1 @ self.X <= M

        # base += [
        #    self.HE_Joint1 @ self.X <= self.HE_Joint2 @ self.X + M * (1 - self.ind_j1),
        #    self.HE_Joint2 @ self.X <= self.HE_Joint1 @ self.X + M * (1 - self.ind_j1)
        #]

        # if face-to edge color mode:
        # bs = self.J_ij @ self.X
        base += self.joint_constraints()
        return base

    def display(self, problem, constraints=None):
        self.print(problem)
        colors = ['blue', 'yellow', 'green', 'red', 'violet', 'orange']
        sol = [a for code, a in self._code_to_x.items()
               if self.X.value[code] > 0.8]
        info = dict(edgecolor='black', linewidth=1)
        fig, ax = plt.subplots(1)
        # Plot all small squares for each brick
        n, m = sorted(self.G.vertices())[-1]
        print('X-------------------------')
        for a, t in sol:
            p1 = self.place_shape(t, a)
            # print(t, a, p1)

            for x, y in self.place_shape(t, a):
                ax.add_patch(patches.Rectangle( (x, y), 1, 1,
                                                facecolor=colors[t] # , **info
                                                ))
        print(np.where(self.J.value == 1))
        for edge, color in zip(*np.where(self.J.value == 1)):
            u, v = self.G.edges[edge]
            print(u, v)
            ax.plot([u[0], v[0]], [u[1], v[1]], c=colors[-(color+1)], linewidth=7.0)
            # ax.add_line(l, color)
        ax.axis([0, n, 0, m])
        ax.axis('off')
        ax.set_aspect('equal')
        plt.show()