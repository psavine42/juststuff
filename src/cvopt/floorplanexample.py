from cvxpy import *
import cvxpy.lin_ops
import cvxpy.utilities
import pylab

import cassowary
import numpy as np
from .constraining import *
from .spatial import *
from .tilings import *
from cvxpy.utilities.performance_utils import compute_once
import networkx as nx
import itertools
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.patches as patches
from .utils import display_face_bottom_left
from .placements import *
from .formulations import *
from .mesh import *
"""
CASSOWRY 
“inside,” “above,” “below,” “left-of,”
“right-of,” and “overlaps.”

# https://github.com/cvxgrp/cvxpy/blob/master/examples/floor_packing.py
# Based on http://cvxopt.org/examples/book/floorplan.html
"""


def linear_index2(m, x, y):
    """ m: rows in matrix, x, y """
    return x * m + y


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


class FPProbllem(object):
    def __init__(self):
        self._problem = None

    def own_constraints(self, **kwargs):
        raise NotImplemented('not implemented in base class')

    def display(self, problem, **kwargs):
        return

    def action_eliminators(self):
        return []

    def objective(self, **kwargs):
        raise NotImplemented('not implemented in base class')

    def print(self, problem):
        print('Problem:----------')
        # print(problem)
        print('----------')
        print(problem.solution.status)
        print(problem.solution.opt_val)
        print(problem.solution.attr)

    @property
    def solution(self):
        return

    def run(self, obj_args={}, const_args={}, verbose=False, show=True, save=None):
        constraints = self.own_constraints(**const_args)
        if verbose is True:
            for c in constraints:
                print(c)
        objective = self.objective(**obj_args)
        self._problem = Problem(objective, constraints)
        print('constraints registered ')
        self._problem.solve(verbose=verbose)
        print('solution created')
        if show:
            self.display(self._problem, save=save, constraints=constraints)

        return self.solution

    def solve(self, **kwargs):
        return self.run(**kwargs)


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
            sum(reshape(self.X, (self.num_con, self.t)), axis=1) <= 1,
            self.A @ self.X <= 1,
            self.X <= B,
            self.X >= 0
        ]
        if self.limits is not None:
            vr = sum(reshape(self.X, (self.num_con, self.t)), axis=0) <= self.limits
            base.append(vr)
        return base

    def objective(self):
        objective = Maximize(sum(self.areas * self.X))
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
            sum(self.J, axis=1) <= 1,

            # c2 -
            # self.M_ec_ec @ self.J <= self.M_ec_x @ self.X,
            sum(self.J, axis=1) <= reshape(Xvar_to_edge, (self.num_edge, self.num_color)),

            # c3 -
            reshape(ec2x2 @ self.X, (self.num_edge, self.num_color))
               <= reshape(Xvar_to_edge @ self.X, (self.num_edge, self.num_color)) + 100 * (1 - self.J)
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
        var_by_t = reshape(self.X, (self.num_act, self.t))

        base = [
            sum(var_by_t, axis=1) <= 1,         # ok in OO
            self.Faces @ self.X <= 1,           # todo OO
            self.X <= B,                        # ok in OO
            self.X >= 0                         # ok in OO ??
        ]

        # max - min tile usages - Note - these are mutex
        if self.max_tile_uses is not None:
            base += [sum(var_by_t, axis=0) <= self.max_tile_uses]
        if self.min_tile_uses is not None:
            base += [sum(var_by_t, axis=0) >= self.min_tile_uses]

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


# ----------------------------------------------------------------
class AdjPlanOO(FPProbllem):
    """

    optional:
        verts_forbidden - list [index] verticies that cannot be on the interior of a placment
        faces_forbidden - list [index] faces that cannot have any placement

    """
    def __init__(self, templates, mesh,
                 max_tiles=None,
                 min_tiles=None,
                 sinks=None,
                 edges_forbidden=None,
                 faces_forbidden=None,
                 verts_forbidden=None):
        FPProbllem.__init__(self)
        for i, t in enumerate(templates):
            if t.color is None:
                t.color = i

        self.T = templates
        self.t = len(templates)
        self.G = mesh

        self.num_vert = len(self.G.vertices())
        self.num_half_edge = len(self.G.half_edges())
        self.num_edge = len(self.G.edges)
        self.num_face = len(self.G.faces_to_vertices())
        self.num_ihes = len(self.G.interior_half_edge_index()[0])
        self.n_half_edge_color = len(set().union(*[np.abs(x.colors(half_edge=True))
                                                   for x in self.T]).difference([0]))

        self.verts_forbidden = verts_forbidden if verts_forbidden else []
        self.edges_forbidden = edges_forbidden if edges_forbidden else []
        self.faces_forbidden = faces_forbidden if faces_forbidden else []
        self.sink_faces = sinks if sinks else []

        self._faces = []
        self._verts = []
        self._edges = []
        self._placements = []
        self._half_edges = []
        self.eps = 0.4

    @property
    def placements(self):
        return self._placements

    @property
    def solution(self):
        for tile in self._placements:
            ix = np.where(np.asarray(tile.X.value, dtype=int) == 1)[0]
            for i in ix:
                # if tile.value[i] > self.eps: face_ix # ,
                face_ix = tile.placements[i][0]
                yield tile, self._faces[face_ix].bottom_left

    def _build_graph(self):
        self._edges = [self.G.get_edge(i,
                                       cls=Edge,
                                       n_colors=self.n_half_edge_color
                                       )
                       for i in range(self.num_edge)]
        # self._verts = [self.G.get_vertex(i) for i in range(self.num_vert)]

        # Edge <-> Half_Edge
        for i in range(self.num_half_edge):
            # he = self.G.get_half_edge(i, n_colors=self.t)
            he = self.G.get_half_edge(i,
                                      cls=HalfEdge,
                                      n_colors=self.n_half_edge_color
                                      )
            j = he.edges(index=True)
            self._edges[j].connect(he)
            he.connect(self._edges[j])
            self._half_edges.append(he)

            # Vertex <-> Half_Edge :: connect vertices
            # not Half Edge must store vertices in directed order
            # src, tgt = he.vertices(index=True)
            # he.connect(self._verts[src])
            # he.connect(self._verts[tgt])

        # Edge <-> Face
        for i in range(self.num_face):
            face = self.G.get_face(i,
                                   cls=Face,
                                   n_colors=self.t,
                                   is_sink=int(i not in self.sink_faces),
                                   is_usable=int(i not in self.faces_forbidden))
            for j in face.edges(index=True):
                self._edges[j].connect(face)
                face.connect(self._edges[j])
            self._faces.append(face)

    def anchor_shape_on_edge(self, t, edge_ix):
        p1, p2 = self._edges[edge_ix].geom
        uvec = p2[0] - p1[0], p2[1] - p1[1]

        template = self.T[t]
        mapping = {}
        verts = []
        t_edges = template.edges
        q1, q2 = t_edges[0]
        qvec = q2[0] - q1[0], q2[1] - q1[1]
        if qvec != uvec:
            return None

        for p, q in template.edges:
            f1, f2 = ((p1[0] + p[0], p1[1] + p[1]), (p1[0] + q[0], p1[1] + q[1]))
            if (f1, f2) in self.G.edges:
                mapping[(p, q)] = (f1, f2)
                verts.append(f1)
                verts.append(f2)
            else:
                return None

        if all((start, end) not in self.edges_forbidden
               and start not in self.verts_forbidden
               and end not in self.verts_forbidden
               for start, end in mapping.values()):
            return mapping
        return None

    def anchor_shape_on_face(self, t, edge_ix):
        p1, p2 = self._edges[edge_ix].geom
        uvec = p2[0] - p1[0], p2[1] - p1[1]

        template = self.T[t]
        mapping = {}
        t_edges = template.edges
        q1, q2 = t_edges[0]
        qvec = q2[0] - q1[0], q2[1] - q1[1]
        if qvec != uvec:
            return None

        for p, q in template.edges:
            # todo check if same direction
            # magnitude = np.sqrt(x**2 + y**2)
            f1, f2 = ((p1[0] + p[0], p1[1] + p[1]), (p1[0] + q[0], p1[1] + q[1]))
            if (f1, f2) in self.G.edges:
                mapping[(p, q)] = (f1, f2)
            else:
                return None

        if all((start, end) not in self.edges_forbidden
               and start not in self.verts_forbidden
               and end not in self.verts_forbidden
               for start, end in mapping.values()):
            return mapping
        return None

    def _pre_compute_placements(self):
        for t in range(self.t):
            tile = self.T[t]
            valid_edges, valid_faces, xforms = [], [], []

            for edge_ix in range(self.num_edge):
                mapping = self.anchor_shape_on_edge(t, edge_ix)
                if mapping is None:
                    continue
                p1, _ = self._edges[edge_ix].geom
                ixs = [self.G.index_of_edge(x) for x in mapping.values()]
                faces_ixs = []
                for x in ixs:
                    faces_ixs += list(self.G.edges_to_faces()[x])
                valid_faces.append([k for k, v in Counter(faces_ixs).items() if v > 1])
                valid_edges.append(ixs)
                xforms.append(p1)

            if tile.max_uses is None:
                tile.max_uses = len(valid_edges)

            placement = Placement(self, t, valid_edges, valid_faces, xforms)
            self._register_placement(placement)

    def _register_placement(self, placement):
        # connections
        for face in self._faces:
            placement.connect(face)
        for half_edge in self._half_edges:
            placement.connect(half_edge)
        for edge in self._edges:
            placement.connect(edge)

        # registration
        for half_edge in self._half_edges:
            half_edge.register_placement(placement)
        for edge in self._edges:
            edge.register_placement(placement)

        # save placement object
        self._placements.append(placement)

    def objective(self, edge=True, face=True):
        # return Maximize(cvx.sum(cvx.vstack([x.objective_max for x in self._faces])))
        edge_ob = 0
        if edge is True:
            edge_ob = cvx.sum(cvx.vstack([x.objective_max for x in self._edges]))

        return Maximize(
            cvx.sum(cvx.vstack([x.objective_max for x in self._placements]))
            # + cvx.sum(cvx.vstack([x.objective_max for x in self._faces]))
            + edge_ob
        )

    def own_constraints(self, edge=True, face=True, tile=True):
        self._build_graph()
        print('graph built')
        self._pre_compute_placements()
        print('placements computed')

        C = []
        for x in self._edges:
            C += x.constraints
        for x in self._half_edges:
            C += x.constraints
        for x in self._verts:
            C += x.constraints
        if face:
            for x in self._faces:
                C += x.constraints
        if tile:
            for x in self._placements:
                C += x.constraints
        print('constraints computed')
        return C

    def display(self, **kwargs):
        self.print(self._problem)
        display_face_bottom_left(self, **kwargs)


# ----------------------------------------------------------------
class FormPlan(AdjPlanOO):
    obj_space = ['edges', 'vertices', 'faces', 'half_edges']
    atr_space = ['placement', 'color']

    def __init__(self, templates, mesh):
        AdjPlanOO.__init__(self, templates, mesh)
        self.__formulations = []
        self._build_graph()
        print('graph built')
        self._pre_compute_placements()
        print('placements computed')

    def add_constraint(self, formulation, **data):
        if isinstance(formulation, type):
            form_instance = formulation(self.G, **data)
        else:
            form_instance = formulation
        self.__formulations.append(form_instance)
        for p in self._placements:
            form_instance.register_action(p)

    def _register_formulations(self):
        for form in self.__formulations:
            for p in self._placements:
                form.register_action(p)

    def _anchor_on_half_edge(self, t, edge_ix):
        geom = self._half_edges[edge_ix].geom
        template = self.T[t]
        transformation = template.align_to(geom)
        if transformation is None:
            return None
        mapped = MeshMapping(self.G, template, transformation)
        if mapped.is_valid():
            return mapped
        return None

    def _pre_compute_placements(self):
        for tile_index in range(self.t):
            maps = []
            for edge_ix in range(self.num_half_edge):
                mapping = self._anchor_on_half_edge(tile_index, edge_ix)
                if mapping is None:
                    continue
                maps.append(mapping)
            placement = Placement2(self, tile_index, maps)
            self._register_placement(placement)

    @property
    def formulations(self):
        return self.__formulations

    def _build(self):
        self._register_formulations()
        print('formulations registered')

    def own_constraints(self, edge=True, face=True, tile=True):
        C = []
        for x in self.__formulations:
            if x.is_constraint:
                C += x.as_constraint()

        for x in self._edges:
            C += x.constraints
        for x in self._half_edges:
            C += x.constraints
        for x in self._verts:
            C += x.constraints
        if face:
            for x in self._faces:
                C += x.constraints
        if tile:
            for x in self._placements:
                C += x.constraints
        print('constraints computed')
        return C


# ----------------------------------------------------------------
# LINE BASED -
# ----------------------------------------------------------------
class LinePlan(AdjPlanOO):
    obj_space = ['vertices', 'edges']
    atr_space = []
    discrete = True

    def __init__(self,
                 mesh: Mesh2d,
                 sink,
                 points,
                 partitions):
        """
        the idea is that the layout is a selection of edges on mesh (undirected graph)
        which is equivelant to a tree (alexandroff topology) with 'sink' as the root node,
        and sources as the children.

        Therefore -
        mesh can be projected to R3 'distance to sink'
        lets say X_i is sink, nodes with a distance of 1

        ---------
        either each 'room' has N source points.
        therefore, each room has one entry and one exit point,
        therefore, the problem can be stated as connection of Rooms
            as either room.in, room.out

        therefore,

        :param mesh:
        :param points:
        :param sink:
        """
        AdjPlanOO.__init__(self, [], mesh)
        self.M = mesh
        self.sink = sink
        # self.X = Variable(shape=len(self.M.edges), boolean=True)
        self.partitions = partitions
        self.points = points
        self.__formulations = []
        self._build_graph()
        print('graph built')

    @compute_once
    def mesh_partitions(self):
        """ """
        return []

    def add_constraint(self, formulation, **data):
        if isinstance(formulation, type):
            form_instance = formulation(self.G, **data)
        else:
            form_instance = formulation
        self.__formulations.append(form_instance)
        for p in self._placements:
            form_instance.register_action(p)

    def _register_formulations(self):
        for form in self.__formulations:
            for p in self._placements:
                form.register_action(p)

    def _pre_compute_placements(self):
        placement = EdgeSet(self, 0)
        self._register_placement(placement)
        # ActiveEdgeSet(self.M)

    @property
    def formulations(self):
        return self.__formulations

    def _build(self):
        """
        objective will be to minimize number of edges used

        X -> the set of all actions (edges)

        constraints:
            sum(Shortest_paths_ij) <= 1
            Shortest_paths_ij = X @ SP
            c += OR(X @ SP, )

        """
        C = []
        dists = nx.shortest_path_length(self.M.G, self.sink)
        # build 'Room Graph'
        # for each partition (room), it must connect to another partition,
        # therefore a graph can be built of all boundaries (walls)

        # when points are NOT known, maximize distance between
        # points within the partition, and Minimize path length
        # each node T_k is a free location var, representing an index
        # T_k >= max( norm(x_j - x_k) + T_j | there is an arc (j,k)}
        M_sinks = np.ones((self.num_vert))
        # M_sinks
        cvx.max(cvx.norm1())
        for k, node_k in enumerate(self._verts):
            # for j in node_k.adj:
            pass

        # within a room, the following problem can be setup:
        # distances of sources within that room
        room_graph = ddict(set)
        rooms_to_points = ddict(set)
        shortest_paths = ddict(dict)
        for i, room in enumerate(self.mesh_partitions()):
            for j, p in enumerate(self.points):
                if not room.contains(p):
                    continue
                rooms_to_points[i].add(j)

        # for each pair of points in a partition,
        # one of the shortest paths must be active
        for room_ix, point_ixs in rooms_to_points.items():
            for pi1, pi2 in itertools.combinations(point_ixs, 2):
                pths = nx.all_shortest_paths(self.M.G, source=pi1, target=pi2)
                shortest_paths[room_ix][(pi1, pi2)] = pths

        # for optimizing between partitions:
        # given that within a room there is some layout that is most
        # efficient, which .. todo ...

        # structural constraints
        # 1) connectedness
        X = Variable(shape=self.num_edge, boolean=True)

        # minize selected edges
        base_obj = cvx.Minimize(cvx.sum(X))

        # s.t. a tree is formed which includes the specified points
        ec = EdgeConnectivity(self.M)
        ic = IncludeNodes(self.M, node=self.points)

        C += ec.as_constraint(X)
        C += ic.as_constraint(X)
        # C += []
        return base_obj, C


def n_free(M:Mesh2d, nfree, sinks):
    """
    8.7  minimax delay problem
    # todo - why this no work
    """
    T_k = Variable(shape=len(M.vertices()), pos=True)
    objective = cvx.Minimize(cvx.max(T_k))
    C = []
    for i, u in enumerate(M.vertices()):
        if i in sinks:
            C += [T_k[i] <= 0]
            continue
        vars = []
        for v in M.G[u].keys():
            j = M.index_of_vertex(v)
            vars.append(1 + T_k[j])
        C += [T_k[i] >= cvx.max(cvx.hstack(vars))]

    C += [cvx.sum(T_k) <= nfree]

    problem = cvx.Problem(objective, C)
    return problem


def line_layout_problem(space=None, sink=None, points=None, partitions=None):
    if partitions and points is None:
        pass
    if partitions is None and points:
        pass



class LinePlanCont(AdjPlanOO):
    obj_space = ['vertices', 'half_edges']
    atr_space = []

    def __init__(self, mesh, points):
        AdjPlanOO.__init__(self, mesh, [])
        self.M = mesh
        self.__formulations = []
        self._build_graph()
        print('graph built')

    def add_constraint(self, formulation, **data):
        if isinstance(formulation, type):
            form_instance = formulation(self.G, **data)
        else:
            form_instance = formulation
        self.__formulations.append(form_instance)
        for p in self._placements:
            form_instance.register_action(p)

    def _register_formulations(self):
        for form in self.__formulations:
            for p in self._placements:
                form.register_action(p)

    def _anchor_on_half_edge(self, t, edge_ix):
        geom = self._half_edges[edge_ix].geom
        template = self.T[t]
        transformation = template.align_to(geom)
        if transformation is None:
            return None
        mapping = template.transform(transformation)
        if mapping.is_valid():
            return mapping
        return None

    @property
    def formulations(self):
        return self.__formulations



# ----------------------------------------------------------------
# HYBRID
# ----------------------------------------------------------------
class SnakePlan(object):
    """
    Computational Network Design from Functional Specifications

    Parameters:
    -----------------------
        sinks :

    Variables:
    -------------------------------
        - Faces     :
        - Edges     :
        - Vertices  :
        - HalfEdges :

    Funcitons
    ----------------------
        - N
        - 4 * 3 * 3 => 36 possible constraints combinations
        -
    """
    def __init__(self, boxes, h, w=None, edge_len=1, sinks=None, graph=None):
        # edge statistics
        if w is None:
            w = h
        self.num_faces = ''
        self.num_vertices = h * w
        self.num_edge = 2 * h * w - h * w
        self.num_half_edge = 2 * self.num_edge

        #
        self.boxes = boxes
        self.h = h
        self.w = w
        self.G = self._grid_graph(graph)

        # Paramaters
        # D
        self.half_edge_dists = None # todo var
        self.sinks = sinks
        self.edge_lens = edge_len       # D_i→j

        # Variables
        # E_x, E_i→j, E_j→i - edges half edges active/inactive states
        self.E_x_active = Variable(shape=self.num_edge, boolean=True)
        self.E_ij_active = Variable(shape=self.num_edge, boolean=True)
        self.E_ji_active = Variable(shape=self.num_edge, boolean=True)

        # source to sink for all half edges todo
        self.L_ij_jk = Variable()

        # Vy - Vertex active
        self.vertex_active = Variable(shape=self.num_vertices, boolean=True)

    def _grid_graph(self, G=None):
        if G is None:
            G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(self.h, self.w))
        nx.set_edge_attributes(G, {e: i for i, e in enumerate(G.edges)}, 'ix')
        return G

    def distances_to_sinks(self):
        nx.all_pairs_shortest_path()
        return

    @compute_once
    def edge_to_edge_adj(self):
        """ indicies of edges adjacent to edge E_ij

        """
        adj_ij = [None] * len(self.G.edges)
        adj_ji = [None] * len(self.G.edges)
        # print(self.G.adj)
        for i, (n1, n2, data) in enumerate(self.G.edges(data=True)):
            ij = {edge_data['ix'] for k, edge_data in self.G.adj[n1].items()}
            ji = {edge_data['ix'] for k, edge_data in self.G.adj[n2].items()}
            adj_ij[i] = sorted(list(ij.difference([i])))
            adj_ji[i] = sorted(list(ji.difference([i])))
        return adj_ij, adj_ji

    @compute_once
    def vertex_to_edge_adj(self):
        """ indicies of edges adjacent vertex ix V_x
        LEXIGRAPHIC ORDERING !!!
        v_i -> [e_i ]
        """
        adj = [None] * len(self.G.nodes)
        for i, neigh in self.G.adj.items():
            edges = set()
            for n, edge_data in neigh.items():
                edges.add(edge_data['ix'])
            adj[i] = sorted(list(edges.difference([i])))
        return adj

    def half_edge_adj(self, ij_or_ji, ix):
        indices = []
        for e in range(self.num_edge):
            adj = self.edge_to_edge_adj()[ij_or_ji][e]
            if ix >= len(adj):
                indices.append(0)
            else:
                indices.append(self.E_x_active[adj[ix]])
        return vstack(indices)

    @property
    def delta_all(self):
        if isinstance(self.edge_lens, (int, float)):
            return self.edge_lens * self.num_half_edge

    def no_tjunctions(self):
        """
            f: Half_edge x Edge
        """
        tvar = 4
        e_ij1, e_ij2, e_ij3 = [self.half_edge_adj(0, i) for i in range(3)]
        e_ji1, e_ji2, e_ji3 = [self.half_edge_adj(1, i) for i in range(3)]
        return [
            0 <= e_ij1 + e_ij2 + e_ij3 + (1 - self.E_ij_active) - tvar,
            3 >= e_ij1 + e_ij2 + e_ij3 + (1 - self.E_ij_active) - tvar,
            0 <= e_ji1 + e_ji2 + e_ji3 + (1 - self.E_ji_active) - tvar,
            3 >= e_ji1 + e_ji2 + e_ji3 + (1 - self.E_ji_active) - tvar,
        ]

    def own_constraints(self):
        constraints = []
        # must_be_covered = sum(self.Cover) == self.h * self.w

        # C1 - No Islands
        # --------------------------------------------------------
        # L_i→j;j→k ≤ E_j→k                                 eq (1)
        # D_j→k − D_i→j + ∆_all * L_i→j;j→k ≤ ∆all − ∆_i→j  eq (2)

        # E_i→j − sum(L_i→j;j→k) ≤ 0.                       eq (3)

        # −1 ≤ E_i→j + E_j→i − 2E_x ≤ 0                     eq (4)
        # if a half edge is active, the edge us active
        constraints += [
            -1 <= self.E_ij_active + self.E_ji_active - 2 * self.E_x_active,
            +0 >= self.E_ij_active + self.E_ji_active - 2 * self.E_x_active,
        ]

        # C2 - Coverage constraints (5,6)
        # --------------------------------------------------------
        # 1 − |E y| ≤ sum(E_x) − |E y |V y ≤ 0, (5)
        # sum of  edges adjacent  to vertex
        # todo sets of vertex -> edge

        # exp5 = 1 - len() <= sum()

        # L1 Point-to-Point Constraint (7,8)
        # --------------------------------------------------------

        # L2 Local Feature Control (9 ... 14)
        # --------------------------------------------------------
        # x is element of Ey
        constraints += [
            sum(self.E_x_active) <= 2,                      # eq (9)

        ]

        # room placement -

        # delta_all - sum of all half edges

        # 9 dead-end avoidance

        # Branch Avoidance

        # zig-zag avoidance

        # T-junction avoidance
        constraints += self.no_tjunctions()
        return constraints

    def objective(self, edge_len_coef, travel_coef):
        # min
        Minimize(edge_len_coef* sum(self.edge_lens* self.E_x_active) \
                 + travel_coef * sum(self.D)
                 )


# ----------------------------------------------------------------
# Continuous - FIXED Tiling
# ----------------------------------------------------------------
class FloorPlan(object):
    """ A minimum perimeter floor plan. """
    MARGIN = 1.0
    ASPECT_RATIO = 5.0

    def __init__(self, boxes, eps=1e3, adj=None, h=None, w=None):
        self.boxes = boxes
        self.eps = eps
        self.adj = adj if adj else []
        self.height = Variable(pos=True, name='fp.h') if h is None else h
        self.width = Variable(pos=True, name='fp.w') if w is None else w
        self.horizontal_orderings = []
        self.vertical_orderings = []

    def __iter__(self):
        for box in self.boxes:
            for b in iter(box):
                yield b

    @property
    def size(self):
        return np.round(self.width.value, 2), np.round(self.height.value, 2)

    def problem(self, horizantal, vertical):
        constraints = []
        for box in self.boxes:
            constraints += box.own_constraints()
        constraints += order_constraints(horizantal, self.boxes, True)
        constraints += order_constraints(vertical, self.boxes, False)
        constraints += within_boundary_constraints(self.boxes, self.height, self.width)
        problem = Problem(Minimize(2 * (self.height + self.width)), constraints)
        return problem

    def own_constraints(self):
        constraints = []
        if isinstance(self.height, Variable):
            constraints.append(self.height <= self.eps)
        if isinstance(self.width, Variable):
            constraints.append(self.width <= self.eps)
        return constraints

    def build_constraints(self):
        if self.eps is None:
            self.eps = 4 * np.sum([box.min_area for box in self.boxes]) ** 0.5

        constraints = []
        for box in self.boxes:
            constraints += box.own_constraints()

        for i, j in self.adj:
            constraints += must_be_adjacent(self.boxes[i], self.boxes[j],
                                            h=self.eps, w=self.eps)

        constraints += within_boundary_constraints(self.boxes, self.height, self.width)
        constraints += no_overlaps(list(iter(self)), self.eps, self.eps)
        constraints += self.own_constraints()
        return constraints

    def problem2(self, constraints=None):
        """ minimiuze perimeter wall length"""
        if not constraints:
            constraints = self.build_constraints()
        problem = Problem(Minimize(2 * (self.height + self.width)), constraints)
        return problem

    def min_wall_len(self):
        constraints = self.build_constraints()
        problem = Problem(Minimize(sum([box.perimeter for box in self.boxes])), constraints)
        return problem

    def show(self, save=None, d=True):
        # Show the layout with matplotlib
        pylab.figure(facecolor='w')
        mxx, mxy = 0, 0
        for k in range(len(self.boxes)):

            for box in iter(self.boxes[k]):
                x, y = box.position
                w, h = box.size
                pylab.fill([x, x, x + w, x + w],
                           [y, y+h, y+h, y],
                           facecolor='#D0D0D0',
                           edgecolor='k'
                           )
                pylab.text(x + 0.5 * w, y + 0.5 * h, "{}".format(box.name))
                mxx, mxy = np.max([mxx, x + w]), np.max([mxy, y+h])
        # x, y = self.size
        pylab.axis([0, mxx, 0, mxy])
        pylab.xticks([])
        pylab.yticks([])
        if isinstance(save, str):
            pylab.savefig(save)
            pylab.clf()
            pylab.close()
        if d is True:
            pylab.show()


# ----------------------------------------------------------------
def __boxes1():
    return [Box(200, name=0), Box(80, name=1),
            Box(80, name=3), Box(120, name=2),
            Box(120, name=4)]


def __boxes2():
    return [Box(200, name=0), Box(80, name=1),
            Box(80, name=3), Box(120, name=2),
            Box(120, name=4)]


# ----------------------------------------------------------------
def test_prob1():
    """ original form """
    boxes = __boxes1()
    fp = FloorPlan(boxes)
    fp.horizontal_orderings.append( [boxes[0], boxes[2], boxes[4]] )
    fp.horizontal_orderings.append( [boxes[1], boxes[2]] )
    fp.horizontal_orderings.append( [boxes[3], boxes[4]] )
    fp.vertical_orderings.append( [boxes[1], boxes[0], boxes[3]] )
    fp.vertical_orderings.append( [boxes[2], boxes[3]] )
    problem = fp.layout()
    print(problem.is_dcp(), problem.is_dqcp(), problem.is_qp(),
          problem.is_dgp(), problem.is_mixed_integer())

    problem.solve()
    for box in boxes:
        print(str(box))
    print(problem)
    print(problem.solution)
    fp.show()


def test_prob2():
    """ """
    boxes = __boxes1()
    fp = FloorPlan(boxes)

    adjacency = [[0, 2], [1, 3]]

    hmat = [[0, 2, 4], [1, 2], [3, 4]]
    vmat = [[1, 0, 3], [2, 3]]

    problem = fp.problem(hmat, vmat)
    print(problem.is_dcp(), problem.is_dqcp(), problem.is_qp(),
          problem.is_dgp(), problem.is_mixed_integer())
    print(problem)
    problem.solve()
    print(problem.solution)
    for box in boxes:
        print(str(box))

    fp.show()


def test_prob3():
    """ with better no-overlap constraints """
    boxes = __boxes1()
    fp = FloorPlan(boxes, adj=[(0, 1),
                               (1, 3)])

    problem = fp.problem2()
    print(problem.is_dcp(), problem.is_dqcp(), problem.is_qp(),
          problem.is_dgp(), problem.is_mixed_integer())
    print(problem)
    problem.solve()
    print(problem.solution)
    for box in boxes:
        print(str(box))
    fp.show()


if __name__ == '__main__':
    test_prob3()

