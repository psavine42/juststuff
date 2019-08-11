from .formul_test import TestContinuous, TestDiscrete
from src.cvopt.floorplanexample import *

from src.cvopt.formulate.fp_cont import *
from src.cvopt.shape.base import R2
from scipy.spatial import voronoi_plot_2d, Voronoi
from example.cvopt.utils import *
from src.cvopt.problem import describe_problem


class FPDiscrete(TestDiscrete):
    def _basic_covering(self, W, w, h, save=None):
        """
        inputs are:
            - discrete domain
            - minimum areas of covering elements

        performance:
            5  x 10 -> 0.2s
            10 x 10 -> 3.8s

        """
        P = len(W)
        space = Mesh2d.from_grid(w, h)
        X = [FaceSet(space, i) for i in range(P)]

        # constraints
        C = []
        C += NoOverlappingFaces(space, actions=X).as_constraint()
        C += LocallyConnectedSet(space, actions=X).as_constraint()
        gs = []
        obj = cvx.Minimize(w * h - cvx.sum(cvx.hstack([X[i].X for i in range(P)])))
        for i in range(P):
            g = RadiallyConnectedSet(space, X[i], W[i])
            gs.append(g)
            # obj = obj + g.as_objective()
            C += g.as_constraint()
            C += TileLimit(space, i, actions=X, lower=W[i]).as_constraint()

        p = cvx.Problem(obj, C)
        p.solve(verbose=True)
        for c in C:
            print(c)

        print(obj)
        print(p.solution)
        assert p.solution.status == 'optimal'
        # assert np.allclose(p.solution.opt_val, w * h)
        # print
        fig, ax = plt.subplots(1, figsize=(7, 7))
        colors = cm.viridis(np.linspace(0, 1, P))
        for i, fs in enumerate(X):
            draw_formulation_discrete(fs, ax, facecolor=colors[i])
        finalize(ax, save=self._save_loc(save), extents=[w, h])

    def _basic_circle(self, W, w, h, R, save=None):
        """
        inputs are:
            - discrete domain
            - minimum areas of covering elements

        performance:
            5  x 10 -> 0.2s
            10 x 10 -> 3.8s

        """
        P = len(W)
        space = Mesh2d.from_grid(w, h)
        X = [FaceSet(space, i) for i in range(P)]
        balls = CircleList(P, r=R)

        b = []
        fpprob = FloorPlanProp(X)

        fpprob += NoOverlappingFaces(space, actions=X)
        fpprob += LocallyConnectedSet(space, actions=X, limit=2)
        for i in range(P):
            bb = RadiallyConnectedSet(space, X[i], R[i], name='bubble.{}'.format(i))
            b.append(bb)
            fpprob += bb
            fpprob += TileLimit(space, i, actions=X, lower=W[i])

        C = fpprob.own_constraints()
        # obj = cvx.Minimize(w * h - cvx.sum(cvx.hstack([X[i].X for i in range(P)])))
        obj = cvx.Minimize(0)
        p = cvx.Problem(obj, C)
        p.solve(verbose=True)
        for c in C:
            print(c)

        print(obj)
        print(p.solution)

        assert p.solution.status == 'optimal'
        fig, ax = plt.subplots(1, figsize=(7, 7))
        colors = cm.viridis(np.linspace(0, 1, P))
        for i, fs in enumerate(X):
            draw_formulation_discrete(fs, ax, facecolor=colors[i], label=False, alpha=0.7)
        for i, fs in enumerate(b):
            print(fs._bubble.value)
            draw_formulation_discrete(fs, ax, facecolor=colors[i], alpha=0.4)
        finalize(ax, save=self._save_loc(save), extents=[w+np.max(R)/2, h+np.max(R)/2])

    def _box_covering3(self, A, w, h, save=None):
        """
        inputs are:
            - discrete domain
            - minimum areas of covering elements

        Trying to overcome slicing and boxing
        Creates Boxes in R2 and PlacementSets in Discrete space.

        """
        P = len(A)
        space = Mesh2d.from_grid(w, h)
        tiles = [BTile(area=A[i], name=str(i)) for i in range(P)]

        X = [FaceSet(space, i) for i in range(P)]
        box_list = BoxInputList(tiles, name='bx')

        fp_prob = FloorPlanProp(X)

        # continuous constraints
        bbx = BoxBoundedSet(space, X, box_list)
        baspect = BoxAspectLinear(box_list, mx_area=A, mx_aspect=3)
        fp_prob.add_formulations(bbx, baspect)

        # discrete constraints
        fp_prob += NoOverlappingFaces(space, actions=X)
        fp_prob += LocallyConnectedSet(space, actions=X, limit=2)
        for i in range(P):
            fp_prob += TileLimit(space, i, actions=X, lower=A[i])

        C = fp_prob.own_constraints()
        obj = cvx.Minimize(0)
        p = cvx.Problem(obj, C)

        print(describe_problem(p))
        p.solve(verbose=True)
        for c in C:
            print(c)

        print(obj)
        # print(p.solution)

        assert p.solution.status == 'optimal'
        fig, ax = plt.subplots(1, figsize=(7, 7))
        colors = cm.viridis(np.linspace(0, 1, P))
        for i, fs in enumerate(X):
            draw_formulation_discrete(fs, ax, facecolor=colors[i], label=False, alpha=0.7)

        print()
        print(baspect.describe())
        print()
        print(box_list.describe())

        draw_formulation_cont(box_list, ax, edgecolor='black', alpha=0.4)
        finalize(ax, save=self._save_loc(save), extents=[w, h])

    def test_basic_mini(self):
        w, h = 4, 4
        W = [4, 4, 4, 4]
        self._basic_covering(W, w, h, 'cover')

    def test_basic_covering(self):
        w, h = 5, 10
        W = [10, 10, 6, 6, 18]
        self._basic_covering(W, w, h, 'basic_cover2')

    def test_basic_covering2x(self):
        w, h = 10, 10
        W = [2* x for x in [10, 10, 6, 6, 18]]
        self._basic_covering(W, w, h, 'basic_cover2x')

    def test_cover_bubble(self):
        w, h = 5, 10
        W = [10, 10, 6, 6, 18]
        R = [2.4, 2.4, 2, 2, 3.2]
        adj = []
        self._box_covering3(W, w, h, 'basic_cover_bub2')

    def test_cover_box(self):
        w, h = 5, 10
        W = [10, 10, 6, 6, 18]
        adj = []
        self._box_covering3(W, w, h, 'basic_cover_boxx3')

    def test_cover_box_lrg(self):
        A = [10, 10, 6, 6, 7, 8, 14, 18]
        d = np.ceil(np.asarray(A).sum() ** 0.5)
        d = int(d.astype(int))
        w, h = d, d
        print('size', d)
        self._box_covering3(A, w, h, 'disccont_cover_box_n=8')

    def test_partial_rpm(self):
        pass

    def test_known_box_locs(self):
        pass

    def test_disc_adj(self):
        pass


