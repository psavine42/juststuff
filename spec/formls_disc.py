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

    def _basic_covering2(self, W, w, h, R, save=None):
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
        # balls = CircleList(P, r=R)

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

    def _box_covering2(self, W, w, h, slack, save=None):
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
        b = []
        fpprob = FloorPlanProp(X)

        fpprob += NoOverlappingFaces(space, actions=X)
        fpprob += LocallyConnectedSet(space, actions=X, limit=2)
        for i in range(P):
            bb = BoxBoundedSet(space, X[i], W[i] * slack, name='bubble.{}'.format(i))
            b.append(bb)
            fpprob += bb
            fpprob += TileLimit(space, i, actions=X, lower=W[i])

        C = fpprob.own_constraints()
        # obj = cvx.Minimize(w * h - cvx.sum(cvx.hstack([X[i].X for i in range(P)])))
        obj = cvx.Minimize(0)
        p = cvx.Problem(obj, C)
        print(describe_problem(p))
        # p.solve(verbose=True)
        p.solve(method='dccp',
                   ep=1e-2,
                   max_slack=1e-2,)
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
            print(fs._box.value)
            draw_formulation_discrete(fs, ax, edgecolor='black', alpha=0.0)
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
        self._basic_covering2(W, w, h, R, 'basic_cover_bub2')

    def test_cover_box(self):
        w, h = 5, 10
        W = [10, 10, 6, 6, 18]
        adj = []
        self._box_covering2(W, w, h, 1.3, 'basic_cover_box')

    def test_cover_bubble_lrg(self):
        w, h = 10, 10
        W = [10, 10, 6, 6, 18]
        R = [2.9, 3, 2, 2, 3.7]
        W = [2 * x for x in W]
        R = [2 * x for x in R]
        self._basic_covering2(W, w, h, R, 'basic_cover_bublarge')

