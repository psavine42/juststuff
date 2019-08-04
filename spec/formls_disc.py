from .formul_test import TestContinuous, TestDiscrete
from src.cvopt.floorplanexample import *

from src.cvopt.formulate.fp_cont import *
from src.cvopt.shape.base import R2
from scipy.spatial import voronoi_plot_2d, Voronoi


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
            g = GloballyConnectedSet(space, X[i], W[i])
            gs.append(g)
            obj = obj + g.as_objective()
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

