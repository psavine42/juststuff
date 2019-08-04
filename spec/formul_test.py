import unittest
from scipy.spatial import voronoi_plot_2d, Voronoi
from src.cvopt.utils import *
import os


const_args_formuls = dict(face=False,
                          edge=False,
                          tile=False,
                          vertex=False,
                          half_edge=False)


class _TestBase(unittest.TestCase):
    def _save_loc(self, save=None):
        if save is not None:
            base = './data/opt/' + self.__class__.__name__
            if not os.path.exists(base):
                os.makedirs(base)
            return base + '/' + save


class TestDiscrete(_TestBase):
    def run_detailed(self, prob, obj_args, save):
        prob.solve(show=False,
                   verbose=True,
                   obj_args=obj_args,
                   const_args=const_args_formuls)
        prob.display(save=self._save_loc(save))


class TestContinuous(_TestBase):
    def save_vor(self, vor: Voronoi, save=None):
        fig = voronoi_plot_2d(vor)
        ax = plt.gca()
        for i, v in enumerate(vor.points):
            draw_vertex(v, ax, index=i)
        finalize(ax=None, save=self._save_loc(save), extents=None)

    def run_detailed(self, prob, disp=[], sargs={}, obj_args={}, save=None):
        prob.solve(show=False,
                   verbose=True,
                   solve_args=sargs,
                   obj_args=obj_args,
                   const_args=const_args_formuls)

    def save_sdp(self, prob, f, save=None):
        self.run_detailed(prob)
        w, h = prob.meta['w'], prob.meta['h']
        fig, ax = plt.subplots(1, figsize=(7, 7))
        ax = draw_box((0, dict(x=0, y=0, w=w, h=h)), ax, label=False,
                      facecolor='white', edgecolor='black')
        if not isinstance(f, list):
            f = [f]
        for fr in f:
            ax = draw_formulation_cont(fr, ax)
        finalize(ax, save=self._save_loc(save), extents=[w, h])

    def _run_sdp(self, p, formulations, ilist, n, dense):
        p.add_formulations(*formulations)
        self.run_detailed(p)
        desc = '_{}_{}'.format(n, 'strict' if dense is True else 'sparse')
        if 'vor' in p.meta:
            self.save_vor(p.meta['vor'], save='sdp_f{}_vor.png'.format(n))
        try:
            self.save_sdp(p, ilist, save='sdp_f{}.png'.format(desc))
        except:
            print('no silution')
