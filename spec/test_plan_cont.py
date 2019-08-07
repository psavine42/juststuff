import unittest
from example.cvopt.garage import *
from src.cvopt.floorplanexample import *
from src.cvopt.problem import *
from src.cvopt.mesh.mapping import MeshMapping
from pprint import pprint
from example.cvopt.famoius import *
from src.cvopt.formulate.fp_cont import *
from src.cvopt.shape.base import R2
from scipy.spatial import voronoi_plot_2d, Voronoi
from .formul_test import TestContinuous, TestDiscrete
from example.cvopt.utils import *
import datetime


class TestPosition(TestContinuous):
    """
    Circle Packing/placement stage 1 positioning
    """
    def _setup_pack(self, obj, n=3, **kwargs):
        cij, areas = generic(n)
        inputs = [BTile(None, area=a) for a in areas]
        circle_list = CircleList(inputs)
        pack = PackCircles(circle_list, obj=obj, **kwargs)
        return Problem(pack.as_objective(), pack.as_constraint()), pack

    def solve(self, prob, pack, verbose=False, save=None, solver=None):
        time_start = datetime.datetime.now()
        print(prob.objective)
        for c in prob.constraints:
            print(c)
        prob.solve(method='dccp', solver=solver,
                   ep=1e-2,
                   max_slack=1e-2, verbose=verbose)
        end_time = datetime.datetime.now()
        print((end_time - time_start).microseconds / 1e6)

        c, r, n = pack.vars, pack.r, pack.vars.shape[0]

        l = cvx.max(cvx.max(cvx.abs(c), axis=1) + r).value * 2

        # plot
        plt.figure(figsize=(5, 5))
        circ = np.linspace(0, 2 * np.pi)
        x_border = [-l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_border = [-l / 2, -l / 2, l / 2, l / 2, -l / 2]
        for i in range(n):
            plt.plot(c[i, 0].value + r[i] * np.cos(circ), c[i, 1].value + r[i] * np.sin(circ), 'b')
            plt.text(c[i, 0].value, c[i, 1].value, '{}'.format(i))
        plt.plot(x_border, y_border, 'g')
        plt.axes().set_aspect('equal')
        plt.xlim([-l / 2, l / 2])
        plt.ylim([-l / 2, l / 2])
        finalize(save=self._save_loc(save))

    def test_ao1(self):
        objs = ['rect', 'sqr', 'ar0', 'ar1',  'ar3', 'ar01', 'ar03', 'arf']
        # for o, tf in itertools.product(objs, [True, False]):
        for o in objs:
            prob, pack = self._setup_pack(o)
            print('\n' + o)
            print(describe_problem(prob))
        prob = self._setup_pack('ar3', verbose=True)

    def test_ao2(self):
        n = 3
        cij = adj_mat_n(n)
        print(cij)
        prob, pack = self._setup_pack('ar0', n=n, cij=cij, verbose=True)
        print(describe_problem(prob))
        self.solve(prob, pack)

    def test_ao5(self):
        n = 5
        cij = adj_mat_n(n)
        print(cij)
        prob, pack = self._setup_pack('ar0', n=n, cij=cij, verbose=True)
        print(describe_problem(prob))
        self.solve(prob, pack, save='circle5')

    def test_ao5ecos(self):
        n = 5
        cij = adj_mat_n(n)
        print(cij)
        prob, pack = self._setup_pack('ar0', n=n, cij=cij, verbose=True)
        print(describe_problem(prob))
        self.solve(prob, pack, save='circle5ecos', solver='ECOS')


# -----------------------------------------------------------------------
#
# -----------------------------------------------------------------------
class TestPlanCont(TestContinuous):
    """ problems with a continuous Covering in R2 -> floor planning """
    # setup/save -----------------------------------------------
    def _make_prob(self):
        cij, areas = generic(3)
        inputs = [BTile(None, area=a) for a in areas]
        return FloorPlan2(inputs), cij

    def _make_prob_stage2(self, n=3, dense=False):
        if dense is False:
            cij, areas, pts, rpm, vor = from_voronoi(n)
        else:
            cij, areas, pts, rpm, vor = from_rand(n)
        w = np.ceil(np.sqrt(areas.sum()))
        print(areas, w, w**2)
        assert w**2 >= areas.sum()
        inputs = [BTile(None, area=a) for i, a in enumerate(areas)]
        problem = FloorPlan2(inputs)
        problem.meta['rpm'] = rpm
        problem.meta['pts'] = pts
        problem.meta['vor'] = vor
        problem.meta['w'] = w # + 2
        problem.meta['h'] = w # + 2
        return problem, cij

    # --------------------------------------------------------------------
    def _sdpN(self, n, dense=False):
        p, cij = self._make_prob_stage2(n, dense=dense)
        w, h, rpm = p.meta['w'], p.meta['h'], p.meta['rpm']
        f = PlaceLayoutSDP(p.domain, p.placements, rpm, width=w, height=h)
        p.add_constraints(f)
        self.run_detailed(p)
        desc = '_{}_{}'.format(n, 'strict' if dense is True else 'sparse')
        self.save_vor(p.meta['vor'], save='sdp{}_vor.png'.format(n))
        self.save_sdp(p, f, save='sdp{}.png'.format(desc))

    def test_sdpF(self, n=10, dense=False):
        p, cij = self._make_prob_stage2(n, dense=dense)
        w, h, rpm = p.meta['w'], p.meta['h'], p.meta['rpm']
        input_list = BoxInputList(p.placements)

        formulations = [
            BoxAspect(inputs=input_list, high=4, is_objective=False),
            MinFixedPerimeters(inputs=input_list, is_objective=True),
            RPM(rpm, inputs=input_list),
            BoundsXYWH(inputs=input_list, w=w, h=h),
        ]
        self._run_sdp(p, formulations, input_list, n, dense)

    # ---------------------------------------------------------------------
    def test_sdpS(self):
        n = 10
        dense = True
        p, cij = self._make_prob_stage2(n, dense=dense)
        w, h, rpm = p.meta['w'], p.meta['h'], p.meta['rpm']
        input_list = BoxInputList(p.placements)

        formuls = [
            SRPM(rpm, inputs=input_list),
            BoxAspect(inputs=input_list, high=4, is_objective=False),
            MinFixedPerimeters(inputs=input_list, is_objective=True),
            BoundsXYWH(inputs=input_list, w=w, h=h),
        ]
        self._run_sdp(p, formuls, input_list, n, dense)

    def test_sdpU(self):
        n = 10
        dense = True
        p, cij = self._make_prob_stage2(n, dense=dense)
        w, h, rpm, pts = p.meta['w'], p.meta['h'], p.meta['rpm'], p.meta['pts']
        input_list = BoxInputList(p.placements)

        srpm = SRPM(rpm, inputs=input_list, points=pts)
        formuls = [
            srpm,
            BoxAspect(inputs=input_list, high=4, is_objective=False),
            MinFixedPerimeters(inputs=input_list, is_objective=True),
            BoundsXYWH(inputs=input_list, w=w, h=h),
            UnusableZone(input_list, rpm=srpm, x=0, y=0, w=1, h=1),

        ]
        self._run_sdp(p, formuls, input_list, 'uz10', dense)

    # scenarios ---------------------------------------------------------------------
    def test_apt(self):
        """ apartment """
        w, h = 100, 40
        hall = BTile(min_aspect=1000, width_min=3)
        elevator = BTile(min_aspect=1, width_min=12, width_height=12)
        stair1 = BTile(min_aspect=5, min_dim=12)
        stair2 = BTile(min_aspect=5, min_dim=12)
        storage1 = BTile(min_aspect=2, min_dim=6)

    def test_disputed_zone(self):
        """
        by some circumstance, a strange outline has been specified
        the total area is achievable, but not under pure box constraints.
        This is the base for the disputed zone.
        """
        w = 20
        h = 14
        pts = np.asarray([[4, 7], [15, 7]])
        inputs = [BTile(None, area=80) for _ in range(2)]
        problem = FloorPlan2(inputs)
        problem.meta['w'] = w
        problem.meta['h'] = h
        input_list = BoxInputList(inputs)

        rpm = RPM.from_points(pts, inputs=input_list)
        zones = [
            UnusableZone(input_list, pts=pts, x=6, y=1, w=12, h=2),
            UnusableZone(input_list, pts=pts, x=4, y=3, w=8, h=2),
            UnusableZone(input_list, pts=pts, x=14, y=13, w=12, h=2),
            UnusableZone(input_list, pts=pts, x=16, y=11, w=8, h=2)
        ]
        formuls = [
            BoxAspect(inputs=input_list, high=4, is_objective=False),
            MinFixedPerimeters(inputs=input_list, is_objective=True),
            BoundsXYWH(inputs=input_list, w=w, h=h),
        ]
        all_forms = zones + formuls + [rpm]
        self._run_sdp(problem, all_forms, [input_list] + zones, 'dz10', True)
        # --------------------------------------------------------------
        inputs = [BTile(None, area=100) for _ in range(2)]
        problem = FloorPlan2(inputs)
        problem.meta['w'] = w
        problem.meta['h'] = h
        input_list = BoxInputList(inputs)

        rpm = RPM.from_points(pts, inputs=input_list)
        zones = [
            UnusableZone(input_list, pts=pts, x=6, y=1, w=12, h=2),
            UnusableZone(input_list, pts=pts, x=4, y=3, w=8, h=2),
            UnusableZone(input_list, pts=pts, x=14, y=13, w=12, h=2),
            UnusableZone(input_list, pts=pts, x=16, y=11, w=8, h=2)
        ]
        formuls = [
            BoxAspect(inputs=input_list, high=4, is_objective=False),
            MinFixedPerimeters(inputs=input_list, is_objective=True),
            BoundsXYWH(inputs=input_list, w=w, h=h),
        ]
        all_forms = zones + formuls + [rpm]
        # self._run_sdp(problem, all_forms, input_list, 'dz10', True)

    # ----------------------------------------------
    def test_stage2_gm(self):
        p, cij = self._make_prob_stage2(4)
        f = PlaceLayoutGM(p.domain, p.placements,
                           p.meta['rpm'],
                           width=p.meta['w'],
                           height=p.meta['h'])
        p.add_constraints(f)
        print(f.rpm.describe(text=True))
        self.run_detailed(p)
        self.save_sdp(p, f, save='sdp4.png')

    def test_stage2_sdp4(self):
        self._sdpN(4)

    def test_stage2_sdp_dense(self):
        self._sdpN(4, True)
        self._sdpN(10, True)
        self._sdpN(30, True)

    def test_stage2_sdp_sparse(self):
        self._sdpN(4, False)
        self._sdpN(10, False)
        self._sdpN(30, False)

    def test_stage2_sdp10(self):
        self.test_sdpF(10, True)

    def test_stage2_sdp30(self):
        self._sdpN(30)

    # MIsc small tests ----------------------------------------------
    def testrun(self):
        p, cij = self._make_prob()
        a = np.sqrt(np.sum([x.area for x in p.placements]))
        f = PlaceCirclesAR(p.domain, p.placements, cij=cij, width=a, height=a)
        p.add_constraints(f)
        self.run_detailed(p)

    def test_stat(self):
        p, cij = self._make_prob()
        a = np.sqrt(np.sum([x.area for x in p.placements]))
        f = PlaceCirclesAR(p.domain, p.placements, cij=cij, width=a, height=a)
        p.add_constraints(f)
        p.make()
        form_utils.expr_tree_detail(p.problem)

    def test_vor_setup(self):
        p, cij = self._make_prob_stage2(10, dense=True)
        w, h, rpm = p.meta['w'], p.meta['h'], p.meta['rpm']
        f = PlaceLayoutSDP(p.domain, p.placements, rpm, width=w, height=h)
        p.add_constraints(f)
        print(f.rpm.describe(text=True))

    def test_rpm(self):
        n = 10
        dense = True
        p, cij = self._make_prob_stage2(n, dense=dense)
        input_list = BoxInputList(p.placements)

        brpm = RPM(p.meta['rpm'], inputs=input_list)
        srpm = SRPM(p.meta['rpm'], inputs=input_list)
        print('dense')
        print(brpm.describe(text=True))
        print('sparse')
        print('----------------')
        print(srpm.describe(text=True))

