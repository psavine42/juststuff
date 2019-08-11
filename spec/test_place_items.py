from .formul_test import TestContinuous, TestDiscrete
from src.cvopt.floorplanexample import *

from src.cvopt.formulate.fp_cont import *
from src.cvopt.formulate.stages import *
from src.cvopt.shape.base import R2
from scipy.spatial import voronoi_plot_2d, Voronoi
from example.cvopt.utils import *
from src.cvopt.problem import describe_problem
import src.cvopt.cvx_objects as cxo
from src.cvopt.formulate.cont_base import *
from src.cvopt.formulate import *


class TestFurnitures(TestContinuous):
    """
    Placing objects of fixed size in continuous space

    Also 'catalogue' of object types
    """
    def _room_problem(self):
        boundary = []
        objects = []
        distances = []
        doors = []
        windows = []
        return

    def _turlet_problem(self):
        """

        """
        return dict(
            boundary={'box': [5, 8 + 2/12]},
            objects=dict(
                turlet={'box': [4, 2]},
                tub={'box':    [4.9, 2.5 ]},
                sink={'box':   [3, 2]},
            ),
            doors=[{'r': 2.5, 'x': None, 'y': None}],
            inv_adj={},
            distances=[],
            windows={'segment': {}})

    def _turlet_problem3f(self):
        """
        """
        return dict(
            boundary={'box': [5, 8 + 2/12]},
            objects=dict(
                turlet={'box': [4, 34/12]},
                tub={'box': [2.5, 4.9]},
                sink={'box': [2, 3]},
                door={'box': [2, 2]},
            ),
            doors=[{'r': 2.5, 'x': None, 'y': None}],
            inv_adj={(0, 1): 3},    # repulsive force
            distances=[],
            windows={'segment': {}})

    def _turlet_problem4f(self):
        """
        """
        return dict(
            boundary={'box': [5, 8 + 2/12]},
            objects=dict(
                turlet={'box': [4, 34/12]},
                tub={'box': [2.5, 4.9]},
                sink={'box': [2, 3]},
                door={'box': [2, 2]},
            ),
            doors=[{'r': 2.5, 'x': None, 'y': None}],
            inv_adj={(0, 1): 3},    # repulsive force
            distances=[],
            windows={'segment': {}})

    def _turlet_problem_f3_01(self):
        return dict(
            name='',
            boundary={'box': [6, 6]},
            objects=dict(
                turlet={'box': [4, 3]},
                tub={'box': [3, 3]},
                sink={'box': [2, 2]},
                door={'box': [2, 2], 'side':'xmax'},
            ),
            doors=[{'r': 2.5, 'x': None, 'y': None}],
            inv_adj={},
            distances=[],
            windows={'segment': {}})

    def _turlet_problem_f3_10(self):
        return dict(
            name='',
            boundary={'box': [5.5, 7.75]},
            objects=dict(
                turlet={'box': [4, 3]},
                tub={'box':  [3, 3]},
                sink={'box': [2, 2]},
                door={'box': [2, 2], 'side':'xmax'},
            ),
            doors=[{'r': 2.5, 'x': None, 'y': None}],
            inv_adj={},
            distances=[],
            windows={'segment': {}})

    def _turlet_problem_f2_01(self):
        return dict(
            boundary={'box': [5, 4.25]},
            objects=dict(
                turlet={'box': [4, 3]},
                sink={'box': [1, 1]},
                door={'box': [2, 2], 'side':'xmax'},
            ),
            doors=[{'r': 2.5, 'x': None, 'y': None}],
            inv_adj={},
            distances=[],
            windows={'segment': {}})

    def test_solvem1(self):
        """ """
        #
        # border = Path(4, name='border')
        # path of width h to all things

        # nothing infront of doors
        N = 3
        data = self._turlet_problem()

        tiles = []
        for k, fixture in data['objects'].items():
            w, h = fixture['box']
            tiles.append(BTile(area=w*h, name=k))

        box_list = BoxInputList(tiles)
        circle_list = CircleList(tiles, dim=2)

        # parrallel constraints
        bw, bh = data['boundary']['box']
        bnds = BoundsXYWH(circle_list, w=bw, h=bh)
        edmc = EuclideanDistanceMatrix(circle_list, obj=Maximize)
        # edmc = PackCircles(circle_list, min_edge=1, is_objective=True)
        # outputs fn(optimized)
        stage1 = Stage(
            circle_list,
            forms=[bnds, edmc]
        )
        p = stage1.make(verbose=True)
        print(describe_problem(p))
        print(p.objective.expr.curvature)
        stage1.solve(verbose=True)
        assert stage1.is_solved is True
        save_pth = self._save_loc(save='p1_s1')
        stage1.display(save=save_pth)

        out = stage1.outputs[:, 0:2]
        print(out)

        rpm = RPM.from_points(out, inputs=box_list)
        edm = EuclideanDistanceMatrix(box_list, obj=Maximize)

        stage2 = Stage(
            box_list,
            forms=[rpm,
                   edm
                   ]
                )

        # thing 1 is closer to X than thing 2
        # cvx.dist_ratio
        #

        # objective - minimize Perimeter
        # miniminze distances of stuff to walls
        # m = Maximize(0)
        return

    def _solvem2(self, data, save=None):
        """ """
        # path of width h to all things

        # nothing infront of doors
        # if '' in data['boundary']
        if data['boundary'] == 'classical':
            bw, bh = Variable(name='W'), Variable(name='H')
        else:
            bw, bh = data['boundary']['box']
        centr = np.asarray([bw/2, bh/2])
        tiles = []
        ws, hs = [], []
        for k, fixture in data['objects'].items():
            w, h = fixture['box']
            ws.append(w)
            hs.append(h)
            tiles.append(BTile(area=w*h, name=k))
        ws = np.asarray(ws)
        hs = np.asarray(hs)

        # list of repulsive forces to maximize distances with
        dist_weights = np.ones((len(tiles), len(tiles)))
        for k, v in data.get('inv_adj', {}).items():
            ki, kj = k
            dist_weights[ki, kj] = v
        dist_weights = dist_weights[np.triu_indices(len(tiles), 1)].tolist()

        # setup of problem objects
        # -------------------------------------------
        # inputs
        box_list = BoxInputList(tiles)
        # if it is a classical problem, solve for min boundary
        pld = PointDistObj(box_list,
                           weight=dist_weights,
                           obj=Maximize)
        fixdim = FixedDimension(box_list, values=[ws, hs],
                                #, indices=[0, 1, 2]
                                )
        # formulations for problem constraints
        frm = [BoundsXYWH(box_list, w=bw, h=bh),
               fixdim,
               pld,
               # FeasibleSet(),
               NoOvelapMIP(box_list)
        ]
        # Add Problem Specific Logic
        # -------------------------------------------
        # 1) There is a door on the left side of the thing
        # no box can overlap the Door swing
        # model the swing as a square
        door = ConstrLambda([box_list.X[3] == 1])
        frm.append(door)

        # 2) each fixture is restricted to a wall
        # tl == BoundsL or tr == BoundsR ....
        choice = OneEdgeMustTouchBoundary(box_list, [0, bh, 0, bw])
        frm.append(choice)

        # 3) Short side restrictions
        # todo - is it worth modeling these differently?
        # However, also, the toilets short side must be on wall
        # and the sinks long-side must be on wall
        # when fdim[0, i] = 1, then Xi.W > Xi.H
        # therefore choice of wall is restricted to xmin, xmax
        # choice[2, i] + choice[3, i] == 1
        #
        # w_short = fixdim.indicators[0]
        # w_long = fixdim.indicators[1]
        # inds = choice.indicators

        # 4) todo There is a bounding box within which turlet exists
        # which can only overlap the door

        # 5) Toilet is not facing Tub (common architecture crit thing)
        # modeled as same orientation
        frm.append(OrientationConstr(box_list, [0, 1]))

        # -----------------------------
        # combine into a problem stage
        stage1 = Stage(box_list, forms=frm)

        p = stage1.make(verbose=True)
        print(describe_problem(p))
        # print(p.objective.expr.curvature)
        p.solve(verbose=True)
        assert stage1.is_solved is True
        save_pth = self._save_loc(save=save)
        stage1.display(save=save_pth, extents=[-1, 6, -1, 10])
        print(box_list.describe())
        print(p.value)
        print('edge_choice\n', choice.indicators.value)
        print('dim_choice\n', fixdim.indicators.value)
        for vr in pld.uv:
            print(vr.name(), vr.value)
        for t in tiles:
            print(t.area)

    def _solvem3(self, data, save=None):
        """ """
        if data['boundary'] == 'classical':
            bw, bh = Variable(name='W'), Variable(name='H')
        else:
            bw, bh = data['boundary']['box']
        centr = np.asarray([bw/2, bh/2])
        tiles = []
        stiles = []
        ws, hs = [], []
        sws, shs = [], []
        for k, fixture in data['objects'].items():
            w, h = fixture['box']
            ws.append(w)
            hs.append(h)
            tiles.append(BTile(area=w * h, name=k))

            small_size = fixture.get('box_s', None)
            if small_size is not None:
                sw, sh = small_size
                sws.append(sw)
                shs.append(sh)
                stiles.append(BTile(area=sw * sh, name=k))

        ws = np.asarray(ws)
        hs = np.asarray(hs)

        # list of repulsive forces to maximize distances with
        dist_weights = np.ones((len(tiles), len(tiles)))
        for k, v in data.get('inv_adj', {}).items():
            ki, kj = k
            dist_weights[ki, kj] = v
        dist_weights = dist_weights[np.triu_indices(len(tiles), 1)].tolist()

        # setup of problem objects
        # -------------------------------------------
        # inputs
        box_list = BoxInputList(tiles)
        # if it is a classical problem, solve for min boundary
        pld = PointDistObj(box_list,
                           weight=dist_weights,
                           obj=Maximize)
        fixdim = FixedDimension(box_list, values=[ws, hs],
                                #, indices=[0, 1, 2]
                                )
        # formulations for problem constraints
        frm = [BoundsXYWH(box_list, w=bw, h=bh),
               fixdim,
               pld,
               # FeasibleSet(),
               NoOvelapMIP(box_list)
        ]
        # Add Problem Specific Logic
        # -------------------------------------------
        # 1) There is a door on the left side of the thing
        # no box can overlap the Door swing
        # model the swing as a square
        door = ConstrLambda([box_list.X[3] == 1])
        frm.append(door)

        # 2) each fixture is restricted to a wall
        # tl == BoundsL or tr == BoundsR ....
        choice = OneEdgeMustTouchBoundary(box_list, [0, bh, 0, bw])
        frm.append(choice)

        # 4) todo There is a bounding box within which turlet exists
        # which can only overlap the door
        # there are actually two boxlists 1 is active when door is there
        # the smaller one

        # 5) Toilet is not facing Tub (common architecture crit thing)
        # modeled as same orientation
        frm.append(OrientationConstr(box_list, [0, 1]))

        # -----------------------------
        # combine into a problem stage
        stage1 = Stage(box_list, forms=frm)

        p = stage1.make(verbose=True)
        print(describe_problem(p))
        # print(p.objective.expr.curvature)
        p.solve(verbose=True)
        assert stage1.is_solved is True
        save_pth = self._save_loc(save=save)
        stage1.display(save=save_pth, extents=[-1, 6, -1, 10])
        print(box_list.describe())
        print(p.value)
        print('edge_choice\n', choice.indicators.value)
        print('dim_choice\n', fixdim.indicators.value)
        for vr in pld.uv:
            print(vr.name(), vr.value)
        for t in tiles:
            print(t.area)

    def test_solvem2(self):
        data = self._turlet_problem3f()
        self._solvem2(data, save='p2_s1')

    def test_f3_01(self):
        data = self._turlet_problem3f()
        self._solvem2(data, save='f3_01')

    def test_misc(self):
        pts = [[0, 1], [3, 1], [2, 1] ]
        h = cxo.ConvexHull(pts)

    def test_fixed(self):
        data = self._turlet_problem()
        bw, bh = data['boundary']['box']
        tiles = []
        ws, hs = [], []
        for k, fixture in data['objects'].items():
            w, h = fixture['box']
            ws.append(w)
            hs.append(h)
            tiles.append(BTile(area=w * h, name=k))

        ws = np.asarray(ws)
        hs = np.asarray(hs)

        box_list = BoxInputList(tiles)

        stage1 = Stage(
            box_list,
            forms=[BoundsXYWH(box_list, w=bw, h=bh),
                   FixedDimension(box_list, values=[ws, hs]),
                   FeasibleSet()
                   ]
        )
        p = stage1.make(verbose=True)
        print(describe_problem(p))
        # print(p.objective.expr.curvature)

        # stage1.solve(verbose=True,
        #              # solve_args={'solver': 'ECOS'})
        #              )

        assert stage1.is_solved is True
        save_pth = self._save_loc(save='p2_sfix')
        print(box_list.describe())
        for t in tiles:
            print(t.area)
        stage1.display(save=save_pth, extents=[-1, 6, -1, 10])

    def test_appr(self):
        from scipy.spatial.distance import pdist
        # https://stackoverflow.com/questions/47147813/finding-the-optimized-location-for-a-list-of-coordinates-x-and-y
        """ (Reduced) Data """
        # removed a duplicate point (for less strange visualization)
        x = np.array(
            [13, 10, 12, 13, 11, 12, 11, 13, 13, 14, 15, 15, 16, 18, 2, 3, 4, 6, 9, 1])  # ,3,6,7,8,10,12,11,10,30])
        y = np.array(
            [12, 11, 10, 9, 8, 7, 6, 6, 8, 11, 12, 13, 15, 14, 18, 12, 11, 10, 13, 15])  # ,16,18,17,16,15,14,13,12,3])
        N = x.shape[0]
        M = 10
        print(N, M)

        """ Mixed-integer Second-order cone problem """
        c = cvx.Variable(2)  # center position to optimize
        d = cvx.Variable(N)  # helper-var: lower-bound on distance to center
        d_prod = cvx.Variable(N)  # helper-var: lower-bound on distance * binary/selected
        b = cvx.Variable(shape=N, boolean=True)  # binary-variables used for >= M points

        dists = pdist(np.vstack((x, y)).T)
        U = np.amax(dists)  # upper distance-bound (not tight!) for bigM-linearization

        helper_expr_c_0 = cvx.vec(c[0] - x)
        helper_expr_c_1 = cvx.vec(c[1] - y)

        helper_expr_norm = cvx.norm(cvx.vstack([helper_expr_c_0, helper_expr_c_1]), axis=0)

        print(helper_expr_c_0.shape)
        print(d.shape, helper_expr_norm.shape)
        constraints = []
        constraints.append(cvx.sum(b) >= M)  # M or more points covered
        constraints.append(d >= helper_expr_norm)  # lower-bound of distances
        constraints.append(d_prod <= U * b)  # linearization of product
        constraints.append(d_prod >= 0)  # """
        constraints.append(d_prod <= d)  # """
        constraints.append(d_prod >= d - U * (1 - b))  # """

        objective = cvx.Maximize(cvx.min(d_prod))

        problem = cvx.Problem(objective, constraints)
        print(describe_problem(problem))
        problem.solve(qcp=True, verbose=False)
        print(problem.status)

