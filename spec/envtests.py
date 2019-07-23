from unittest import TestCase
from src.experiment import *
from src.problem import Problem, PointsInBound, InitializeFullPriority
from src.problem.example import *
import src.algo.metroplis_hastings as algo
import src.objectives as objectives
import numpy as np
import src.utils
from PIL import Image
from src.actions.basic import *
from scipy import optimize

random.seed(69)


class TestLayouts(TestCase):
    """ test the layout objects do what they need to be doing """
    pass


class TestMHEnv(TestCase):
    def setUp(self):
        """ setup constraints and searcher"""
        problem = setup_house_L(size=(40, 40))

        env = MetroLayoutEnv()

        costfn = objectives.ConstraintsHeur(problem,
                                            wmap={'AspectConstraint':0.1,
                                                  'AreaConstraint': 2
                                                  },
                                            default=1.)

        model = algo.MetropolisHastings(env, costfn)

        self.exp = SimpleMH(
            env,
            problem,
            model=model,
            cost_fn=costfn,
            num_iter=1000,
            initializer=PointsInBound(problem, env, size=3, seed=69)
        )

    def test_init_state(self):
        exp = self.exp
        assert len(exp.problem) == 8

        fp = exp.problem.footprint
        assert fp is not None
        print('footprint,', fp)
        init_state = exp._initializer()
        init_state.show()
        assert init_state is not None
        print('init_state\n:', init_state)
        cost = exp.cost(init_state)
        assert cost is not None
        print(cost)
        #

    def test_img(self):
        exp = self.exp
        init_state = exp._initializer()
        nparr = init_state.to_numpy()
        print(nparr)
        img = Image.fromarray(nparr)
        img.show()

    def test_logf(self):
        exp = self.exp
        init_state = exp._initializer()
        cost = exp.cost(init_state)

        # c1 = sum(v for v in cost1.values())
        # init_state.show()
        param = dict(grow=True, name='bed1', on_x=True, dist=2)
        new_state1 = slide_wall(init_state, **param)
        c1 = exp.cost(new_state1)

        param = dict(grow=False, name='bed1', on_x=True, dist=1)
        new_state2 = slide_wall(init_state, **param)
        c2 = exp.cost(new_state2)
        # c2 = sum(v for v in cost2.values())
        print(cost, c1, c2)
        value = np.exp(cost - c2)
        print(np.exp(cost - c1), np.exp(cost - c2))
        assert cost > c1
        assert cost < c2
        assert np.exp(cost - c1) > np.exp(cost - c2)
        src.utils.plotpoly([new_state1, new_state2])

    def test_param(self):
        exp = self.exp
        init_state = exp._initializer(size=(40, 40))
        for i in range(10):
            print(slide_wall_params(init_state))

    def test_run(self):
        chain = self.exp.run()
        print(len(chain))
        ixs = len(chain) // 12
        show = []
        for i in range(12):
            show.append(chain[i  * ixs])

        src.utils.plotpoly(show)


class TestConcEnv(TestMHEnv):
    def setUp(self):
        """ setup constraints and searcher"""
        problem = setup_2room_rect(size=(40, 40))

        env = MetroLayoutEnv()

        costfn = objectives.ConstraintsHeur(problem, wmap=None, default=1.)

        model = algo.ConstrainedLeastSquares(env, costfn)
        self.exp = SimpleMH(
            env,
            problem,
            model=model,
            cost_fn=costfn,
            num_iter=1000,
            initializer=InitializeFullPriority(problem, env, size=3)
        )

    def test_cost(self):

        x0_rosenbrock = np.array([2, 2])

        def fun_rosenbrock(x):
            return np.array([10 * (x[1] - x[0] ** 2), (1 - x[0])])

        res_1 = optimize.least_squares(fun_rosenbrock, x0_rosenbrock)

        print(res_1.x)
        print(res_1.cost)

        res = optimize.least_squares(fun_rosenbrock, x0_rosenbrock,
                            bounds=([-np.inf, 1.5], np.inf))
        print(res.x)
        print(res.cost)

    def test_gh(self):
        def matr_t(t):
            return np.array([[t[0], 0], [t[2] + complex(0, 1) * t[3], t[1]]])

        def target(t):
            arr2 = matr_t(t)
            ret = 0
            for i, v1 in enumerate(arr):
                for j, v2 in enumerate(v1):
                    ret += abs(arr[i][j] - arr2[i][j]) ** 2
            return ret

        def con(t):
            return t[0] + t[1] - 1

        cons = {'type': 'eq', 'fun': con}
        def con_real(t):
            return np.sum(np.iscomplex(t))

        cons = [{'type': 'eq', 'fun': con},
                {'type': 'eq', 'fun': con_real}]

        # optimize.minimize(func, x0, constraints=cons)

    def test_cvx1(self):
        from cvxopt import matrix
        from cvxopt.solvers import lp

        A = matrix([[-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0]])  # constraints matrix
        b = matrix([1.0, -2.0, 0.0, 4.0])  # constraint RHS vector
        c = matrix([2.0, 1.0])  # cost function coefficients
        sol = lp(c, A, b)

    def test_opt2(self):
        """
        http://apmonitor.com/che263/index.php/Main/PythonOptimization
        """
        from scipy.optimize import minimize

        def objective(x):
            return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

        def constraint1(x):
            return x[0] * x[1] * x[2] * x[3] - 25.0

        def constraint2(x):
            sum_eq = 40.0
            for i in range(4):
                sum_eq = sum_eq - x[i] ** 2
            return sum_eq

        # initial guesses
        x0 = np.asarray([1.0, 5.0, 5.0, 1.0])
        # show initial objective
        print('Initial Objective: ' + str(objective(x0)))

        # optimize
        b = (1.0, 5.0)
        bnds = (b, b, b, b)
        con1 = {'type': 'ineq', 'fun': constraint1}
        con2 = {'type': 'eq', 'fun': constraint2}
        cons = (con1, con2)
        solution = minimize(objective, x0, method='SLSQP',
                            bounds=bnds, constraints=cons)
        x = solution.x

        # show final objective
        print('Final Objective: ' + str(objective(x)))

        # print solution
        print('Solution')
        print('x1 = ' + str(x[0]))
        print('x2 = ' + str(x[1]))
        print('x3 = ' + str(x[2]))
        print('x4 = ' + str(x[3]))

    def test_opt3(self):
        bounds = []
        constraints = []
        problem = self.exp.problem
        init_layuot = self.exp._initializer()
        xmn, ymn, xmx, ymx = problem.footprint.bounds

        for const in problem.constraints():
            constraints.extend(const.qp(problem))

        for i in range(len(problem)):
            bounds.append([xmn, ymn, xmx, ymx])
            bounds.append([xmn, ymn, xmx, ymx])
            bounds.append(None)
            bounds.append(None)

        x0 = init_layuot.to_vec4()
        res = optimize.minimize(self.exp.cost, x0, method='SLSQP',
                                bounds=bounds)
        print(res)

    def test_opt4(self):
        problem = self.exp.problem
        # init_layuot = self.exp._initializer()
        constraints = []
        for const in problem.constraints():
            constraints.extend(const.to_qp(problem))

        bounds = []
        xmn, ymn, xmx, ymx = problem.footprint.bounds
        print(xmn, ymn, xmx, ymx )
        area = problem.footprint.area
        for i in range(len(problem)):
            bounds.append([xmn, xmx])
            bounds.append([ymn, ymx])
            bounds.append([4., 30.])
            bounds.append([4., 30.])

        r1 = Room.from_geom(geometry.box(0, 0, 10, 10), **problem['r1'].kwargs)
        r2 = Room.from_geom(geometry.box(10, 10, 20, 20), **problem['r2'].kwargs)
        init_layuot = BuildingLayout(problem, rooms=[r1, r2])
        x0 = init_layuot.to_vec4()
        print(x0)

        def costfn(X):
            cost = np.double(0.0)
            # print(X)
            for i in range(2):

                x_i, y_i, u_i, v_i = X[4*i:4*i+4]
                # print(c1)
                cost += (150 - 4 * u_i * v_i)
                # cost += (u_i + v_i) # **2
                cost += ((u_i - v_i) / np.max([u_i , v_i])) ** 2

            return cost

        def grad_(X):
            return
        print(costfn(x0))
        res = optimize.minimize(costfn, x0,
                                method='Newton-CG', # 'COBYLA',
                                bounds=bounds,
                                # loss='soft_l1',
                                constraints=tuple(constraints))
        print(res)
        print(costfn(res.x))

    def test_optnn(self):
        problem = self.exp.problem
        init_layuot = self.exp._initializer()
        x0 = init_layuot.to_vec4()

