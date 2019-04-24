import numpy as np
import scipy.stats
from src.interfaces import LayoutAlgo
import scipy.optimize as optim


transition_model = lambda x: [x[0], np.random.normal(x[1], 0.5, (1,))]


def prior(x):
    """
    https://towardsdatascience.com/from-scratch-bayesian-inference-markov-chain-monte-carlo-and-metropolis-hastings-in-python-ef21a29e25a
    x[0] = mu, x[1]=sigma (new or current)
    returns 1 for all valid values of sigma. Log(1) =0, so it does not affect the summation.
    returns 0 for all invalid values of sigma (<=0). Log(0)=-infinity, and Log(negative number) is undefined.
    It makes the new sigma infinitely unlikely.
    """
    if x[1] <= 0:
        return 0
    return 1


def manual_log_like_normal(x, data):
    """
    # Computes the likelihood of the data given a sigma (new or current) according to equation (2)
    #x[0]=mu, x[1]=sigma (new or current)
    #data = the observation
    """
    return np.sum(-np.log(x[1] * np.sqrt(2 * np.pi))-((data-x[0])**2) / (2*x[1]**2))


def log_lik_normal(x, data):
    """ Same as manual_log_like_normal(x,data), but using scipy implementation. It's pretty slow. """
    return np.sum(np.log(scipy.stats.norm(x[0], x[1]).pdf(data)))


def acceptance(x, x_new):
    """
    Defines whether to accept or reject the new sample
    """
    if x_new > x:
        return True
    else:
        accept = np.random.uniform(0, 1)
        # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
        # less likely x_new are less likely to be accepted
    return accept < (np.exp(x_new-x))


def metropolis_hastings(likelihood_computer, prior, transition_model, param_init, iterations, data, acceptance_rule):
    """
    # likelihood_computer(x,data): returns the likelihood that these parameters generated the data
    # transition_model(x): a function that draws a sample from a symmetric distribution and returns it
    # param_init: a starting sample
    # iterations: number of accepted to generated
    # data: the data that we wish to model
    # acceptance_rule(x,x_new): decides whether to accept or reject the new sample
    """
    x = param_init
    accepted = []
    rejected = []
    for i in range(iterations):
        x_new = transition_model(x)
        x_lik = log_lik_normal(x, data)
        x_new_lik = log_lik_normal(x_new, data)
        if acceptance(x_lik + np.log(prior(x)), x_new_lik + np.log(prior(x_new))):
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)

    return np.array(accepted), np.array(rejected)


class MetropolisHastings(LayoutAlgo):
    def __init__(self,
                 env,
                 cost_fn,
                 beta=1.,
                 num_iter=10000,
                 log_every=1):
        LayoutAlgo.__init__(self, log=True, log_every=log_every)
        self._niter = num_iter
        self._env = env
        self._beta = beta
        self._cost_fn = cost_fn

    def acceptance_fn(self, state, prior):
        value = np.exp(self._beta * (self._cost_fn(prior) - self._cost_fn(state)))
        res = np.random.uniform(0.9, 1) < value
        if res is True:
            self.log(self._step, value)
        return res

    def __call__(self, X, **kwargs):
        X_new = self._env.transition_model(X)
        if self.acceptance_fn(X_new, X):
            self._step += 1
            return X_new
        else:
            return X

    def run(self, init_state, **kwargs):
        accepted = [init_state]
        rejected = []

        for i in range(self._niter):
            X = self.__call__(accepted[-1])
            if X == accepted[-1]:
                accepted.append(X)
            else:
                rejected.append(X)

        return accepted, rejected


class StateSearch(LayoutAlgo):
    def __init__(self, **kwargs):
        LayoutAlgo.__init__(self, **kwargs)


class SimulatedAnal(LayoutAlgo):
    def __init__(self,
                 env,
                 cost_fn,
                 beta=1.,
                 num_iter=10000,
                 log_every=1):
        LayoutAlgo.__init__(self, log=True, log_every=log_every)
        self._env = env
        self._cost_fn = cost_fn

    @staticmethod
    def tree_action(trees, tree_size):
        # ln = tree.__len__()
        tree_idx = np.random.randint(0, tree_size)
        node_idx = np.random.randint(0, tree_size)
        move = np.random.randint(-2, 2, 2)

        tree = trees[tree_idx]
        node = trees[tree_idx].update(node_idx, move)

        return trees

    def run(self, trees, num_iter=10000, tabu=False):
        """
        sim anneal
        Let s = s0
        For k = 0 through kmax (exclusive):
            T ← temperature(k ∕ kmax)
            Pick a random neighbour, snew ← neighbour(s)
            If P(E(s), E(snew), T) ≥ random(0, 1):
                s ← snew
        Output: the final state s

        nd exp ⁡ ( − ( e ′ − e ) / T ) {\displaystyle \exp(-(e'-e)/T)} \exp(-(e'-e)/T) otherwise.
        todo finish this shoit
        """
        num_moves = 4
        kmax = num_iter
        tree_size = sum(tree.__len__() for tree in trees)

        hist = []
        cost = self._cost_fn(trees_to_layout(trees).to_vec4())
        state = trees

        print(cost)
        for k in range(kmax):
            temp = k / kmx

            # returns new tree
            snew = tree_action(state, tree_size * num_moves)
            cost_new = self._cost_fn(trees_to_layout(snew).to_vec4())
            if cost_new < cost:
                cost = cost_new
                state = snew
            elif np.exp(- (cost - cost_new) / temp) < np.random.uniform():
                cost = cost_new
                state = snew
        print(cost)
        return state


class ConstrainedLeastSquares(LayoutAlgo):
    """
    NonLinear Least Squares
    Objective function

    The above optimization is a non-linear least squares
    problem subject to linear equality and inequality con-
    straints. We use the active set algorithm implemented
    by the minbleic package in ALGLIB library to solve for
    the optimum configuration.


    """
    def __init__(self, env, cost_fn, loss='soft_l1', **kwargs):
        LayoutAlgo.__init__(self, **kwargs)
        self._env = env
        self._cost_fn = cost_fn
        self._loss = loss

    def serialize(self, layout):
        return layout

    def gen_random(self):
        """ generate training data """
        pass

    def __call__(self, init_layout, train_data=False, **kwargs):
        """
        Convert Constraints to bounds
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
        """
        # http://www.alglib.net/optimization/boundandlinearlyconstrained.php
        # xs = [2, 2]
        # bounds = ([-np.inf, 1.5], np.inf)
        bounds = []
        for const in init_layout.problem.constraints():
            lb = -np.inf if const._min is None else const._min
            ub = +np.inf if const._max is None else const._max
            bounds.append([lb, ub])

        X_0 = init_layout.to_vec4()

        # if train_data is True:
        #    t_train = np.linspace(t_min, t_max, n_points)
        #    y_train = gen_data(t_train, a, b, c, noise=0.1, n_outliers=3)
        res = optim.least_squares(self._cost_fn, X_0,
                                  loss='soft_l1',
                                  # args=(t_train, y_train),
                                  bounds=bounds)

        # optim.lsq_linear(A, b, bounds=[], method='blvs')
        res2 = optim.minimize(self._cost_fn,
                              X_0,
                              method="L-BFGS-B"
                              )
        # optim.LinearConstraint
        """
        https://cvxopt.org/examples/tutorial/qp.html
        position
        
        adjacent 
             < r1_x - r2_x 
        
        
        """
        return res

    def bleic(self, init_layout):
        # http://www.alglib.net/translator/man/manual.cpython.html#example_minbleic_d_2
        state = xalglib.minbleiccreate(x)
        return

    def run(self, init_state, **kwargs):
        pass
