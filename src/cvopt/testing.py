import cvxpy as cp
import numpy as np



def example_min_len_least_squares():
    n = 10
    np.random.seed(1)
    A = np.random.randn(n, n)
    x_star = np.random.randn(n)
    b = A @ x_star
    epsilon = 1e-2
    x = cp.Variable(n)
    mse = cp.sum_squares(A @ x - b)/n
    problem = cp.Problem(cp.Minimize(cp.length(x)), [mse <= epsilon])
    print("Is problem DQCP?: ", problem.is_dqcp())

    problem.solve(qcp=True)
    print("Found a solution, with length: ", problem.value)


def example_placement1():
    num_nodes = 10



def slide23():
    x = cp.Variable(10)
    cost = cp.sum_squares()


if __name__ == '__main__':
    example_min_len_least_squares()

