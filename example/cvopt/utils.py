import numpy as np


np.random.seed(0)


def adj_mat(n):
    m = np.random.randint(0, 2, (n, n))
    m[np.tril_indices(n)] = 0
    return m


def weighted_adj_mat(n, mx=10):
    m = adj_mat(n)
    ixs = np.where(m == 1)
    w = np.random.randint(1, mx, ixs[0].shape[0])
    m[ixs] = w
    return m


def adj_mat_n(n):
    cij = np.zeros((n, n))
    if n == 5:
        cij[0, 1] = 2
        cij[0, 2] = 5
        cij[1, 2] = 5
        cij[1, 3] = 7
        cij[2, 4] = 1
        cij[2, 3] = 2
        cij[3, 4] = 6
    elif n == 3:
        cij[0, 1] = 2
        cij[0, 2] = 5
        cij[1, 2] = 0
    return cij

