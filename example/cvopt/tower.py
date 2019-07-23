import cvxpy as cvx


def tower_simple(n=6, m=5):
    """
    a tower parametrized by N curves each with H points in R3

    - top floor of tower area < bottom floor area
    - height must be atleast H
    -

    minimizable
    - total length of lines

    """
    X = cvx.Variable(shape=(n, m), pos=True)

    for i in range(n):
        pass
        # constraints on point[i] to point[i+-1] on level
        # constraints on point[i] to point[i+-1] on level

    # corner conditions of f(X)
    # joint conditions  of f(X)
    # edge conditions   of f(X)

    return


def adaptive_truss():
    """
    a truss parametrized by 3 sets of N points

    each tuple (x, y, z) correspond to bottom1, top, bottom2

    constraints for gable roof
        y.h >= x.h
        y.h >= z.h

    minimize area or roof (sum of sections)
    probably will be approximated
    """

    return


def router():
    """

    """
    return


def structure_location(n, m):
    """"""
    return


def infill_placement():
    """

    """
    return


