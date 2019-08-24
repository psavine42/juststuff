from cvxpy import Variable
from cvxpy.lin_ops.lin_utils import transpose
import cvxpy as cvx



def enumerate_tilings(mesh, templates):
    cnt = 0
    tilings_i_x = {}
    for node in mesh.nodes:
        for template in templates:
            if mesh.can_place(node, template):
                tilings_i_x[cnt] = {node: template}
                cnt += 1
    return tilings_i_x


def anchorShape(shp, n, m, p, q, noncov=[]):
    pts = [(p + x, q + y) for x,y in shp]
    if all(0 <= x and x<n and 0<=y and y < m and (x, y) not in noncov for x,y in pts):
        return pts
    else:
        return None


def indicator(a, x, b, M):
    z = Variable(shape=len(x), boolean=True)
    return a @ x <= b + M * (1 - z)


def disjunction(x, b, M, a=1):
    """

    aTx * b <= M

    (aT1 x ≤ b1)∨ (aT2x ≤ b2)∨ ⋯ ∨ (aT_k x ≤ bk).
    z1+⋯+zk≥1,
    z1,…,zk ∈ {0,1},
    aTix ≤ bi+M(1−zi), i=1,…,k.

    Arguments
    ------
        - x Variable shape()
        - b Variable (same shape as b)
        - M Parameter (scalar) is an upper bound on aTx + b  <= M
        - a Parameter (scalar or matrix)
    """
    z = Variable(shape=len(x), boolean=True)
    zsum = cvx.sum(z) >= 1
    expr = a * x <= b + M * (1 - z)
    return [zsum, expr]


def boolean_not(lhs, rhs):
    assert isinstance(lhs, Variable) and isinstance(rhs, Variable)
    assert lhs.attributes['boolean'] is True
    assert rhs.attributes['boolean'] is True
    return [lhs == 1 - rhs]


def implies(iftrue, then):
    """
    MS foundation lanugage
    """
    return


def or_constraint(xs, y):
    """ y is true if any element in xs is True

    xs: Variable(boolean=True)
    y : Variable(boolean=True)

    returns list of Constraints
    """
    N = len(xs)
    if isinstance(xs, list):
        xs = cvx.hstack(xs)
    C = [
         -N + 1 <= cvx.sum(xs) - N * y,
         0      >= cvx.sum(xs) - N * y
    ]
    return C


def and_constraint(xs, y):
    """ y is true if all elements in xs is True

    xs: Variable(boolean=True)
    y : Variable(boolean=True)

    if dimenison of xs is 2 then:
        dimension 0 -> number of ands


    returns: list of Constraints
    """
    N = len(xs)
    if isinstance(xs, list):
        xs = cvx.hstack(xs)
    C = [
         0     <= cvx.sum(xs) - N * y,
         N - 1 >= cvx.sum(xs) - N * y
    ]
    return C


def xor_constraint(xs, y):
    """
    y is true if elements in x sum to odd. y is false if elements in x sum to even

    returns: list of Constraint Expressions
    """


def conditional_constraint():
    """
    f1 (x 1 , x 2 , . . . , x n ) > b1
    implies that f2 (x1 , x2 , . . . , x n ) ≤ b2 .
    """


def compound_alternative(fx_exprs, b, N):
    """
    f_i(x) falls in one of N disjoint regions b_i

    f 1 (x 1 , x 2 ) − B 1 y 1 ≤ b 1    Region 1 constraints
    f 2 (x 1 , x 2 ) − B 2 y 1 ≤ b 2

    f 3 (x 1 , x 2 ) − B 3 y 2 ≤ b 3    Region 2 constraints
    f 4 (x 1 , x 2 ) − B 4 y 2 ≤ b 4

    f 5 (x 1 , x 2 ) − B 5 y 3 ≤ b 5   Region 3 constraints
    f 6 (x 1 , x 2 ) − B 6 y 3 ≤ b 6 
    f 7 (x 1 , x 2 ) − B 7 y 3 ≤ b 7

    y1 + y2 + y3 ≤ 2,
    x1 ≥ 0, x2 ≥ 0,

    y1, y2 , y3 binary

    :return:
    """
    # N = len(fx_exprs)
    # Variable(shape=)






