__author__ = 'Xinyue'
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import dccp
from cvxpy import Problem, Variable, Minimize

if __name__ == '__main__':
    np.random.seed(0)
    n = 10
    r = np.linspace(1, 5, n)

    c = Variable((n, 2))
    constr = []
    for i in range(n-1):
        for j in range(i+1, n):
            constr.append(cvx.norm(cvx.vec(c[i, :]-c[j, :]), 2) >= r[i]+r[j])

    # prob = Problem(
    #     Minimize(
    #         cvx.max(
    #             cvx.max(cvx.abs(c), axis=1) + r     # max dim = c_max + r
    #         )
    #     ),
    #     constr
    # )

    prob = Problem(
        Minimize(
            cvx.max(cvx.abs(c[:, 0]) + r) +
            cvx.max(cvx.abs(c[:, 1]) + r)
            ),
        constr
    )

    print(prob.is_dcp(), prob.is_dcp())
    print(dccp.is_dccp(prob))
    prob.solve(method='dccp', solver='ECOS', ep=1e-2, max_slack=1e-2, verbose=True)

    l = cvx.max(cvx.max(cvx.abs(c), axis=1)+r).value*2
    pi = np.pi
    ratio = pi*cvx.sum(cvx.square(r)).value/cvx.square(l).value
    print("ratio =", ratio)

    # plot
    plt.figure(figsize=(5, 5))
    circ = np.linspace(0, 2*pi)
    x_border = [-l/2, l/2, l/2, -l/2, -l/2]
    y_border = [-l/2, -l/2, l/2, l/2, -l/2]
    for i in range(n):
        plt.plot(c[i, 0].value+r[i]*np.cos(circ), c[i, 1].value+r[i]*np.sin(circ),'b')
    plt.plot(x_border, y_border, 'g')
    plt.axes().set_aspect('equal')
    plt.xlim([-l/2, l/2])
    plt.ylim([-l/2, l/2])
    plt.show()

