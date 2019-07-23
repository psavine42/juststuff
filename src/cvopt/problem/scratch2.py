from .base import FPProbllem
from cvxpy import Variable, Maximize, Minimize, Problem
import cvxpy as cvx
from src.cvopt.mesh import Mesh2d


def dgp():
    # DGP requires Variables to be declared positive via `pos=True`.
    x = Variable(pos=True)
    y = Variable(pos=True)
    z = Variable(pos=True)

    objective_fn = x * y * z
    constraints = [
        4 * x * y * z + 2 * x * z <= 10,
        x <= 2 * y,
        y <= 2 * x,
        z >= 1
    ]
    problem = Problem(Maximize(objective_fn), constraints)
    problem.solve(gp=True)
    print("Optimal value: ", problem.value)
    print("x: ", x.value)
    print("y: ", y.value)
    print("z: ", z.value)


class TreeLikeProblem(FPProbllem):
    def __init__(self, source, sinks):
        """

        Solve a location of segments in a tree structure to form a layout (piping)

        generates a sequence of optimization problems in R3 corresponding to branching structures
        initial number of nodes is size of binary tree containing the sinks


        can be soft constrained by:
            - bounding boxes where nodes (branches) should be located

        can be hard constrained by:
            -

        minimze lengths of segments (X_i, X_j)

        source : point in r3
        sinks:  list of points in r3
        """
        FPProbllem.__init__(self)
        self._source = source
        self._sinks = sinks

    def add_constraint(self, *args):
        pass

    def solution(self):
        return

    @property
    def available_formulations(self):
        return []

    def __scratch(self):
        """

        """
        s0 = self._init_a_tree()
        num_nodes = len(s0)
        X = Variable(shape=(num_nodes, 3), pos=True)

    def objective(self, **kwargs):
        Minimize()


class MinMaxDelay(FPProbllem):
    def __init__(self, space: Mesh2d, source, sinks):
        """
        """
        FPProbllem.__init__(self)
        self.G = space
        self._source = source
        self._sinks = sinks
        #
        self.X = Variable(shape=len(self.G.vertices))
        self.T = Variable(shape=len(self.G.edges))

    def _anchor(self, *args):
        return

    def _pre_compute_placements(self):
        node_ij2_var_index = {}
        cnt = 0
        for edge_ix, ij in self.G.edges.base.items():
            i, j = list(ij)
            # node_ij2_var_index[cnt]

        return

    def own_constraints(self, **kwargs):
        return

    def objective(self, **kwargs):
        # minimze max(T_k ) if k is a source
        # T_k = 0 if k is a sink
        # T_k >= max( norm(x_j - x_k) + T_j | s.t. E arc from k to j )
        # ->
        # T_k >=
        return Minimize(cvx.max())

