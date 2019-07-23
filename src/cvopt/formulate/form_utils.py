import numpy as np
import cvxpy as cvx
from cvxpy.utilities.canonical import Canonical
from cvxpy.atoms.atom import Atom, Expression


def triu_indices(n, diag=1):
    tri_i, tri_j = np.triu_indices(n, diag)
    return zip(tri_i.tolist(), tri_j.tolist())


def expr_tree_detail(problem, offs=0, sign=None):
    st = ''
    if isinstance(problem, cvx.Problem):
        atoms = problem.objective.args
    elif isinstance(problem, Atom):
        atoms = problem.args
    elif isinstance(problem, Expression):
        atoms = problem.args
    else:
        raise Exception('not problem or list', type(problem))

    for atom in atoms:
        t_name = type(atom).__name__
        vars = set([x.name() for x in atom.variables()])
        print(' ' * offs, t_name, atom.curvature, atom.sign, vars, sign)
        for i, arg in enumerate(atom.args):
            expr_tree_detail(arg, offs+2, atom.is_incr(i))

