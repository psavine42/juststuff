from cvxpy import Problem
import dccp
import dmcp
from collections import Counter
from collections import defaultdict as ddict


def describe_problem(problem: Problem):
    """
    DMCP https://github.com/cvxgrp/dmcp
    dccp
    """
    st = 'Curvature {}'.format(problem.objective.expr.curvature)
    st += '\nis disciplined quasiconvex    {}'.format(problem.is_dqcp())
    st += '\nis disciplined geometric      {}'.format(problem.is_dgp())
    st += '\nis disciplined quadratic      {}'.format(problem.is_qp())
    st += '\nis disciplined convex         {}'.format(problem.is_dcp())
    st += '\nis disciplined concave-convex {}'.format(dccp.is_dccp(problem))
    st += '\nis disciplined multi-convex   {}'.format(dmcp.is_dmcp(problem))
    # todo
    # SQP sequential quadratic program
    # SCP seperable convex program
    #
    return st


class FPProbllem(object):
    def __init__(self):
        self._problem = None
        self._constraints = None
        self.G = None
        self._formulations = []
        self._placements = []
        self._meta = {}

    def action_eliminators(self):
        return []

    def print(self, problem):
        print('Problem:----------')
        print('----------')
        print(problem.solution.status)
        print(problem.solution.opt_val)
        print(problem.solution.attr)

    def make(self, verbose=False, obj_args={}, const_args={}):
        constraints = self.own_constraints(**const_args)

        objective = self.objective(**obj_args)
        if verbose is True:
            print('Constraints')
            print('---------------------------')
            for c in constraints:
                print(c)
            print('Objective')
            print('---------------------------')
            print(objective)
            print('problem ready')
        self._problem = Problem(objective, constraints)
        return self._problem

    def run(self, obj_args={}, const_args={}, solve_args={}, verbose=False, **kwargs):
        if self._problem is None:
            self.make(verbose=verbose,
                      const_args=const_args,
                      obj_args=obj_args)
        self._problem.solve(verbose=verbose, **solve_args)
        if self._problem.solution.status == 'infeasible':
            print(self._problem._solver_stats.__dict__)
            for x in self._problem.constraints:
                print(x)

        print('solution created')
        print(self._problem.solution)
        #if show:
        #    self.display(self._problem, save=save)
        return self.solution

    def solve(self, **kwargs):
        return self.run(**kwargs)

    def serialize(self):
        """
        {'linkToNode': {
            '_4vnql30j2': ['AdjacencyEdgeJoint.0.output.canonicle',
                           'GridLine.0.input.edges'],
            '_gp7xax837': ['GeomContains.0.input.inner',
                           'AdjacencyEdgeJoint.0.output.canonicle'],
            '_wam2iykmp': ['AdjacencyEdgeJoint.0.output.canonicle',
                           'AdjacencyEdgeJoint.0.output.canonicle']},
        'nodeToLink': {
            'AdjacencyEdgeJoint.0.output.canonicle': [
                '_4vnql30j2',
                '_wam2iykmp',
                '_gp7xax837'],
            'GeomContains.0.input.inner': ['_gp7xax837'],
            'GridLine.0.input.edges': ['_4vnql30j2']}
        'active': [
            'AdjacencyEdgeJoint.0' ,
            'GridLine.0'
            ]
        }
        :return:
        """
        node_to_link = {}
        link_to_node = {}
        active = []
        cnt = ddict(int)
        for f in self._formulations:
            name = f.__class__.__name__
            if name in cnt:
                cnt[name] += 1
            else:
                cnt[name] = 0

            inst_name = name + '.' + str(cnt[name])
            active.append(inst_name)

            # outputs can be
            inputs = f.inputs

        solution = None
        if self._problem is not None:
            if self._problem.solution is not None:
                solution = str(self._problem.solution)

        json_dict = {
            'solution': solution,   # todo
            'image': None,
            'active': active,
            'nodeToLink': node_to_link,
            'linkToNode': link_to_node
        }
        return json_dict

    @property
    def formulations(self):
        return self._formulations

    @property
    def placements(self):
        return self._placements

    # implement in supercalss ----------------
    @property
    def solution(self):
        return

    @property
    def domain(self):
        return self.G

    @property
    def meta(self):
        return self._meta

    @property
    def problem(self):
        return self._problem

    def _anchor(self, *args):
        return

    def _pre_compute_placements(self):
        return

    def own_constraints(self, **kwargs):
        raise NotImplemented('not implemented in base class')

    def objective(self, **kwargs):
        raise NotImplemented('not implemented in base class')

    def display(self, problem, **kwargs):
        return



