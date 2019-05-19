from abc import ABC, abstractmethod
from src.building import Area
import src.layout_ops as lops
from collections import defaultdict as ddict
from collections import OrderedDict as odict


class _ProblemBase(ABC):
    @property
    @abstractmethod
    def program(self):
        pass


class Problem:
    """ Stores constraints, specifications etc """
    def __init__(self, program={}):
        self._program = program
        self._name_to_index = {}
        self._index_to_name = {}
        self._constraints = []
        self._adj = ddict(set)
        self._footprint = None

    def __getitem__(self, item):
        return self._program[item]

    def __contains__(self, item):
        return item in self._program

    def __len__(self):
        return len(self._program)

    def __repr__(self):
        st = 'Program:'
        for k, v in self._program.items():
            st += '\n' + str(v)
        st += 'Constraints:'
        for v in self._constraints:
            st += '\n' + str(v)
        return st

    def index_of(self, item):
        return self._name_to_index.get(item)

    @property
    def G(self):
        return self._adj

    @property
    def program(self):
        return list(self._program.values())

    def constraints(self, otype=None):
        if otype is None:
            return self._constraints
        return [x for x in self._constraints if isinstance(x, otype)]

    @property
    def footprint(self):
        return self._footprint

    @footprint.setter
    def footprint(self, footprint):
        self._footprint = footprint

    def add_program(self, prog_ent):
        self._index_to_name[len(self)] = prog_ent.name
        self._name_to_index[prog_ent.name] = len(self)
        self._program[prog_ent.name] = prog_ent

    def add_constraint(self, constraint) -> None:
        assert isinstance(constraint, lops.Constraint), 'invalid constraint'
        if isinstance(constraint, lops.AdjacencyConstraint):
            self._adj[constraint._ent].add(constraint._ent2)
            self._adj[constraint._ent2].add(constraint._ent)
        self._constraints.append(constraint)


class ProgramEntity(object):
    def __init__(self, name, prog_type, problem=None):
        self.name = name
        self.prog_type = prog_type
        self._problem = problem
        if problem and name not in problem:
            problem._program[name] = self
        elif problem and name in problem:
            print('WARNING duplicate item added ', name)

    @property
    def kwargs(self):
        return dict(name=self.name, program=self.prog_type)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name
        return False

    def __str__(self):
        return '{}: {}, program {}'.format(
            self.__class__.__name__, self.name, self.prog_type
        )


class DictProblem(Problem):
    def __init__(self, program={}, target_state=None, footprint=None):
        """

        :param program:     dictionary with constraints by type
        :param target_state: tensor of size [C, N, M]
        :param footprint:
        """
        Problem.__init__(self)
        self._program = program
        self._footprint = footprint
        self._target_state = target_state

    @property
    def program(self):
        return self._program


