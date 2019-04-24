from abc import ABC, abstractmethod


class LayoutAlgo(ABC):
    def __init__(self, log=False, log_every=1000):
        self._log_every = log_every
        self._log = log
        self._step = 0

    def log(self, *xs):
        if self._log and self._step % self._log_every == 0:
            print(xs)


class Constraint(ABC):
    pass


class AdjacencyGraph(ABC):
    """"""
    pass


class Tiling(ABC):
    """
    Given a polygon, 
    """ 
    pass


class Refiniment(ABC):
    """ """
    pass


class InsideOutLayout(LayoutAlgo):
    """
    stage 1: create graph with basic structure. 
        assign usage by private-public reqs
    stage 2: 
        Generate 2d positions

    stage 3: 
        Grow rooms by exerting 'pressure' on edge nodes
    
    Datastructures:
        graph, 
    """
    pass


class GrowthLayout(LayoutAlgo):
    """
    stage 1: create adjacency graph with room info/constraints.
    stage 2:
        place 'feature' where all constraints are met.
        expand (on grid) ? with cost at each step
    """
    pass


class Subdivision(LayoutAlgo):
    """
    """
    pass


class TilePlacement(LayoutAlgo):
    """
    """


class CostFunction(ABC):
    """
    """
    pass


class Layout(ABC):
    # @abstractmethod
    # def geometry(self):
    #     """ get all geometric objects """
    #     pass

    @abstractmethod
    def rooms(self):
        """ rooms which are part of program """
        pass




    # @abstractmethod
    # def outline(self):
    #     pass
    #
    # @abstractmethod
    # def groups(self):
    #     """ groups of rooms """
    #     pass
    #
    # def __hash__(self):
    #     pass

