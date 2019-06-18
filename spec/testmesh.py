import unittest
import networkx as nx
from src.cvopt.spatial import *
import transformations as xf


class TestMesh(unittest.TestCase):
    def test_bnd(self):
        M = Mesh2d(g=nx.grid_2d_graph(7, 8))
