import unittest
import networkx as nx
from src.cvopt.spatial import *
import transformations as xf
from src.cvopt.formulations import *
from example.cvopt.garage import *
from src.cvopt.floorplanexample import *
from src.cvopt.mesh.mapping import MeshMapping
from pprint import pprint


def re(exp, got):
    return 'expected {}, got {}'.format(exp, got)


class TestMap(unittest.TestCase):
    tgt = ((1, 4), (2, 4))
    def setUp(self):
        M = Mesh2d.from_grid(5, 7)
        template = TemplateTile(2, 1)
        template.add_color(1, half_edges=[0])
        xform = template.align_to(self.tgt).astype(int)
        self.m = MeshMapping(M, template, xform)

    def test_algn(self):
        template = self.m.base
        angles = [-math.pi/2, 0, math.pi/2, math.pi]
        b = (1, 4)
        mats = []
        alg_targets = [(b, v) for v in [(2, 4), (1, 5), (0, 4), (1, 3)]]
        for angle, tgt in zip(angles, alg_targets):
            mats.append(template.align_to(tgt).astype(int))
        assert len(np.unique(mats)) == 4

    def test1(self):
        M = Mesh2d(g=nx.grid_2d_graph(6, 8))
        template = TemplateTile(2, 1)
        template.add_color(1, half_edges=[0])

        xform = template.align_to( ((1, 3), (2, 3)))
        new_mesh = MeshMapping(M, template, xform)
        print('---------')
        tsf = new_mesh.transformed
        pprint(tsf.vertices())
        # pprint(tsf.G.get_data())
        # print(template.half_edges())
        pprint(new_mesh.vertex_map())
        pprint(new_mesh.edge_map())
        pprint(new_mesh.half_edge_map())
        pprint(new_mesh.face_map())
        return

    def test3(self):
        n = 6
        coords = itertools.product(range(n), range(n))
        tiles = [BTile(x) for x in coords]

        assert tiles[0].coords == [(0, 0), (0, 1), (1, 1), (1, 0)]
        M = Mesh2d(g=tiles)
        he = M.half_edges.values()
        assert len(he) == len(set(he))

        T = Mesh2d(g=[BTile(x) for x in itertools.product(range(2), range(2))])
        xform = np.eye(3)
        xform[0, 2] = 1
        xform[1, 2] = 2
        print(xform)
        mapped = MeshMapping(M, T, xform)
        assert mapped.is_valid()

        print('-----\n', mapped._data)
        print('-----')
        print(mapped.tile.vertices)
        # print(mapped.edge_map())
        # print(mapped.transformed.info())
        print('-----')
        print(Mesh2d.vertices.fget(mapped.transformed).geom)
        print(Mesh2d.vertices.fget(mapped.space).geom)
        print(mapped.vertex_map())
        print(mapped.half_edge_map())
        print(mapped.face_map())
        print(mapped.edge_map())
        # assert mapped.vertex_map() == mapped._data

    def test_show(self):
        self.m.show()

    def test_show_he(self):
        self.m.show(he=True)

    def test_save(self):
        self.m.show(save='/home/psavine/source/layout/data/testmap1.png')

    def test4(self):
        mapped = self.m
        M = mapped.space
        fmap = mapped.face_map()

    def test2(self):
        tgt = self.tgt
        mapped = self.m

        M = mapped.space
        tgt_ix = M.index_of_edge(tgt)

        n1_mix = M.index_of_vertex(tgt[0])
        n2_mix = M.index_of_vertex(tgt[1])
        print(mapped._data)

        print(mapped.vertex_map())
        print('f', mapped.face_map())
        assert mapped.vertex_map()[0] == n1_mix, re(n1_mix, mapped.vertex_map()[0])
        assert mapped.vertex_map()[1] == n2_mix, re(n2_mix, mapped.vertex_map()[0])

        # print(mapped._index)
        # assert mapped._rec_vert(tgt[0]) == 0, re(0, mapped._rec_vert(tgt[0]))
        assert mapped.base.vertices[0] == (0, 0), re((0,0), mapped.base.vertices()[0])
        assert mapped.base.vertices[1] == (0, 1), re((0, 1), mapped.base.vertices()[1])
        assert mapped.base.index_of_edge(((0, 0), (0, 1))) == 0

        print(mapped.edge_map())
        print(mapped.base.edges)

        assert set(mapped.faces) == {8, 9}, 'got {}'.format(mapped.faces)
        assert mapped.edges[0] == tgt_ix, re(tgt_ix, mapped.edges[0])
        assert mapped.edges[0:2] == [22, 34], re([22, 34], mapped.edges[0:2])

        assert mapped.faces[0] == 9, re(9, mapped.faces[0])

        assert mapped.boundary.vertices == [()]


# ---------------------------
class TestMap2(TestMap):
    """
    todo: test that mapping in all directions is ok
    -
    """
    tgt = ((2, 4), (1, 4))


# --------------------------------------------------------------------------
class LineLayouts(unittest.TestCase):
    def test1(self):
        s = 10
        M = Mesh2d(g=nx.grid_2d_graph(s, s))
        s += 1
        boxes = [[(0, 0), (5, 5)], [(0, 5), (5, s)],
                 [(5, 5), (s, s)], [(5, 0), (s, 5)]
                 ]
        sink_vertex = 3
        points = []

    def test_nf(self):
        s = 5
        M = Mesh2d(g=nx.grid_2d_graph(s, s))
        problem = n_free(M, 5, [0])
        problem.solve(qcp=True, verbose=True)
        print(problem.solution)


class TestFrm(unittest.TestCase):
    """

    """

    def mini_problem(self):
        tiles = parking_simple()
        M = Mesh2d.from_grid(6, 8)
        print('m build')
        # print(M.edges.to_half_edges)
        return FormPlan(tiles, M)

    def test_base(self):
        prob = self.mini_problem()
        prob.solve(show=False, verbose=True)
        prob.display(save='./data/opt/TestFrm_test_mini.png')

    def test_overlap(self):
        prob = self.mini_problem()
        c1 = NoOverlappingFaces(prob.G, is_constraint=True)
        prob.add_constraint(c1)
        prob.solve(show=False, verbose=True)
        prob.display(save='./data/opt/TestFrm_test_ovr.png')

    def test_gridlines(self):
        prob = self.mini_problem()
        road, lot = prob.placements
        c1 = GridLine(prob.G, is_constraint=True,
                      edges=[i for i in range(30, 42, 2)])
        prob.add_constraint(c1)
        assert c1.num_actions == len(road) + len(lot)

        prob.solve(show=False, verbose=True)
        prob.display(save='./data/opt/TestFrm_test_gridlines.png')

    def test_gridlines2(self):
        prob = self.mini_problem()
        road, lot = prob.placements
        prob.add_constraint(GridLine, is_constraint=True, edges=[i for i in range(15, 29, 2)])

        c1 = prob.formulations[0]
        assert c1.num_actions == len(road) + len(lot)

        prob.solve(show=False)
        prob.display(save='./data/opt/TestFrm_test_gridlines2.png')

    def test_gridlines3(self):
        prob = self.mini_problem()
        road, lot = prob.placements
        prob.add_constraint(GridLine, is_constraint=True, edges=[i for i in range(15, 29, 2)])

        c1 = prob.formulations[0]
        assert c1.num_actions == len(road) + len(lot)

        prob.solve(show=False)
        prob.display(save='./data/opt/TestFrm_test_gridlines2.png')

