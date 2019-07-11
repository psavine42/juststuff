import unittest
from example.cvopt.garage import *
from src.cvopt.floorplanexample import *
from src.cvopt.problem import *
from src.cvopt.mesh.mapping import MeshMapping
from pprint import pprint


def re(exp, got):
    return 'expected {}, got {}'.format(exp, got)


def dumb_problem(tiles, w=2, h=3):
    M = Mesh2d.from_grid(w, h)
    print('m build')
    form = FormPlan(tiles, M)
    for t in tiles:
        assert len(t.colors(half_edge=True)) > 0
    x = {tuple(e.inv_map.keys()) for e in form._half_edges}
    assert len(x) > 1
    print('pre-checks passed')
    return form


class TestMap(unittest.TestCase):
    tgt = ((1, 4), (2, 4))
    def setup(self):
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

    def _setup_test(self, tgt):
        M = Mesh2d.from_grid(2, 3)
        template = TemplateTile(1, 1)
        template.add_color(1, half_edges=[0])
        xform = template.align_to(tgt).astype(int)
        mapping = MeshMapping(M, template, xform)
        assert mapping.is_valid() is True

    def test8(self):
        self._setup_test(((0, 0), (0, 1)))

    def test5(self):
        self._setup_test(((0, 1), (1, 1)))

    def test6(self):
        self._setup_test(((1, 1), (1, 0)))

    def test7(self):
        self._setup_test(((1, 0), (0, 0)))

    def test_base(self):
        self.test5()
        self.test6()
        self.test7()
        self.test8()

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
    def test_nf(self):
        s = 5
        M = Mesh2d(g=nx.grid_2d_graph(s, s))
        problem = n_free(M, 5, [0])
        problem.solve(qcp=True, verbose=True)
        print(problem.solution)

    def test_num_plc(self):
        p = dumb_problem(parking_simple())
        road_tile, lot_tile = p.placements
        print(str(lot_tile.template))
        # print(road_tile.template.boundary.ext_half_edges)
        assert len(road_tile) == 24, re(24, len(road_tile))
        assert len(lot_tile) == 14, re(14, len(lot_tile))


def env():
    T = TestFrm()
    return T.mini_problem()


const_args_formuls = dict(face=False,
                          edge=False,
                          tile=False,
                          vertex=False,
                          half_edge=False)

class TestFrm(unittest.TestCase):
    """

    """
    def mini_problem(self):
        tiles = parking_simple()
        M = Mesh2d.from_grid(6, 8)
        print('m build')
        form = FormPlan(tiles, M)
        for t in tiles:
            assert len(t.colors(half_edge=True)) > 0

        x = {tuple(e.inv_map.keys()) for e in form._half_edges}
        assert len(x) > 1

        print('pre-checks passed')
        return form

    def test_base(self):
        prob = self.mini_problem()
        prob.solve(show=False, verbose=True)
        prob.display(save='./data/opt/TestFrm_test_mini.png')

    def test_overlap(self):
        const_args = dict(face=False,
                          edge=True,
                          tile=False,
                          vertex=False,
                          half_edge=False)
        obj_args = dict(edge=True)
        prob = self.mini_problem()
        c1 = NoOverlappingFaces(prob.G, is_constraint=True)
        prob.add_constraint(c1)
        x = {tuple(e.inv_map.keys()) for e in prob._half_edges}
        print(x)
        assert len(x) > 1

        prob.solve(show=False, verbose=True,
                   obj_args=obj_args,
                   const_args=const_args)
        prob.display(save='./data/opt/TestFrm_test_ovr.png')

    def test_overlap2(self):
        obj_args = dict(edge=False)
        tiles = parking_simple()
        prob = dumb_problem(tiles)

        # constraints
        c1 = NoOverlappingFaces(prob.G, is_constraint=True)
        c2 = AdjacencyEC(prob.G, is_constraint=False)
        c3 = TileLimit(prob.G, 0, upper=2)

        prob.add_constraint(c1, c2, c3)
        prob.solve(show=False, verbose=True,
                   obj_args=obj_args,
                   const_args=const_args_formuls)
        prob.display(save='./data/opt/TestFrm_test_ovr2.png')

    def test_overlap3(self):
        tiles = parking_tiling_nd()
        prob = dumb_problem(tiles, 5, 7)

        # constraints
        c1 = NoOverlappingFaces(prob.G, is_constraint=True)
        c2 = AdjacencyEC(prob.G, is_constraint=False)
        c3 = TileLimit(prob.G, 0, upper=7)

        prob.add_constraint(c1, c2, c3)
        self.run_detailed(prob, dict(edge=False), save='./data/opt/TestFrm_test_ovr3.png')

    def test_overlap4(self):
        tiles = parking_tiling_nd()
        prob = dumb_problem(tiles, 5, 7)

        # constraints
        c1 = NoOverlappingFaces(prob.G, is_constraint=True)
        c2 = AdjacencyEdgeJoint(prob.G, num_color=1)
        c3 = TileLimit(prob.G, 0, upper=7)

        prob.add_constraint(c1, c2, c3)
        print(c2)
        self.run_detailed(prob, dict(edge=False), save='./data/opt/TestFrm_test_ovr4.png')

    def test_overlap5(self):
        tiles = parking_tiling_2color()
        prob = dumb_problem(tiles, 5, 7)

        # constraints
        c1 = NoOverlappingFaces(prob.G, is_constraint=True)
        c2 = AdjacencyEC(prob.G, is_constraint=False)
        c3 = TileLimit(prob.G, 0, upper=7)

        prob.add_constraint(c1, c2, c3)
        self.run_detailed(prob, dict(edge=False), save='./data/opt/TestFrm_test_ovr5.png')

    def run_detailed(self, prob, obj_args, save):
        prob.solve(show=False,
                   verbose=True,
                   obj_args=obj_args,
                   const_args=const_args_formuls)
        prob.display(save=save)

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

