import unittest
from example.cvopt.garage import *
from src.cvopt.floorplanexample import *
from src.cvopt.problem import *
from src.cvopt.mesh.mapping import MeshMapping
from pprint import pprint
from example.cvopt.famoius import *
from src.cvopt.formulate.fp_cont import *
from src.cvopt.shape.base import R2
from scipy.spatial import voronoi_plot_2d, Voronoi
from .formul_test import TestContinuous, TestDiscrete

def re(exp, got):
    return 'expected {}, got {}'.format(exp, got)


const_args_formuls = dict(face=False,
                          edge=False,
                          tile=False,
                          vertex=False,
                          half_edge=False)


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


def save_vor(self, vor, save=None):
    from scipy.spatial import voronoi_plot_2d
    fig = voronoi_plot_2d(vor)
    finalize(ax=None, save=self._save_loc(save), extents=None)


# ---------------------------
class TestPlanDisc(unittest.TestCase):
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


# --------------------------------------------------------------------------
class TestTileDisc(unittest.TestCase):
    """
    Discrete coverings with Tile Patterns
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
        prob.display(save='test_ovr2.png')

    def test_overlap3(self):
        tiles = parking_tiling_nd()
        prob = dumb_problem(tiles, 5, 7)

        # constraints
        c1 = NoOverlappingFaces(prob.G, is_constraint=True)
        c2 = AdjacencyEC(prob.G, is_constraint=False)
        c3 = TileLimit(prob.G, 0, upper=7)

        prob.add_constraint(c1, c2, c3)
        self.run_detailed(prob, dict(edge=False), save='test_ovr3.png')

    def test_overlap4(self):
        tiles = parking_tiling_nd()
        prob = dumb_problem(tiles, 5, 7)

        # constraints
        c1 = NoOverlappingFaces(prob.G, is_constraint=True)
        c2 = AdjacencyEdgeJoint(prob.G, num_color=1)
        c3 = TileLimit(prob.G, 0, upper=7)

        prob.add_constraint(c1, c2, c3)
        self.run_detailed(prob, dict(edge=False), save='test_ovr4.png')


class TestPathDisc(unittest.TestCase):
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

    def _setup_path_small(self):
        space = Mesh2d.from_grid(5, 7)
        plc = HalfEdgeSet(space)
        prob = SimplePlan([plc], space)
        return prob, space

    def _setup_path_medium(self):
        space = Mesh2d.from_grid(5, 7)
        plc = HalfEdgeSet(space)
        prob = SimplePlan([plc], space)
        return prob, space

    def test_shortest_path(self):
        prob, space = self._setup_path_small()
        c1 = ShortestPath(space, [3], [30])
        prob.add_constraints(c1)
        self.run_detailed(prob, {}, save='shortpath.png')

    def test_shortest_trees(self):
        prob, space = self._setup_path_small()
        c1 = ShortestTree(space, [3], [30, 33])
        prob.add_constraints(c1)
        assert c1.num_actions > 0
        self.run_detailed(prob, {}, save='shortestTree.png')

    def test_shortest_trees2(self):
        prob, space = self._setup_path_small()
        c2 = ShortestTree(space, [2], [25, 23])
        prob.add_constraints( c2)
        self.run_detailed(prob, {}, save='shortestTree2.png')

    def test_shortest_2trees(self):
        prob, space = self._setup_path_small()
        c1 = ShortestTree(space, [2], [25, 23])
        c2 = ShortestTree(space, [6], [33, 31])
        prob.add_constraints(c1, c2)
        self.run_detailed(prob, {}, save='shortest2Trees.png')

    def test_shortest_path_no_ovrlap(self):
        prob, space = self._setup_path_small()
        pths = [ShortestTree(space, [2], [25, 23]),
                ShortestTree(space, [6], [24, 31])]
        c1 = RouteNoEdgeOverlap(space, pths)
        prob.add_constraints(c1)
        self.run_detailed(prob, {}, save='noOverlapTrees.png')

    def test_path_no_ovrlap_verts(self):
        prob, space = self._setup_path_small()
        pths = [ShortestTree(space, [2], [25, 22]),
                ShortestTree(space, [6], [24, 30])]
        c1 = RouteNoEdgeOverlap(space, pths)
        c2 = RouteNoVertOverlap(space, pths)
        prob.add_constraints(c1, c2)
        self.run_detailed(prob, {}, save='noOverlapVertTrees.png')

    def run_detailed(self, prob, obj_args, save):
        prob.solve(show=False,
                   verbose=True,
                   obj_args=obj_args,
                   const_args=const_args_formuls)
        if save is not None:
            save = './data/opt/' + self.__class__.__name__ + '_' + save
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


class TestPathCont(TestContinuous):
    """ problems with a continuous path in R_n """
    def path1(self):
        src = [1, 1]
        tgt = [9, 5]
        w, h = 10, 10
        # points = PointList([None for _ in range(2)])
        input_list = PathFormulationR2(2, tgt, src=src)

        assert len(input_list) == 4
        assert input_list.vars[0].shape == input_list.vars[1].shape
        assert input_list.vars[0].shape == (4,), str(input_list.vars[0].shape)

        problem = FloorPlan2(input_list)
        problem.meta['w'] = w
        problem.meta['h'] = h
        formuls = [
            SegmentsHV(inputs=input_list, N=np.max([w, h])),
            ShortestPathCont(inputs=input_list, is_objective=True),
        ]
        self._run_sdp(problem, formuls, [input_list], 'dz10', True)

    def path2dest(self):
        src = [1, 1]
        tgt = [9, 5]
        tg2 = [8, 3]
        w, h = 10, 10
        points = PointList(2)
        main = PathFormulationR2(points, tgt, src=src)
        branch = PathFormulationR2(points, tg2, src=None)

        problem = FloorPlan2(main)
        problem.meta['w'] = w
        problem.meta['h'] = h
        formuls = [
            SegmentsHV(inputs=main, N=np.max([w, h])),
            ShortestPathCont(inputs=main, is_objective=True),
            SegmentsHV(inputs=branch, N=np.max([w, h])),
            ShortestPathCont(inputs=branch, is_objective=True),
        ]
        self._run_sdp(problem, formuls, [main, branch], '2pth', True)

    def path2dest_obst(self):
        src = [1, 1]
        tgt = [9, 5]
        tg2 = [8, 3]
        w, h = 10, 10
        points = PointList(2)
        main = PathFormulationR2(points, tgt, src=src)

        problem = FloorPlan2(main)
        problem.meta['w'] = w
        problem.meta['h'] = h
        obstacle = UnusableZoneSeg(main, 5, 3, 7, 2)
        formuls = [
            obstacle,
            SegmentsHV(inputs=main, N=np.max([w, h])),
            ShortestPathCont(inputs=main, is_objective=True),
        ]
        self._run_sdp(problem, formuls,  [main, obstacle], '2pth_obs', True)

        vxr, vxl, vyu, vyb = obstacle.bins
        iseg = np.asarray([[i - 1, i] for i in range(1, len(obstacle.inputs))])
        print(iseg)

        ip1, ip2 = iseg[:, 0], iseg[:, 1]
        s = cvx.vstack(obstacle.bins)
        print(s.value.astype(int).T)
        print()
        s = cvx.vstack([
            vxr[ip1] + vxr[ip2],
            vxl[ip1] + vxl[ip2],
            vyu[ip1] + vyu[ip2],
            vyb[ip1] + vyb[ip2],
        ])
        print(s.value.astype(int))
        print(cvx.max(s.T, axis=1).value.astype(int))

    def path2dest_obst_obj(self):
        src = [1, 1]
        tgt = [9, 5]
        tg2 = [8, 3]
        w, h = 10, 10
        points = PointList(2)
        main = PathFormulationR2(points, tgt, src=src)

        problem = FloorPlan2(main)
        problem.meta['w'] = w
        problem.meta['h'] = h
        obstacle = UnusableZoneSeg(main, 5, 3, 7, 2)
        formuls = [
            obstacle,
            SegmentsHV(inputs=main, N=np.max([w, h])),
            ShortestPathCont(inputs=main, is_objective=True),
        ]
        self._run_sdp(problem, formuls,  [main, obstacle], '2pth_obs', True)

        vxr, vxl, vyu, vyb = obstacle.bins
        iseg = np.asarray([[i - 1, i] for i in range(1, len(obstacle.inputs))])
        print(iseg)

        ip1, ip2 = iseg[:, 0], iseg[:, 1]
        s = cvx.vstack(obstacle.bins)
        print(s.value.astype(int).T)
        print()
        s = cvx.vstack([
            vxr[ip1] + vxr[ip2],
            vxl[ip1] + vxl[ip2],
            vyu[ip1] + vyu[ip2],
            vyb[ip1] + vyb[ip2],
        ])
        print(s.value.astype(int))
        print(cvx.max(s.T, axis=1).value.astype(int))




