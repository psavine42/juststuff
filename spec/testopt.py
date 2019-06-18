import unittest
from src.cvopt.floorplanexample import *
from collections import Counter
from pprint import pprint

class BoxDumb(Box):
    class DumbyVar:
        def __init__(self, v):
            self.value = v

    def __init__(self, x, y, w, h, name=None):
        Box.__init__(self, None)
        self.x = BoxDumb.DumbyVar(x)
        self.y = BoxDumb.DumbyVar(y)
        self.width = BoxDumb.DumbyVar(w)
        self.height = BoxDumb.DumbyVar(h)
        self.name = name

    def __iter__(self):
        yield self


# -----------------------------------------
def boxes2():
    bx = [Box(200, name=0), Box(80, name=1),
            Box(80, name=3), Box(120, name=2),
            Box(120, name=4)]
    adj = [(0, 1),
           (1, 3)]
    return bx, adj


def boxes_groups():
    bx = [Group(name=0, min_area=100, aspect=2),
          Group(name=1, min_area=100, aspect=2)]
    adj = [(0, 1)]
    return bx, adj

def boxes_group2():
    bxs = [
        Box(None, name='l.0', min_dim=3, aspect=10),
        Box(None, name='l.1', min_dim=3, aspect=10),
        Box(None, name='g.0', min_dim=3, aspect=10),
        Box(None, name='g.0', min_dim=3, aspect=10),
    ]
    adj = [
        (0, 1), (2, 3)
    ]
    return bxs, adj


def house_simple():
    bxs = [
        Box(200, name='lv1', aspect=1.5),
        Box(100, name='lv2', aspect=1.5),
        Box(10, name='hall', min_dim=3),
        Box(100, name='bed1', aspect=1.5),
        Box(140, name='bed2', aspect=1.5),
        Box(80, name='bth1', aspect=2, min_dim=6),
        Box(70, name='bth2', aspect=2.5, min_dim=6),
    ]
    adj = [
        (0, 1), (1, 2), (2, 5), (2, 3), (2, 4), (4, 6)
    ]
    return bxs, adj


def hallway(n=10):
    bx = [Box(100, aspect=None, min_dim=3,
              name=0)]
    adj = []
    for i in range(1, n+1):
        bx.append(Box(300, name=i))
        adj.append((0, i))
    return bx, adj


def hall_colored(verbose=False):
    hall_tile = TemplateTile(2, 1, color=0)
    hall_tile.add_color(0, edges=[0, 3])        # setting these on exterior boundary !!!!
    hall_tile.add_color(1, edges=[1, 2, 4, 5])

    unit_tile = TemplateTile(2, 3, weight=2, color=1)

    if verbose is True:
        print(str(hall_tile))
        print('----------')
    return [hall_tile, unit_tile]


def hall_colored3(verbose=False):
    hall_tile = TemplateTile(2, 1,  color=0)
    hall_tile.add_color(0, edges=[0, 3])        # setting these on exterior boundary !!!!

    unit_tile = TemplateTile(2, 3, weight=4, color=1)
    unit_tile.add_color(1, edges=[3, 4, 8, 9])

    if verbose is True:
        print(str(hall_tile))
        print('----------')
    return [hall_tile, unit_tile]


def hall_colored4(verbose=False):
    hall_tile = TemplateTile(2, 1,  color=0)
    hall_tile.add_color(0, half_edges=[0, 3])        # setting these on exterior boundary !!!!

    unit_tile = TemplateTile(2, 3, weight=4, color=1)
    unit_tile.add_color(1, half_edges=[3, 4, 8, 9], sign=-1)

    if verbose is True:
        print(str(hall_tile))
        print('----------')
    return [hall_tile, unit_tile]


def hall_colored2(verbose=False):
    hall_tile = TemplateTile(2, 1, color=0)
    unit_tile1 = TemplateTile(2, 3, weight=2, color=1)
    unit_tile2 = TemplateTile(2, 2, weight=2, color=2)

    if verbose is True:
        print(str(hall_tile))
        print('----------')
    return [hall_tile, unit_tile1, unit_tile2]


def pf(expected, got):
    return 'expected {}, got {}'.format(expected, got)


def parking(verbose=False):
    road_tile = TemplateTile(1, 1, color=0)
    lot_tile1 = TemplateTile(2, 1, color=1, weight=5)

    road_tile.add_color(-1, half_edges=[0, 1, 2])
    road_tile.add_color(+1, half_edges=[3])

    lot_tile1.add_color(+1, half_edges=[0, 3])

    tiles = [road_tile, lot_tile1]
    return tiles

# ---------------------------------------------------
def solve_show(problem, fp):
    print('\n'.join(['{} : {}'.format(x.__name__, x(problem)) for x in
                    [Problem.is_dcp, Problem.is_dqcp, Problem.is_qp,
                        Problem.is_dgp, Problem.is_mixed_integer]]))
    print(problem)
    problem.solve()
    print(problem.solution)
    print('n')
    for p in problem.variables():
        print(p, p.value)
    for box in iter(fp):
        print(box.detail())
    fp.show()


def test_prob3(boxes, adj, save=None):
    """ with better no-overlap constraints """
    fp = FloorPlan(boxes, adj=adj)

    problem = fp.problem2()
    print(problem.is_dcp(), problem.is_dqcp(), problem.is_qp(),
          problem.is_dgp(), problem.is_mixed_integer())
    print(problem)
    problem.solve()
    print(problem.solution)
    for box in iter(fp):
        print(str(box))
    fp.show()


# --------------------------------------------------------------------
class TestSnk(unittest.TestCase):
    def test_vl1(self):
        tiles = parking()
        M = Mesh2d(g=nx.grid_2d_graph(6, 8))
        prob = AdjPlanOO(tiles, M, faces_forbidden=[])
        C = prob.own_constraints()
        bjective = prob.objective()
        p = Problem(bjective, C)
        road_tile, lot_tile = prob.placements

        # set values and check
        # ---------------------------------
        zt0 = np.zeros(road_tile.X.shape)
        zt1 = np.zeros(lot_tile.X.shape)

        zt0[15] = 1
        zt1[[1, 22]] = 1

        road_tile.X = Variable

    def test_oo5(self):
        tiles = parking()
        M = Mesh2d(g=nx.grid_2d_graph(6, 8))
        prob = AdjPlanOO(tiles, M, faces_forbidden=[])
        C = prob.own_constraints()
        face = prob._faces[15]
        edge_ix = face.edges(index=True)[0]
        hes = prob.G.faces_to_half_edges()[15][0]
        # hes_ix = prob.G.index_of_half_edge(hes)
        bjective = prob.objective()

        # --------------------------------
        edge = prob._edges[edge_ix]
        assert prob.n_half_edge_color == 1, pf(prob.n_half_edge_color, 1)
        road_tile, lot_tile = prob.placements

        colors_t0 = road_tile.template.colors(half_edge=True)
        colors_t1 = lot_tile.template.colors(half_edge=True)
        assert colors_t0 == [-1, -1, -1, 1]
        assert colors_t1 == [1, 0, 0, 1, 0, 0]
        assert len(edge._half_edges) == 2, len(edge._half_edges)
        # --------------------------------
        print(edge.X.shape)
        he1, he2 = prob._edges[edge_ix]._half_edges
        print(he1.index, he1.inv_map)
        print(he2.index, he2.inv_map)

        print('edge {}, hes {} {}'.format(edge_ix, he1.index, he2.index))
        print('HE1 map:\n', he1._map[0], '\n', he1._map[1])
        print('HE2 map:\n', he2._map[0], '\n', he2._map[1])
        # print(he2._map)
        assert len(he1._map[0]) == road_tile.X.shape[0]
        assert len(he1._map[1]) == lot_tile.X.shape[0]
        # --------------------------------------
        # adj = Face: 15, E:32, HE:{47, 48}
        # => if 1 (lot ) - prefer face[8] to have placement 0
        # => if 0 (road) - prefer face[1] to have placement 1
        # aka , if F_15 = 0, then he_map[0][15] =
        plix_lot_at1 = [i for i, x in enumerate(lot_tile.placements)
                        if x[0] == 1][0]
        print(plix_lot_at1)
        # check that mappings for half edge are correct
        assert he1.map[0][15] == -1, pf(-1, he1.map[0][15])
        assert he1.map[1][15] == 1, pf(1, he1.map[1][15])

        assert he2.map[0][8] == -1, pf(-1, he2.map[0][8])
        assert he2.map[1][1] == 1, pf(1, he2.map[1][1])

        # actions X_0,15 = 1 & X_1,1 = 1
        # half edge should have value 1

        # ---------------------------------------
        p = Problem(bjective, C)
        p.solve()
        for i in list(range(9)) + [edge_ix]:
            print('edge', i, prob._edges[i].X.value)
        print('\n------')
        pprint(prob._edges[edge_ix]._check_acks)
        # for c in prob._edges[edge_ix].constraints:
        #     print(c)

        prob.display(p, constraints=constraints, save='./data/opt/s2.png')

    def test_ggf(self):
        #
        tiles = parking()
        M = Mesh2d(g=nx.grid_2d_graph(6, 8))
        prob = AdjPlanOO(tiles, M, faces_forbidden=[])
        C = prob.own_constraints()
        face = prob._faces[15]
        edge_ix = face.edges(index=True)[0]
        hes = prob.G.faces_to_half_edges()[15][0]
        # hes_ix = prob.G.index_of_half_edge(hes)
        bjective = prob.objective()

    def constraint_map(self):
        base_types = dict(edge=True, vert=True, half_edge=True, face=True)

    def test1(self):
        s = SnakePlan([1], 10, 7)
        print(s.edge_to_edge_adj())

    def test_gg(self):
        from pprint import pprint
        coords = sort_cw([(0, 1), (1, 0), (1, 1), (0, 0)])
        cm1 = list(map(lambda x: (x[0] + 1, x[1] + 1), coords  ))
        g = nx.grid_2d_graph(3, 4)
        res, G = nx.check_planarity(g)

        # pprint(list(G.edges))

        f00_cw  = G.traverse_face((0, 0), (1, 0))
        f00_ccw = G.traverse_face((0, 0), (0, 1))
        print('coords ', coords)
        print('\nf00cw  ', f00_cw, coords == f00_cw)
        print('f00ccw ', f00_ccw, coords == f00_ccw)

        # assert f2 != coords, 'error ' + str(f2)
        #

        f11cw = G.traverse_face((1, 1), (1, 2))
        f11ccw = G.traverse_face((1, 1), (2, 1))
        f11ccw2 = G.traverse_face((1, 2), (2, 2))
        print('\ncoord   ', cm1)
        print('\nf11cw   ', f11cw, cm1 == f11cw)
        print('f11ccw  ', f11ccw, cm1 == f11ccw)
        print('f11ccw2 ', f11ccw2, cm1 == f11ccw2)

        # traversing cw leads to wierd shyt

    def test2(self):
        shapes = [rectangle(2, 2),
                  rectangle(3, 2),
                  rectangle(2, 3),
                  rectangle(1, 4),
                  rectangle(4, 1)
                  ]
        prob = MIPFloorPlan(shapes, 28, 25,
                            limits=[2, None, None, None, None])
        solution = prob.run()

    def test3(self):
        g = nx.grid_2d_graph(4, 6)
        M = Mesh2d(g)
        # print(M.faces_to_vertices())
        he = M.get_half_edge(10)
        assert he.index == 10
        (us, vs), (ue, ve) = he.u, he.v
        assert (he.u, he.v) in M

    def test4(self):
        shapes = [TemplateTile(w, h) for w, h in
                  [[3, 1],
                   [1, 3],
                   [2, 2],
                   ]]
        g = nx.grid_2d_graph(7, 7)
        M = Mesh2d(g)
        # print(M.half_edges())
        prob = AdjPlan(shapes, M)

        solution = prob.run()
        print(prob.T[0].half_edges())

    def test5(self):
        """
        """
        adjacency_colors = {
            1: 'hall_to_hall',
            2: 'hall_to_unit'
        }
        hall_tile = TemplateTile(2, 1)
        hall_tile.add_he_color([0, 3], 1)   # setting these on exterior boundary !!!!
        hall_tile.add_he_color([1, 2, 4, 5], 3)
        print(str(hall_tile))

        print('----------')
        unit_tile = TemplateTile(2, 3)
        unit_tile.add_he_color([3, 4], 3)
        unit_tile.add_he_color([8, 9], 3)
        print(str(unit_tile))

        M = Mesh2d(g=nx.grid_2d_graph(8, 8))
        prob = AdjPlan([hall_tile, unit_tile], M)
        # p#rob.own_constraints()
        print(np.sum(prob.HE_Joint1), np.sum(prob.HE_Joint2))
        # solution = prob.run()
        # print(prob.T[0].half_edges())

    def test5r(self):
        """
        """
        adjacency_colors = {
            1: 'hall_to_hall',
            2: 'hall_to_unit'
        }
        tiles = hall_colored()

        M = Mesh2d(g=nx.grid_2d_graph(8, 8))
        prob = AdjPlan(tiles, M)
        solution = prob.run()

    def test_bnd(self):
        """ test boundary half edge stats """
        t1 = TemplateTile(1, 3)
        assert len(t1.edges) == 10
        assert len(t1.boundary.edges) == 8, \
            'got {}'.format(len(t1.boundary.edges))
        assert len(t1.boundary.ext_half_edges) == 8, \
            'got {}'.format(len(t1.boundary.ext_half_edges))
        assert len(t1.half_edges()) == 20 - 8
        # print(t1.edges_to_half_edges())
        cnt = Counter([len(v) for v in t1.edges_to_half_edges().values()])
        assert dict(cnt.items()) == {1: 8, 2: 2}
        assert len(t1.interior_half_edge_index()[0]) % 2 == 0
        assert len(t1.interior_half_edge_index()[0]) == 2
        # print(t1.interior_half_edge_index())
        f2v = t1.faces_to_vertices()
        assert [len(x) for x in f2v.values()] == [4]*len(f2v)
        assert len(f2v) == 3, 'got {}'.format(f2v)

        coords = tuple(sort_cw([(0, 1), (1, 0), (1, 1), (0, 0)]))
        edges = verts_to_edges(coords)
        print('\n', coords, '\n' )
        print('\n', edges, '\n' )
        f2he = t1.faces_to_half_edges()
        f2v = t1.faces_to_vertices()
        assert f2v[0] == coords
        assert f2he[0] == edges

    def _common_stat(self, w, h):
        t1 = TemplateTile(w, h)

        # stats
        perim = 2 * (w + h)
        nverts = (w+1) * (h+1)

        # all quantities
        b_ext = t1.boundary.ext_half_edges
        b_int = t1.boundary.int_half_edges

        # boundary
        assert b_ext[0][0] == (0, 0), 'seq must start with (0,0)'
        assert b_int[0][0] == (0, 0), 'seq must start with (0,0)'
        assert b_int[-1][-1] == (0, 0), 'seq must end with (0,0)'
        assert b_ext[-1][-1] == (0, 0), 'seq must end with (0,0)'
        assert len(b_ext) == 2 * (w + h), 'got {}'.format(len(b_ext))
        assert len(b_int) == 2 * (w + h), 'got {}'.format(len(b_int))
        # assert sort_cw(b_int) == b_int

        # face_to_verts
        f2v = t1.faces_to_vertices()
        assert len(f2v) == w * h, 'got {} {}'.format(len(f2v), w * h)
        assert [len(x) for x in f2v.values()] == [4]*len(f2v)

        # face to half_edges
        f2he = t1.faces_to_half_edges()
        assert [len(x) for x in f2he.values()] == [4] * len(f2v)

        # interior index
        iheix = t1.interior_half_edge_index()
        assert len(iheix[0]) == len(iheix[1])
        # assert

        # vertices
        verts = t1.vertices()
        assert len(verts) == nverts

        # verts to face

        # half_edge to face

        #

    def test_bnd2(self):
        self._common_stat(3, 1)
        self._common_stat(4, 2)
        self._common_stat(3, 6)

    def test_oo1(self):
        tiles = hall_colored(verbose=True)
        M = Mesh2d(g=nx.grid_2d_graph(8, 8))
        prob = AdjPlanOO(tiles, M)
        constr = prob.own_constraints()
        for i, f in enumerate(prob._faces):
            assert f.index == i
        for i, plc in enumerate(prob._placements):
            print('-')
            assert len(plc.adj_vars(face=True)) == prob.num_face
            # print(plc._mutex_placements())
        obj = prob.objective()

    def test_oo2(self):
        tiles = hall_colored()
        M = Mesh2d(g=nx.grid_2d_graph(8, 8))
        tiles[0].max_uses = 2
        faces_forbidden = [3]
        prob = AdjPlanOO(tiles, M, faces_forbidden=faces_forbidden)
        solution = prob.run()

    def test_oo3(self):
        tiles = hall_colored()
        M = Mesh2d(g=nx.grid_2d_graph(7, 8))
        tiles[0].max_uses = 2
        faces_forbidden = [3]
        prob = AdjPlanOO(tiles, M, faces_forbidden=faces_forbidden)
        solution = prob.run()

    def test_oo4(self):
        tiles = hall_colored2()
        M = Mesh2d(g=nx.grid_2d_graph(7, 8))
        tiles[0].max_uses = 2
        faces_forbidden = [3]
        prob = AdjPlanOO(tiles, M, faces_forbidden=faces_forbidden)
        solution = prob.run()



class TestOpt(unittest.TestCase):
    def test1(self):
        boxes, adj = boxes2()
        test_prob3(boxes, adj)

    def test_h1(self):
        boxes, adj = hallway(n=4)
        test_prob3(boxes, adj)

    def test_h2(self):
        """
        """
        boxes, adj = house_simple()
        test_prob3(boxes, adj)

    def show_h2(self):
        boxes, adj = house_simple()
        v = [ BoxDumb(0.0, 0.0, 13.97, 14.32),
              BoxDumb(0.0, 14.32, 8.48, 11.79),
              BoxDumb(13.97, 10.91, 3.0, 3.37),
              BoxDumb(8.48, 14.32, 8.48,11.79),
              BoxDumb(13.97, 0.0, 12.86, 10.89),
                BoxDumb(16.97, 17.99, 9.86, 8.11),
            BoxDumb(16.97, 10.89, 9.86, 7.1)]
        for res, nm in zip(v, boxes):
            res.name = nm.name
        fp = FloorPlan(v)
        fp.show()

    def test_gr1(self):
        boxes, adj = boxes_groups()
        test_prob3(boxes, adj)

    def test_gr2(self):
        boxes = [
            Box(None, name='l.0', min_dim=3, aspect=10),
            Box(None, name='l.1', min_dim=3, aspect=10),
            Box(None, name='g.0', min_dim=3, aspect=10),
            Box(None, name='g.1', min_dim=3, aspect=10),
        ]
        adj = [
            (0, 1), (2, 3)
        ]
        fp = FloorPlan(boxes, adj=adj)
        C = fp.build_constraints()
        C.append(boxes[0].area + boxes[1].area >= math.sqrt(200))
        C.append(boxes[2].area + boxes[3].area >= math.sqrt(100))
        problem = Problem(Minimize(2 * (fp.height + fp.width)), C)
        solve_show(problem, fp)

    def test_gr2_mwl(self):
        boxes = [
            Box(None, name='l.0', min_dim=3, aspect=10),
            Box(None, name='l.1', min_dim=3, aspect=10),
            Box(None, name='g.0', min_dim=3, aspect=10),
            Box(None, name='g.1', min_dim=3, aspect=10),
        ]
        adj = [
            (0, 1), (2, 3)
        ]
        fp = FloorPlan(boxes, adj=adj, h=10, w=10)
        C = fp.build_constraints()
        C.append(boxes[0].area + boxes[1].area >= 9)
        C.append(boxes[2].area + boxes[3].area >= 9)
        C.append(boxes[0].area + boxes[2].area
                 + boxes[3].area + boxes[1].area >= 10.)
        obj = Minimize(sum([box.perimeter for box in fp.boxes]))
        problem = Problem(obj, C)
        solve_show(problem, fp)


    def test_gr3(self):
        boxes = [
            Box(None, name='l.0', min_dim=3, aspect=10),
            Box(None, name='l.1', min_dim=3, aspect=10),
            Box(None, name='g.0', min_dim=3, aspect=10),
            Box(None, name='g.1', min_dim=3, aspect=10),
        ]
        adj = [
            (0, 1), (2, 3)
        ]
        fp = FloorPlan(boxes, adj=adj)
        C = fp.build_constraints()
        C.append(boxes[0].area + boxes[1].area >= math.sqrt(200))
        C.append(boxes[2].area + boxes[3].area >= math.sqrt(100))
        problem = Problem(Minimize(2 * (fp.height + fp.width)), C)
        solve_show(problem, fp)

    def test_heir(self):
        pass

