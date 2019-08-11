
"""

Todo change these unittests to something better looking
"""


def _turlet_problem4f():
    """
    """
    return dict(
        boundary={'box': [5, 8 + 2 / 12]},
        objects=dict(
            turlet={'box': [4, 34 / 12]},
            tub={'box': [2.5, 4.9]},
            sink={'box': [2, 3]},
            door={'box': [2, 2]},
        ),
        doors=[{'r': 2.5, 'x': None, 'y': None}],
        inv_adj={(0, 1): 3},  # repulsive force
        distances=[],
        windows={'segment': {}})


def _turlet_problem_3c():
    """

    """
    return dict(
        boundary={'box': [5, 8 + 2 / 12]},
        objects=dict(
            turlet={'box': [4, 34 / 12]},
            tub={'box': [2.5, 4.9]},
            sink={'box': [2, 3]},
            door={'box': [2, 2]},
        ),
        doors=[{'r': 2.5, 'x': None, 'y': None}],
        inv_adj={(0, 1): 3},  # repulsive force
        distances=[],
        windows={'segment': {}})


def _solvem2(self, data, save=None):
    """ """
    # path of width h to all things

    # nothing infront of doors
    # if '' in data['boundary']
    if data['boundary'] == 'classical':
        bw, bh = Variable(name='W'), Variable(name='H')
    else:
        bw, bh = data['boundary']['box']
    centr = np.asarray([bw / 2, bh / 2])
    tiles = []
    ws, hs = [], []
    for k, fixture in data['objects'].items():
        w, h = fixture['box']
        ws.append(w)
        hs.append(h)
        tiles.append(BTile(area=w * h, name=k))
    ws = np.asarray(ws)
    hs = np.asarray(hs)

    # list of repulsive forces to maximize distances with
    dist_weights = np.ones((len(tiles), len(tiles)))
    for k, v in data.get('inv_adj', {}).items():
        ki, kj = k
        dist_weights[ki, kj] = v
    dist_weights = dist_weights[np.triu_indices(len(tiles), 1)].tolist()

    # setup of problem objects
    # -------------------------------------------
    # inputs
    box_list = BoxInputList(tiles)
    # if it is a classical problem, solve for min boundary
    pld = PointDistObj(box_list,
                       weight=dist_weights,
                       obj=Maximize)
    fixdim = FixedDimension(box_list, values=[ws, hs],
                            # , indices=[0, 1, 2]
                            )
    # formulations for problem constraints
    frm = [BoundsXYWH(box_list, w=bw, h=bh),
           fixdim,
           pld,
           # FeasibleSet(),
           NoOvelapMIP(box_list)
           ]
    # Add Problem Specific Logic
    # -------------------------------------------
    # 1) There is a door on the left side of the thing
    # no box can overlap the Door swing
    # model the swing as a square
    door = ConstrLambda([box_list.X[3] == 1])
    frm.append(door)

    # 2) each fixture is restricted to a wall
    # tl == BoundsL or tr == BoundsR ....
    choice = OneEdgeMustTouchBoundary(box_list, [0, bh, 0, bw])
    frm.append(choice)

    # 3) Short side restrictions
    # todo - is it worth modeling these differently?
    # However, also, the toilets short side must be on wall
    # and the sinks long-side must be on wall
    # when fdim[0, i] = 1, then Xi.W > Xi.H
    # therefore choice of wall is restricted to xmin, xmax
    # choice[2, i] + choice[3, i] == 1
    #
    # w_short = fixdim.indicators[0]
    # w_long = fixdim.indicators[1]
    # inds = choice.indicators

    # 4) todo There is a bounding box within which turlet exists
    # which can only overlap the door

    # 5) Toilet is not facing Tub (common architecture crit thing)
    # modeled as same orientation
    frm.append(OrientationConstr(box_list, [0, 1]))

    # -----------------------------
    # combine into a problem stage
    stage1 = Stage(box_list, forms=frm)

    p = stage1.make(verbose=True)
    print(describe_problem(p))
    # print(p.objective.expr.curvature)
    p.solve(verbose=True)
    assert stage1.is_solved is True
    save_pth = self._save_loc(save=save)
    stage1.display(save=save_pth, extents=[-1, 6, -1, 10])
    print(box_list.describe())
    print(p.value)
    print('edge_choice\n', choice.indicators.value)
    print('dim_choice\n', fixdim.indicators.value)
    for vr in pld.uv:
        print(vr.name(), vr.value)
    for t in tiles:
        print(t.area)

