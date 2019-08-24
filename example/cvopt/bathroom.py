import numpy as np
"""

Todo change these unittests to something better looking
"""


def ft_in(ft=0, inch=0):
    return (ft * 12 + inch) / 12


""" gathered from googlings """
_sizes = {
    'toilet':   {
        'inner': [ft_in(inch=30), ft_in(inch=18)],
        'outer': [ft_in(inch=50), ft_in(inch=24)],
        'same_axis': 1,
    },
    'tub':      {'inner': [2.5, 4.9], 'outer': [2.5, 4.9]},
    'shower':   {'inner': [2.5, 2.5], 'outer': [2.6, 2.6]},
    'sink':     {
        'inner': [ft_in(inch=24), ft_in(inch=18)],
        'outer': [ft_in(inch=40), ft_in(inch=24)],
        'same_axis': 0,
        'notes': ''
    },
    'outlet'  : {'inner': [1, 2, 8]},
    'ada_turn': {'radius': ft_in(inch=30)},
    'door24': {
        'inner': [ft_in(inch=2), ft_in(inch=24)],
        'outer': [ft_in(inch=24), ft_in(inch=24)],
        'notes': '24'
    },
    'stall': {
        'inner': [ft_in(inch=60), ft_in(inch=30)],
        'outer': [ft_in(inch=81), ft_in(inch=30)],
        'same_axis': 1
    },
    'ada_stall': {

    }
}


def adj_to_list(data):
    N = len(data['objects'])
    dist_weights = np.ones((N, N))

    for k, v in data.get('inv_adj', {}).items():
        ki, kj = k
        dist_weights[ki, kj] = v
    return dist_weights[np.triu_indices(N, 1)].tolist()


def bathroom_problem_3f_01():
    """
    """
    return dict(
        boundary={'box': [5, 8 + 2 / 12]},
        objects=dict(
            toilet=_sizes['toilet'],
            tub=_sizes['tub'],
            sink=_sizes['sink'],
            door=_sizes['door24'],
        ),
        inv_adj={(0, 1): 3},  # repulsive force
        distances=[],
        windows={'segment': {}})


def bathroom_problem_2f_01():
    """
    """
    return dict(
        boundary={'box': [4.25, 5]},
        objects=dict(
            toilet=_sizes['toilet'],
            sink=_sizes['sink'],
            door=_sizes['door24'],
        ),
        inv_adj={},
        distances=[],
        windows={'segment': {}})


def bathroom_problem_pub_3s():
    """
    """
    return dict(
        boundary={'box': [4.25, 5]},
        objects=dict(
            st1=_sizes['stall'],
            st2=_sizes['stall'],
            st3=_sizes['stall'],
            sink1=_sizes['sink'],
            sink2=_sizes['sink'],
            door=_sizes['door24'],
        ),
        inv_adj={},
        distances=[],
        windows={'segment': {}})


def template_problem():
    outer_t, inner_t = [], []

    ws, hs = [], []
    ows, ohs = [], []
    dom_ax = []
    for k, fixture in data['objects'].items():
        if 'side' in fixture:
            pass
        if 'radius' in fixture:
            pass
        else:
            w, h = fixture['inner']
            ws.append(w)
            hs.append(h)
            inner_t.append(BTile(area=w * h, name=k))
        dom_ax.append(1 - fixture.get('same_axis', 1))
        small_size = fixture.get('outer', None)
        if small_size is not None:
            sw, sh = small_size[0], small_size[1]
            ows.append(sw)
            ohs.append(sh)
            outer_t.append(BTile(area=sw * sh, name='bbx.' + k))
