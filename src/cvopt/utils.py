import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np


def place_tile2(tile, point):
    result = []
    px, py = point
    for i, _ in enumerate(tile.template.faces_to_vertices()):
        bottom_left = tile.template.get_face(i).bottom_left
        result.append((px + bottom_left[0], py + bottom_left[1]))
    return result


def display_face_bottom_left(problem, save=None, **kwargs):
    colors = ['blue', 'yellow', 'green', 'red', 'violet', 'orange']
    info = dict(edgecolor='black', linewidth=1)
    fig, ax = plt.subplots(1)
    n, m = sorted(problem.G.vertices())[-1]

    print('Placement-------------------------')
    for placement in problem._placements:
        print('', placement.solution)

    print('X-------------------------')
    for tile, xy in problem.solution:
        for x, y in place_tile2(tile, xy):
            ax.add_patch(
                Rectangle((x, y), 1, 1, facecolor=colors[tile.index], **info)
            )

    for tile in problem._faces:
        x, y = tile.bottom_left
        ax.text(x + 0.5, y + 0.5, "{}".format(tile.index))

    for i, he in problem.G.edges_to_half_edges().items():
        (x1, y1), (x2, y2) = problem.G.edges[i]
        heis = ','.join([str(problem.G.index_of_half_edge(x)) for x in he])
        x = x1 + (x2 - x1) / 3
        y = y1 + (y2 - y1) / 3
        ax.text(x, y, "{}:{}".format(i, heis))

    ax.axis([0, n, 0, m])
    ax.axis('off')
    ax.set_aspect('equal')
    if isinstance(save, str):
        plt.savefig(save)
        plt.clf()
        plt.close()
    else:
        plt.show()


def display_edges(edges):
    for i, he in problem.G.edges_to_half_edges().items():
        (x1, y1), (x2, y2) = problem.G.edges[i]
        heis = ','.join([str(problem.G.index_of_half_edge(x)) for x in he])
        x = x1 + (x2 - x1) / 3
        y = y1 + (y2 - y1) / 3
        ax.text(x, y, "{}:{}".format(i, heis))
    return


# def deep_tup(x):
#     """fully copies trees of tuples or lists to a tree of lists.
#          deep_list( (1,2,(3,4)) ) returns [1,2,[3,4]]
#          deep_list( (1,2,[3,(4,5)]) ) returns [1,2,[3,[4,5]]]"""
#     if not isinstance(x, (tuple, list)):
#         return x
#     return tuple(list(map(deep_tup, x)))


# def translate(point_list, plus):
#     """
#     verts d:2 list[tuple(int, int) ]
#     edges d:3 list[tuple(tuple(int, int), tuple(int, int)) ]
#     faces d:3 list[tuple(tuple(int, int), tuple(int, int), tuple(int, int) ... ) ]
#     """
#     res = []
#     for el in (np.asarray(point_list) + np.asarray(plus)).tolist():
#         res.append(deep_tup(el))
#     return res
