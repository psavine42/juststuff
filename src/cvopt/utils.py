import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from src.cvopt.mesh import Mesh2d
import src.geom.r2 as r2


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
    n, m = problem.G.vertices[-1]

    print('Placement-------------------------')
    for placement in problem._placements:
        print('', placement.solution)

    print('X-------------------------')
    for placement, mapping in problem.solution:
        for face_ix in mapping.faces:
            x, y = problem.G.faces.geom[face_ix][0]
            ax.add_patch(
                Rectangle((x, y), 1, 1, facecolor=colors[placement.index], **info)
            )

    # label faces
    for tile in problem._faces:
        x, y = tile.bottom_left
        ax.text(x + 0.5, y + 0.5, "{}".format(tile.index))

    # draw
    for i, he in problem.G.edges.to_half_edges.items():
        (x1, y1), (x2, y2) = problem.G.edges[i]
        heis = ','.join([str(problem.G.index_of_half_edge(x)) for x in he])
        x = x1 + (x2 - x1) / 3
        y = y1 + (y2 - y1) / 3
        ax.text(x, y, "{}:{}".format(i, heis))
    print(n, m)
    finalize(ax, save=save, extents=[n, m])
    if isinstance(save, str):
        plt.savefig(save)
        plt.clf()
        plt.close()
    else:
        plt.show()


def finalize(ax, save=None, extents=None):
    if extents is not None:
        if len(extents) == 2:
            n, m = extents
            ax.axis([0, n, 0, m])
        elif len(extents) == 1:
            n = extents[0]
            ax.axis([0, n, 0, n])
        elif len(extents) == 4:
            ax.axis(extents)
    ax.axis('off')
    ax.set_aspect('equal')


def _get_geom(imesh, fn):
    if isinstance(imesh, Mesh2d):
        return fn.fget(imesh).geom
    elif isinstance(imesh, (list, tuple)):
        return imesh
    else:
        raise Exception('')


def draw_edges(imesh:Mesh2d, ax, label=False, **kwargs):
    info = dict(color='black', linewidth=1)
    opts = {**info, **kwargs}
    pts = _get_geom(imesh, Mesh2d.edges)

    for i, pt in enumerate(pts):
        (x1, y1), (x2, y2) = list(pt)
        ax.plot([x1, x2], [y1, y2], **opts)
        if label is True:
            x = x1 + (x2 - x1) / 3
            y = y1 + (y2 - y1) / 3
            ax.text(x, y, "e.{}".format(i))
    return ax


def draw_half_edges(imesh: Mesh2d, ax, label=False, **kwargs):
    info = dict(color='black', length_includes_head=True, linewidth=1)
    opts = {**info, **kwargs}
    pts = _get_geom(imesh, Mesh2d.half_edges)

    for i, pt in enumerate(pts):
        (x1, y1), (x2, y2) = list(pt)
        ax.arrow(x1, y1, x2 - x1, y2 - y1, **opts)
        if label is True:
            x = x1 + (x2 - x1) / 3
            y = y1 + (y2 - y1) / 3
            ax.text(x, y, "e.{}".format(i))
    return ax


def label_edges(imesh, ax):
    """ """
    for tile in imesh._faces:
        x, y = tile.bottom_left
        ax.text(x + 0.5, y + 0.5, "{}".format(tile.index))
    return ax


def label_half_edges(imesh: Mesh2d, ax):
    """ todo  """
    for tile in imesh.half_edges:
        x, y = tile.bottom_left
        ax.text(x + 0.5, y + 0.5, "{}".format(tile.index))
    return ax


def label_faces(imesh: Mesh2d, ax, labels=None):
    """ write the face index at the centroid"""
    geom = _get_geom(imesh, Mesh2d.faces)
    if labels and not len(geom) == len(labels):
        raise Exception('llabel_len must == len(geoms)')

    for face_ix, verts in imesh.faces.to_vertices.items():
        x, y = r2.centroid(verts)
        ax.text(x, y, "f:{}".format(face_ix))
    return ax

