import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from src.cvopt.mesh import Mesh2d
import src.geom.r2 as r2
import cycler

_colors = ['#68b382', '#67c2ba', 'green', 'red', 'violet', 'orange']


def place_tile2(tile, point):
    result = []
    px, py = point
    for i, _ in enumerate(tile.template.faces_to_vertices()):
        bottom_left = tile.template.get_face(i).bottom_left
        result.append((px + bottom_left[0], py + bottom_left[1]))
    return result


def display_face_bottom_left(problem, save=None, edges=False, **kwargs):
    matplotlib.rc('font', size=6)
    colors = ['#68b382', '#67c2ba', 'green', 'red', 'violet', 'orange']
    info = dict(edgecolor='#acacad', linewidth=1)
    fig, ax = plt.subplots(1)

    imesh = problem.G
    n, m = imesh.vertices[-1]

    print('Placement-------------------------')
    for placement in problem._placements:
        print('', placement.solution)

    print('X-------------------------')
    for placement, mapping in problem.solution:
        for face_ix in mapping.faces:
            x, y = imesh.faces.geom[face_ix][0]
            ax.add_patch(
                Rectangle((x, y), 1, 1, facecolor=colors[placement.index],
                          **info)
            )
        ax = draw_half_edges(mapping, ax, offset=True, bnd_only=True)

    # label faces
    for face in problem._faces:
        x, y = face.bottom_left
        ax.text(x + 0.5, y + 0.5, "{}".format(face.index))

    # draw
    if edges:
        for i, he in imesh.edges.to_half_edges.items():
            (x1, y1), (x2, y2) = imesh.edges[i]
            heis = ','.join([str(imesh.index_of_half_edge(x)) for x in he])
            x = x1 + (x2 - x1) / 3
            y = y1 + (y2 - y1) / 3
            ax.text(x, y, "{}:{}".format(i, heis))
    finalize(ax, save=save, extents=[n, m])


def finalize(ax=None, save=None, extents=None):
    if ax is not None:
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
    plt.gcf()
    if isinstance(save, str):
        plt.savefig(save)
        plt.clf()
        plt.close()
    else:
        plt.show()


def _get_geom(imesh, fn):
    if isinstance(imesh, Mesh2d):
        return fn.fget(imesh).geom
    elif isinstance(imesh, (list, tuple)):
        return imesh
    else:
        raise Exception('')


def _parse_drawable(x):
    if isinstance(x, int):
        return x, {}
    elif isinstance(x, dict):
        i = x.pop('index')
        data = x
        return i, data
    elif isinstance(x, tuple):
        i, data = x
        return i, data
    else:
        raise Exception('unkown display object')


#  Items-----------------------------------------------------
def draw_edge(ax, geom, index=None, **kwargs):
    (x1, y1), (x2, y2) = list(geom)
    ax.plot([x1, x2], [y1, y2], **kwargs)
    if index:
        x = x1 + (x2 - x1) / 3
        y = y1 + (y2 - y1) / 3
        ax.text(x, y, "e.{}".format(index))
    return ax


def draw_vertex(geom, ax, index=None, marker='o', **kwargs):
    x, y = geom
    ax.plot([x], [y], marker=marker, **kwargs)
    if index:
        ax.text(x, y, "v.{}".format(index))
    return ax


def draw_edges(imesh:Mesh2d, ax, label=False, **kwargs):
    info = dict(color='black', linewidth=1)
    opts = {**info, **kwargs}
    edges = _get_geom(imesh, Mesh2d.edges)

    for i, geom in enumerate(edges):
        ax = draw_edge(ax, geom, index=i if label else None, **opts)
    return ax


def draw_box(d, ax, label=True, edgecolor='black', **kwargs):
    i, data = _parse_drawable(d)
    x, y = data.pop('x'), data.pop('y')
    w = data.pop('w') if 'w' in data else 1
    h = data.pop('h') if 'h' in data else 1

    ax.add_patch(Rectangle((x, y), w, h, edgecolor=edgecolor,
                           **kwargs))
    if label is True:
        name = data.get('name', None)
        txt = name if name is not None else str(i)
        ax.text(x + w / 2, y + h / 2, "box.{}".format(txt))
    return ax


# formulation -----------------------------------------------------
def draw_formulation_he(form, ax):
    he = form.space.half_edges.geom
    for i in form.display()['half_edges']:
        ax = draw_edge(ax, he[i], index=i, color='black')
    return ax


def draw_formulation_discrete(form, ax, **kwargs):
    vg = form.space.vertices.geom
    for x in form.display()['vertices']:
        i, data = _parse_drawable(x)
        draw_vertex(vg[i], ax, index=i, **data)
    ax = draw_formulation_he(form, ax)
    return ax


def draw_formulation_cont(form, ax, **kwargs):
    disp_dict = form.display()
    colors = cm.viridis(np.linspace(0, 1, len(disp_dict['boxes'])))
    for i, d in enumerate(disp_dict['boxes']):
        draw_box(d, ax, facecolor=colors[i], **kwargs)
    # ax = draw_formulation_he(form, ax)
    return ax


# --------------------------------------------------
def draw_half_edges(imesh, ax, bnd_only=False, label=False, offset=False, **kwargs):
    neg_colors = ['#ff0000']
    pos_colors = ['#ff00f7']
    info = dict(length_includes_head=True, linewidth=1)
    opts = {**info, **kwargs}
    from src.cvopt.mesh import MeshMapping
    from src.cvopt.placements import ActionBase
    if isinstance(imesh, MeshMapping):
        mapping = imesh
        he_cols = mapping.match_col(he=True)
        imesh = mapping.transformed
    elif isinstance(imesh, ActionBase):
        return
    else:
        mapping = None
        he_cols = None
    hes = imesh.half_edges.to_vertex_geom
    bnd = imesh.boundary.ext_half_edges
    for i, pt in hes.items():
        if bnd_only is True and pt not in bnd:
            continue
        (x1, y1), (x2, y2) = list(pt)
        if offset is True:
            # offset in direction of face center
            face_ix = imesh.half_edges.to_faces[i]
            face = imesh.get_face(face_ix)
            center = face.np.mean(axis=0)
            edge_mid = np.asarray([x2 + x1, y2 + y1]) / 2
            dir = (center - edge_mid) * 0.1
            x1, x2 = x1 + dir[0], x2 + dir[0]
            y1, y2 = y1 + dir[1], y2 + dir[1]

        if mapping:
            i = mapping.half_edge_map()[i]
        if he_cols and i in he_cols:
            col_index = he_cols[i]
            if col_index < 0:
                color = neg_colors[-1*col_index -1]
            else:
                color = pos_colors[col_index -1]
        else:
            color = 'black'
        ax.arrow(x1, y1, x2 - x1, y2 - y1, color=color, **opts)
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

