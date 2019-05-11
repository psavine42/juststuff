import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import math
import networkx as nx
import pylab
from matplotlib.collections import LineCollection
import PIL
import visdom


def get_keys(data, keys):
    st = ''
    if 'all' in keys:
        keys = data.keys()
    for k in keys:
        st += '\n {}: {} '.format(k, data.get(k, ''))
    return st


def plotpoly(layouts, titles=None, show=True, figsize=(12, 12)):
    import src.layout
    if not isinstance(layouts, list):
        layouts = [layouts]
    use_tit = True if titles and len(titles) == len(layouts) else False
    n_row = int(math.ceil(len(layouts) ** 0.5))
    fig = plt.figure(figsize=figsize)
    for fignum, poly in enumerate(layouts):
        ax = fig.add_subplot(n_row, n_row, fignum+1)
        for polygon in poly.geoms:
            x, y = polygon.exterior.coords.xy
            points = np.array([x, y], np.int32).T
            shape = patches.Polygon(points, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(shape)

        if isinstance(poly, src.layout.BuildingLayout):
            fp = poly.problem.footprint
            if fp is not None:
                x, y = fp.exterior.coords.xy
                points = np.array([x, y], np.int32).T
                shape = patches.Polygon(points, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(shape)
        if use_tit:
            ax.set_title(titles[fignum])
        ax.relim()

        ax.autoscale_view()
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.axis('off')
    if show is True:
        plt.show()


def make_image(sequence, epoch, name='_output_'):
    """plot drawing with separated strokes"""
    strokes = np.split(sequence, np.where(sequence[:, 2] > 0)[0] + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                    canvas.tostring_rgb())
    name = str(epoch) + name + '.jpg'
    pil_image.save(name, "JPEG")
    plt.close("all")


def plot_constraints(problem):
    pass


def layout_disc_to_viz(layout, viz=None):
    if viz is None:
        viz = visdom.Visdom()
    nx.draw(layout._G, pos={x: x for x in layout._G.nodes()}, node_color='r', node_size=2)
    nx.draw(layout._T, pos={x: x for x in layout._T.nodes()}, node_color='b', node_size=3)
    viz.matplot(plt.gcf())
    plt.clf()


def plot_figs(Gs, num_trial, horiz=False):
    # num_expirements = len(Gs) // num_trial
    # n_block = int(math.ceil(num_expirements ** 0.5))
    # blocksize = int(math.ceil(num_trial ** 0.5))

    n_row = len(Gs) // num_trial
    n_col = num_trial
    size = 4

    if horiz is True:
        sx, sy = size * n_row/n_col, size
        n_row, n_col = n_col, n_row
    else:
        sx, sy = size, size * n_row / n_col
    fig = plt.figure(figsize=(sx, sy))
    for fignum, G in enumerate(Gs):

        # block_num = fignum // num_trial
        # block_idx = fignum % num_trial

        # block_start = block_num

        ax = fig.add_subplot(n_row, n_col, fignum + 1)
        pos = {x: x for x in G.nodes()}
        nds = np.asarray([x for x in G.nodes()])

        # draw_nodes, nxd.draw(G, pos, ax=ax) does something to multuplot
        ax.scatter(nds[:, 0], nds[:, 1], s=1)

        edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in list(G.edges())])
        line_segments = LineCollection(edge_pos,
                                       linewidths=1,
                                       linestyle='solid')
        ax.add_collection(line_segments)
        if fignum % num_trial == 1:
            ax.set_title(G.name, size=6, pad=0)

        # reset limits ax.relim()
        ax.set_xlim(0, 22)
        ax.set_ylim(0, 15)
        # ax.autoscale_view()

        # hide labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)
    plt.show()


def show_trees(trees):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    for root in trees:
        nds = np.asarray(list(root.points()))
        ax.scatter(nds[:, 0], nds[:, 1], s=2, c='r')
        edges = np.asarray(list(root.edges()))
        edge_pos = np.asarray([(pos[0], pos[1]) for pos in edges])
        line_segments = LineCollection(edge_pos,
                                       linewidths=1,
                                       linestyle='solid')
        ax.add_collection(line_segments)
        bnd_pos = []
        boundary = list(root.get_boundary())
        for i in range(len(boundary)):
            bnd_pos.append([boundary[i-1], boundary[i]])
        bnd_segs = LineCollection(np.asarray(bnd_pos),
                                  linewidths=1,
                                  linestyle='solid')
        ax.add_collection(bnd_segs)

    plt.show()


def simple_plot(G, kys=[], meta={}, label=True, layout=False, save=True):
    if layout is True:
        pos = nx.spring_layout(G)
    else:
        pos = {x:x for x in G.nodes()}

    colors, labels, sizes = [], {}, []
    for (p, d) in G.nodes(data=True):
        labels[p] = get_keys(d, kys)
        n_type = d.get('type', None)
        colors.append(meta.get(n_type, {}).get('color', 0.45))
        sizes.append(meta.get(n_type, {}).get('size', 20))
    nx.draw(G, pos,
            labels=labels,
            with_labels=label,
            arrowsize=20,
            node_size=sizes,
            node_color=colors,
            edge_cmap=plt.cm.Blues,
            font_size=10)
    if isinstance(save, str):
        pylab.savefig(save)
        pylab.clf()
        pylab.close()
    else:
        pylab.show()


def cells_to_nx(res):
    """ visualize propagator network with nx """
    q = []
    for r in res:
        q += r.neighbors
    G = nx.DiGraph()
    seen = set()
    while q:
        el = q.pop(0)
        if el.id not in seen:
            seen.add(el.id)
            G.add_node(el.id, type='prop')
            out = el.output
            G.add_node(out.id, type='cell', content=str(out.contents), var=out._var)
            G.add_edge(el.id, out.id, weight=el._cnt)
            q.extend(out.neighbors)
            for n in el.inputs:
                if n.id not in seen:
                    G.add_node(n.id, type='cell', content=str(n.contents), var=n._var)
                    q.extend(n.neighbors)
                    G.add_edge(n.id, el.id, weight=el._cnt)
    return G


def prop_plot(G,  meta={}, label=True, pos=None):
    posx = pos if pos is not None else nx.spring_layout(G)
    # print(pos)
    colors, labels, sizes = [], {}, []
    for (p, d) in G.nodes(data=True):
        n_type = d.get('type', None)

        if n_type == 'cell' :
            if d.get('var', None) is not None and 'IN_' in d.get('var', ''):
                n_type = 'cell+input'

            elif d.get('var', '') == 'res':
                n_type = 'cell+res'

            elif d.get('content', None) is not None:
                n_type = 'cell+content'

        tkys = meta.get(n_type, {})['keys']
        labels[p] = get_keys(p, d, tkys)
        colors.append(meta.get(n_type, {}).get('color', 0.45))
        sizes.append(meta.get(n_type, {}).get('size', 20) )

    nx.draw(G, posx,
            labels=labels,
            with_labels=label,
            arrowsize=20,
            node_size=sizes,
            node_color=colors,
            edge_cmap=plt.cm.Blues,
            font_size=11)
    pylab.show()
