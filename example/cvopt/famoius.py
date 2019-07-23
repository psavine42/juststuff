import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

"""
30-department Nugent et al. (1968) QAP problem.

armour and buffa 20-department
"""
np.random.seed(60)

def generic(n=3):
    """
    return
        - C_ij matrix
        - list of target areas
        - space limits
    """
    Cij = np.random.random((n, n))
    Cij[np.diag_indices(n)] = 0
    areas = np.random.randint(5, 10, n)
    return Cij, areas

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def from_voronoixx(n=3):
    """ fuuuck """
    _, areas = generic(n=n)
    points = np.random.rand(n, 2)
    vor = Voronoi(points)
    Cij = np.zeros((n, n), dtype=int)
    # Cij[np.triu_indices(n, 1)] = 1

    rpm = np.zeros((n, n, 2), dtype=int)
    # rpm[np.triu_indices(n, 1)] = 2
    print(vor.regions)
    print(vor.vertices)
    print('---------')
    nr, nv = voronoi_finite_polygons_2d(vor)
    print(nr)
    print(nv)
    print('---------')
    print(vor.ridge_dict.keys())
    print(points)
    print(vor.points)
    print(vor.min_bound)
    # construct cij matrix from vor-adjacencvy
    # x = [p[0] for p in points]
    # y = [p[1] for p in points]

    for r, (i, j) in enumerate(vor.ridge_points):
        # ridge index, adj
        i, j = sorted([i, j])
        reg_i = vor.ridge_vertices[vor.point_region[i]]
        reg_j = vor.ridge_vertices[vor.point_region[j]]
        reg_i_pt = vor.vertices[reg_i].mean(0)
        reg_j_pt = vor.vertices[reg_j].mean(0)
        # cent_i = [np.mean(reg_i_pt[:, 1]), np.sum(reg_i_pt[:, 1]) / len(reg_i_pt))

        dx, dy = np.abs(points[i] - points[j])
        # print(points[i], points[j], ' -> ', reg_i_pt, reg_j_pt, )
        if dx >= dy:
            if points[i, 0] >= points[j, 0]: # i to left of j
                rpm[i, j, 0] = 1
            else: # i to right of j
                rpm[i, j, 0] = 2

        else:
            if points[i, 1] >= points[j, 1]: # i to left of j
                rpm[i, j, 1] = 1
            else:               # i to right of j
                rpm[i, j, 1] = 2
    return Cij, areas, points, rpm


def from_voronoi(n=3, verbose=False):
    _, areas = generic(n=n)
    points = np.random.rand(n, 2)
    vor = Voronoi(points)

    # hack to move voronois about a bit
    import pytess
    vor2 = pytess.voronoi(points.tolist())
    vor2 = {tuple(x): np.asarray(y) for x, y in vor2 if x is not None}
    rpm = np.zeros((n, n, 2), dtype=int)
    # construct cij matrix from vor-adjacencvy
    if verbose:
        print(vor.ridge_points)

    # keep track of bounds
    bnd_lr_ud = np.zeros((n, 2), dtype=int)

    for r, (i, j) in enumerate(vor.ridge_points):
        # ridge index, adj
        i, j = sorted([i, j])
        reg_i_pt = vor2[tuple(points[i].tolist())].mean(0)
        reg_j_pt = vor2[tuple(points[j].tolist())].mean(0)

        dx, dy = np.abs(points[i] - points[j])
        # print(points[i], points[j], ' -> ', reg_i_pt, reg_j_pt, )
        if dx >= dy:
            # Left/right
            if points[i, 0] >= points[j, 0]: # i to right of j
                rpm[i, j, 0] = 2
            else: # i
                rpm[i, j, 0] = 1
            bnd_lr_ud[[i, j], 0] = 1
        else:
            # Up/down
            if points[i, 1] >= points[j, 1]:
                rpm[i, j, 1] = 1
            else:
                rpm[i, j, 1] = 2
            bnd_lr_ud[[i, j], 1] = 1

    for i, (bnd_lr, bnd_ud) in enumerate(bnd_lr_ud.tolist()):
        if bnd_lr == 0:
            pass
        if bnd_ud == 0:
            pass

    return None, areas, points, rpm, vor


def from_rand(n=3, verbose=False):
    from src.cvopt.formulate.form_utils import triu_indices
    _, areas = generic(n=n)
    points = np.random.rand(n, 2)
    rpm = np.zeros((n, n, 2), dtype=int)
    for r, (i, j) in enumerate(triu_indices(n)):
        i, j = sorted([i, j])
        dx, dy = np.abs(points[i] - points[j])
        if dx >= dy:
            # Left/right
            if points[i, 0] >= points[j, 0]:
                rpm[i, j, 0] = 2
            else: # i
                rpm[i, j, 0] = 1
        else:
            # Up/down
            if points[i, 1] >= points[j, 1]:
                rpm[i, j, 1] = 1
            else:
                rpm[i, j, 1] = 2
    return None, areas, points, rpm, Voronoi(points)
