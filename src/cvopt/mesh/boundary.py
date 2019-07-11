import numpy as np
import scipy.spatial
from cvxpy.utilities.performance_utils import compute_once, lazyprop
import src.geom.r2 as r2


class Boundary(object):
    def __init__(self, parent, pts=None, hes=None):
        self._parent = parent
        self._vertices = pts
        self._ixs = hes

    @compute_once
    def _xvertices(self):
        """ CCW list of vertecies on boundary
            IN OTHER WORDS - REPRESENTING WHERE
        """
        pnts = np.asarray(list(self._parent.nodes), dtype=int)
        hull = scipy.spatial.ConvexHull(pnts)
        bnds = [(x, y) for x, y in zip(pnts[hull.vertices, 0], pnts[hull.vertices, 1])]
        od = {i: [bnds[i]] for i in range(-1, len(bnds)-1)}
        for x3, y3 in pnts:
            for i in range(-1, len(bnds)-1):
                (x1, y1), (x2, y2) = bnds[i], bnds[i+1]
                if (x3, y3) in [bnds[i], bnds[i+1]]:
                    break
                a = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                if a == 0:
                    od[i].append((x3, y3))
        extras = []
        for k, v in od.items():
            if len(v) == 1:
                pass
            elif v[0] < v[1]:
                v.sort()
            else:
                v.sort(reverse=True)
            extras += v
        minv = min(extras)
        min_ix = extras.index(minv)
        fl = extras[min_ix:] + extras[0:min_ix]
        return fl

    @lazyprop
    def edges(self):
        """ returns tuples [  ((x1, y1), (x2, y2)) ... ] of edge geometry"""
        bnd = self._vertices  # ccw
        return [tuple(sorted([bnd[i], bnd[i+1]])) for i in range(-1, len(bnd)-1)]

    @lazyprop   # CCW
    def ext_half_edges(self):
        """
        not-valid half edges that are on the boundary
        list of [ half_edge_coord, ... ]
        """
        bnd_ccw = self._vertices # ccw
        return [(bnd_ccw[i], bnd_ccw[i + 1]) for i in range(len(bnd_ccw) - 1)] + [(bnd_ccw[-1], bnd_ccw[0])]

    @lazyprop   # CW
    def int_half_edges(self):
        """
        VALID half edges that are on the boundary
        list of [ half_edge_coord, ... ]
        """
        bnd_cw = list(reversed(self._vertices))  # cw
        return [(bnd_cw[-1], bnd_cw[0])] + [(bnd_cw[i], bnd_cw[i + 1]) for i in range(len(bnd_cw)-1)]

    @lazyprop
    def half_edge_indicies(self):
        bnd_ccw = [self._parent.index_of_vertex(x) for x in self._vertices]
        rng = list(range(len(bnd_ccw) - 1)) + [-1]
        bnd = [(bnd_ccw[i], bnd_ccw[i + 1]) for i in rng]
        return [self._parent.index_of_half_edge(x) for x in bnd]

    @property
    def vertices(self):
        """ returns tuples [ [(x1, y1) ... ] of vertex geometry"""
        return self._vertices

    def __getitem__(self, item):
        return self._vertices[item]

    def __contains__(self, item):
        return

    def contains(self, value, ntype=None):
        return

    @classmethod
    def from_points(cls, parent, points):
        return cls(parent, pts=r2.sort_cw(points))

    @classmethod
    def from_kvs(cls, parent, kvs, points):
        assert len(kvs) == len(points)
        pts = r2.sort_cw(points)
        order = [None] * len(points)
        for i, p in enumerate(kvs):
            order[pts.index(points[i])] = p
        return cls(parent, pts=pts, hes=order)

    def __str__(self):
        return str(self.vertices)

    def __repr__(self):
        return '{}:{}'.format(self.__class__.__name__, str(self.vertices))

