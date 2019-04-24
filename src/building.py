from shapely.geometry import MultiPolygon, Polygon, LineString, GeometryCollection
from shapely import ops
from shapely.affinity import translate
from shapely.geometry import JOIN_STYLE
from src.geom.cg import are_collinear
import numpy as np

_eps = 0.00001
_jnt = JOIN_STYLE.mitre


def round_shape(shp, n=3):
    if isinstance(shp, Polygon):
        coords = shp.exterior.coords
        return Polygon([(round(x, n), round(y, n)) for x, y in coords])
    # elif isinstance(shp, MultiPolygon):
    #     ml = []
    #     for sh in shp.geoms:
    #         coords = sh.exterior.coords
    #         mp.append(Polygon([(round(x, n), round(y, n)) for x, y in coords]))
    elif isinstance(shp, LineString):
        return LineString([(round(x, n), round(y, n)) for x, y in shp.coords])
    elif isinstance(shp, list):
        return [(round(x, n), round(y, n)) for x, y in shp]
    else:
        # print('rounding type not implemented for ', type(shp))
        return shp


def polygon_union(polys, round=0, eps=_eps):
    mpoly = ops.cascaded_union(polys)
    mpoly = mpoly.buffer(+eps, 1, join_style=_jnt) \
                 .buffer(-eps, 1, join_style=_jnt)
    return round_shape(mpoly, round)


class Area(Polygon):
    P = 1

    def __init__(self, args, env=None, name=None):
        if isinstance(args, np.ndarray):
            args = args.tolist()
        Polygon.__init__(self, args)
        self._name = name
        self._parent = env
        self._s = 0.25

    @classmethod
    def from_geom(cls, geom, **kwargs):
        if isinstance(geom, Polygon):
            return cls(round_shape(list(geom.exterior.coords), cls.P), **kwargs)

    @property
    def kwargs(self):
        return dict(name=self._name)

    @property
    def name(self):
        return self._name

    def __str__(self):
        return '{}: {}'.format(self._name, Polygon.__str__(self))

    def __copy__(self):
        return self.__class__(list(self.exterior.coords), **self.kwargs)

    def __hash__(self):
        center = list(self.centroid)
        center.append(self.area)
        center.append(self.length)
        return hash(tuple(center))

    def __sub_op(self, fn, this, other, **kw):
        """
        returning a copy of 'this' with metadata
        after processing and cleaning a shapely boolean op
        """
        try:
            if isinstance(other, dict):
                coords = fn(this, **other)
            elif other is None:
                coords = fn(this)
            else:
                coords = fn(this, other)

            if isinstance(coords, Polygon):
                coords = list(coords.exterior.simplify(self._s).coords)
                return self.__class__(coords, **kw)

            elif isinstance(coords, GeometryCollection):
                for item in coords.geoms:
                    if isinstance(item, Polygon):
                        crd = list(item.exterior.simplify(self._s).coords)
                        return self.__class__(crd, **kw)

            elif isinstance(coords, LineString):
                return coords.simplify(self._s)

            elif isinstance(coords, MultiPolygon):
                return coords.simplify(self._s)
        except Exception as e:
            return None

    @property
    def uvbounds(self):
        xmn, ymn, xmx, ymx = self.bounds
        cntr = self.centroid
        x, y = cntr.x, cntr.y
        u, v = (xmx - xmn) / 2, (ymx - ymn) / 2
        return [x, y, u, v]

    def __getitem__(self, item):
        """
        RL indexing scheme:
        0:                         -> whole_shape
        1...num_lines+1:           -> line[i]
        num_lines+1...2*num_lines: -> point[i]
        """
        ext = list(self.exterior.coords)[0:-1]
        num_points = len(ext)
        if item == 0:                   # select whole
            ixs = list(range(num_points))
        elif 0 < item < num_points:     # line at index i
            ixs = [item - 1, item - 0]
        elif item == num_points:        # line last point -> origin
            ixs = [item - 1, 0]
        elif item > num_points + 1:     # point at i
            print('WARNING POINT MOVE', item, num_points)
            ixs = [item - num_points]
        else:
            raise IndexError(str(item))
        return [(i, ext[i]) for i in ixs]

    def __setitem__(self, key, value):
        pass

    def update(self, item, xform):
        """
        item with
        update based on index
        # todo make xform an affine transform and dot them
        """
        ext = list(self.exterior.coords)[0:-1]
        ixs = self.__getitem__(item)
        for ix, _ in ixs:
            ext[ix] = tuple((np.asarray(list(ext[ix])) + xform).tolist())
        ls = LineString(ext).simplify(self._s)
        if len(ls.coords) > 2 and len(ls.coords) % 2 == 0:
            return self.__class__(ls, **self.kwargs)

    def validate(self):
        return self.is_valid and self.exterior.is_simple and self.area > 0

    def union(self, other):
        return self.__sub_op(polygon_union, [self, other], None, **self.kwargs)

    def translate(self, **kwargs):
        return self.__sub_op(translate, self, kwargs, **self.kwargs)

    def difference(self, other):
        return self.__sub_op(Polygon.difference, self, other, **self.kwargs)

    def intersection(self, other):
        return self.__sub_op(Polygon.intersection, self, other, **self.kwargs)


class Room(Area):
    def __init__(self, args, program=None, **kwargs):
        Area.__init__(self, args, **kwargs)
        self._program = program

    def walls(self):
        """ segments which have have walls on them
        store as seperate structure --
        """
        return

    @property
    def kwargs(self):
        return dict(name=self._name, program=self._program)

    @property
    def prog_type(self):
        return self._program

    def corners(self):
        return

    def drag(self, ix, direction):
        seg = self.sides[ix]
        return




