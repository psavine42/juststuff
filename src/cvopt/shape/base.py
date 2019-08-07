

class BTile(object):
    def __init__(self, p1=None, aspect=None, p2=None, area=None, **kwargs):
        self.p1 = p1
        self._area = area
        self._aspect = aspect
        self.p2 = None
        if p1 is not None:
            self.p2 = p2 if p2 else [p1[0] + 1, p1[1] + 1]

    @property
    def coords(self):
        p1, p2 = self.p1, self.p2
        return [tuple(x) for x in [p1, [p1[0], p2[1]], p2, [p2[0], p1[1]]]]

    @property
    def area(self):
        if self._area is not None:
            return self._area
        (x1, y1), (x2, y2) = self.p1, self.p2
        return (x2 - x1) * (y2 - y1)

    @property
    def height(self):
        (x1, y1), (x2, y2) = self.p1, self.p2
        return y2 - y1

    @property
    def width(self):
        (x1, y1), (x2, y2) = self.p1, self.p2
        return x2 - x1

    def __getitem__(self, item):
        return getattr(self, item, None)

    @property
    def aspect(self):
        if self._aspect is not None:
            return self._aspect
        return max(self.height, self.width) / min(self.height, self.width)



class R2(object):
    def __init__(self, bounds=None):
        self._bounds = bounds

    @property
    def edges(self):
        return None

