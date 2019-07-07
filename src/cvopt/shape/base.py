

class BTile(object):
    def __init__(self, p1, p2=None):
        self.p1 = p1
        self.p2 = p2 if p2 else [p1[0] + 1, p1[1] + 1]

    @property
    def coords(self):
        p1, p2 = self.p1, self.p2
        return [tuple(x) for x in [p1, [p1[0], p2[1]], p2, [p2[0], p1[1]]]]

