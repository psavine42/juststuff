import operator
import numpy as np
import uuid


class FloatingNode:
    def __init__(self, val=None):
        if isinstance(val, np.ndarray):
            val = val.tolist()
        self._value = tuple(val)
        self._id = str(uuid.uuid4())
        self.left = None
        self.right = None

    # def __getitem__(self, item):
    @property
    def value(self):
        return self._value

    @property
    def np(self):
        return np.asarray(list(self._value))

    def __comp(self, other, op):
        if isinstance(other, self.__class__):
            return op(self._value, other._value)
        elif other is None:
            return True
        return False

    def __gt__(self, other):
        return self.__comp(other, operator.gt)

    def __lt__(self, other):
        return self.__comp(other, operator.lt)

    def __len__(self):
        return len(list(self.__iter__()))

    def __getitem__(self, item):
        ln = 0
        q = [self]
        while q:
            node = q.pop()
            if isinstance(item, tuple) and node.value == item:
                return node
            if isinstance(item, int) and ln == item:
                return node
            if node.left is not None:
                ln += 1; q.append(node.left)
                q.append(node.left)
            if node.right is not None:
                ln += 1; q.append(node.right)
        raise IndexError()

    def copy(self, parent=None, left=True):
        new = FloatingNode(self._value)
        if parent is not None:
            if left is True:
                parent.left = new
            else:
                parent.right = new
        if self.left is not None:
            self.left.copy(parent=new, left=True)
        if self.right is not None:
            self.right.copy(parent=new, left=False)
        return new

    def update_(self, transform):
        """ INPLACE transform is 3 x 3 matrix ?? """
        # x = np.dot(self.np, transform)
        x = self.np + transform
        self._value = tuple(x[0:2].tolist())
        if self.left is not None:
            self.left.update_(transform)
        if self.right is not None:
            self.right.update_(transform)

    def add_child(self, node):
        if self.left is None and self.right is None:
            self.left = node
        elif self.left is not None and self.right is None:
            if node < self.left:
                pass

    def split(self, c1, c2):
        pass

    @property
    def isleaf(self):
        return self.left is None and self.right is None

    def get_boundary(self):
        return [n.value for n in iter(self) if n.isleaf is True]

    def __iter__(self):
        q = [self]
        while q:
            node = q.pop(0)
            if node.left is not None:
                q.append(node.left)
            if node.right is not None:
                q.append(node.right)
            yield node

    def points(self):
        return [n.np for n in iter(self)]

    def edges(self):
        p1 = self.np
        if self.left is not None:
            yield np.stack([p1, self.left.np])
            for edge in self.left.edges():
                yield edge
        if self.right is not None:
            yield np.stack([p1, self.right.np])
            for edge in self.right.edges():
                yield edge


class PolyTree(FloatingNode):
    def __init__(self, seed):
        FloatingNode.__init__(self, val=seed)

    def __hash__(self):
        return 0

    def update(self, node_id, xform):
        """ if the parent """
        new = self.copy()
        node = new[node_id]
        node.update_(xform)
        return new


