import enum


"""
https://github.com/psavine42/Spaceplanning_ADSK_PW
"""

class NodeType(enum.Enum):
    CONTAINER = 0
    SPACE = 1


class SDNode:
    def __init__(self, type, parent=None, left=None, right=None, center=None):
        self.node_type = type
        self.parent = parent
        self.left = left
        self.right = right
        self.num_nodes = 0
        self.center = center


def add_node(parent, child):
    if parent.left is not None and parent.right is not None:
        return False
    if child.node_type == NodeType.CONTAINER:
        parent.right = child
        child.parent = parent
    elif child.node_type == NodeType.SPACE:
        parent.left = child
        child.parent = parent
    return True


class SpaceDataTree:
    def __init__(self, root, origin, x, y):
        self.root = root

    def add_node(self, ):
        pass

