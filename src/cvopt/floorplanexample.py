from cvxpy import *
import cvxpy.lin_ops
import cvxpy.utilities
import pylab
import math
import cassowary
import numpy as np

"""
CASSOWRY 
“inside,” “above,” “below,” “left-of,”
“right-of,” and “overlaps.”
"""

# https://github.com/cvxgrp/cvxpy/blob/master/examples/floor_packing.py
# Based on http://cvxopt.org/examples/book/floorplan.html

def order_constraints(list_of_orders, boxes, horizontal):
    constraints = []
    for boxes_ixs in list_of_orders:
        curr = boxes[boxes_ixs[0]]
        for box_ix in boxes_ixs[1:]:
            if horizontal:
                constraints.append(curr.right + FloorPlan.MARGIN <= boxes[box_ix].left)
            else:
                constraints.append(curr.top + FloorPlan.MARGIN <= boxes[box_ix].bottom)
            curr = boxes[box_ix]
    return constraints


def aspect_constraints(boxes):
    constraints = []
    for box in boxes:
        constraints += [
            (1 / box.ASPECT_RATIO) * box.height <= box.width,
            box.width <= box.ASPECT_RATIO * box.height
        ]
    return constraints


def min_area_constraints(boxes):
    return [geo_mean(vstack([box.width, box.height])) >= math.sqrt(box.min_area)
            for box in boxes]


def within_boundary_constraints(boxes, height, width):
    """# Enforce that boxes lie in bounding box."""
    constraints = []
    for box in boxes:
        constraints += [
            box.bottom >= FloorPlan.MARGIN,
            box.top + FloorPlan.MARGIN <= height,
            box.left >= FloorPlan.MARGIN,
            box.right + FloorPlan.MARGIN <= width
        ]
    return constraints


def no_overlaps(boxes, width, height):
    """http://yetanothermathprogrammingconsultant.blogspot.com/2017/07/rectangles-no-overlap-constraints.html
    xi+wi ≤ xj or
    xj+wj ≤ xi or
    yi+hi ≤ yj or
    yj+hj ≤ yi

    is transfromed to linear inequalities


    """
    constraints = []
    num_boxes = len(boxes)
    for i in range(num_boxes):
        for j in range(i+1, num_boxes):
            b1, b2 = boxes[i], boxes[j]
            or_vars = [Variable(boolean=True, name='delta_{},{},{}'.format(i, j, k)) for k in range(4)]

            expr1 = b1.x + b1.width <= b2.x + width * or_vars[0]
            expr2 = b2.x + b2.width <= b1.x + width * or_vars[1]
            expr3 = b1.y + b1.height <= b2.y + height * or_vars[2]
            expr4 = b2.y + b2.height <= b1.y + height * or_vars[3]

            v1, v2, v3, v4 = or_vars
            exprs = v1 + v2 + v3 + v4 <= 3

            constraints += [expr1, expr2, expr3, expr4, exprs]
    return constraints


def is_fully_within(box1, box2, margin=0):
    """ b1 is fully within box2 """
    return [box1.x - margin >= box2.x,
            box1.y - margin >= box2.y,
            box1.x + box1.width + margin <= box2.x + box2.width,
            box1.y + box1.height + margin <= box2.y + box2.height
            ]


class Box(object):
    """ A box in a floor packing problem. """
    ASPECT_RATIO = 2.0

    def __init__(self, min_area,
                 max_area=None,
                 min_dim=None,
                 max_dim=None,
                 name=None,
                 aspect=2.0
                 ):
        # CONSTRAINTS
        self.name = name
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.max_area = max_area
        self.min_area = min_area
        self.aspect = aspect

        # VARS
        self.height = Variable(pos=True, name='{}_h'.format(name))
        self.width = Variable(pos=True, name='{}_w'.format(name))
        self.x = Variable(pos=True, name='{}_x'.format(name))
        self.y = Variable(pos=True, name='{}_y'.format(name))

    def __str__(self):
        return 'x: {}, y: {}, w: {} h: {}'.format(
            *[round(x, 2) for x in [self.x.value, self.y.value, self.width.value, self.height.value]]
        )

    @property
    def position(self):
        return np.round(self.x.value, 2), np.round(self.y.value, 2)

    @property
    def size(self):
        return np.round(self.width.value, 2), np.round(self.height.value, 2)

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y

    @property
    def top(self):
        return self.y + self.height

    def own_constraints(self):
        constraints = []
        if self.aspect:
            constraints.append((1 / self.aspect) * self.height <= self.width)
            constraints.append(self.width <= self.aspect * self.height)

        if self.min_area:
            constraints.append(geo_mean(vstack([self.width, self.height])) >= math.sqrt(self.min_area))

        if self.min_dim:
            # height and width must be atleast
            constraints.append(self.height >= self.min_dim)
            constraints.append(self.width >= self.min_dim)

        if self.max_dim:
            # height and width must less than
            constraints.append(self.height <= self.max_dim)
            constraints.append(self.width <= self.max_dim)

        return constraints


class FloorPlan(object):
    """ A minimum perimeter floor plan. """
    MARGIN = 1.0
    ASPECT_RATIO = 5.0

    def __init__(self, boxes, eps=1e4):
        self.boxes = boxes
        self.eps = eps
        self.height = Variable(pos=True)
        self.width = Variable(pos=True)
        self.horizontal_orderings = []
        self.vertical_orderings = []

    @property
    def size(self):
        return np.round(self.width.value, 2), np.round(self.height.value, 2)

    @staticmethod
    def _order(boxes, horizontal):
        # Return constraints for the ordering.
        if len(boxes) == 0:
            return
        constraints = []
        curr = boxes[0]
        for box in boxes[1:]:
            if horizontal:
                constraints.append(curr.right + FloorPlan.MARGIN <= box.left)
            else:
                constraints.append(curr.top + FloorPlan.MARGIN <= box.bottom)
            curr = box
        return constraints

    def layout(self):
        # Compute minimum perimeter layout.
        constraints = []
        for box in self.boxes:
            # Enforce that boxes lie in bounding box.
            constraints += [
                box.bottom >= FloorPlan.MARGIN,
                box.top + FloorPlan.MARGIN <= self.height
            ]

            constraints += [
                box.left >= FloorPlan.MARGIN,
                box.right + FloorPlan.MARGIN <= self.width
            ]

            # Enforce aspect ratios.
            constraints += [
                (1/box.ASPECT_RATIO) * box.height <= box.width,
                box.width <= box.ASPECT_RATIO*box.height
            ]

            # Enforce minimum area
            constraints += [
                geo_mean(vstack([box.width, box.height])) >= math.sqrt(box.min_area)
            ]

            # Enforce maximum area
            #if box.max_area:
            #     constraints += [
            #         geo_mean(vstack([box.width, box.height])) <= box.max_area
            #    ]

        # boxA above boxB
        # Enforce the relative ordering of the boxes.
        for ordering in self.horizontal_orderings:
            constraints += self._order(ordering, True)

        for ordering in self.vertical_orderings:
            constraints += self._order(ordering, False)

        # build the problem
        problem = Problem(Minimize(2*(self.height + self.width)), constraints)
        return problem

    def problem(self, horizantal, vertical):
        constraints = []
        for box in self.boxes:
            constraints += box.own_constraints()
        constraints += order_constraints(horizantal, self.boxes, True)
        constraints += order_constraints(vertical, self.boxes, False)
        constraints += within_boundary_constraints(self.boxes, self.height, self.width)
        problem = Problem(Minimize(2 * (self.height + self.width)), constraints)
        return problem

    def own_constraints(self):
        return [self.width <= self.eps, self.height <= self.eps]

    def problem2(self):
        constraints = []
        for box in self.boxes:
            constraints += box.own_constraints()

        constraints += within_boundary_constraints(self.boxes, self.height, self.width)
        constraints += no_overlaps(self.boxes, self.eps, self.eps)
        problem = Problem(Minimize(2 * (self.height + self.width)), constraints)
        return problem

    # Show the layout with matplotlib
    def show(self):
        pylab.figure(facecolor='w')
        for k in range(len(self.boxes)):
            box = self.boxes[k]
            x,y = box.position
            w,h = box.size
            pylab.fill([x, x, x + w, x + w],
                       [y, y+h, y+h, y],
                       facecolor='#D0D0D0',
                       # color='r',
                       edgecolor='k'
                       )
            pylab.text(x+.5*w, y+.5*h, "%d" %(k+1))
        x,y = self.size
        pylab.axis([0, x, 0, y])
        pylab.xticks([])
        pylab.yticks([])

        pylab.show()


def generate_packing_opts(base_adj, num):
    """ start with minimal """
    # specified
    used = set(np.unique(np.array(base_adj)).tolist())

    unspecified = set(list(range(num))).difference(used)
    ix_vertical = []
    ix_vertical = []

    for kv in used:
        kv

    for pair in base_adj:
        pass


def __boxes1():
    return [Box(200, name=1), Box(80, name=2),
            Box(80, name=4), Box(80, name=3),
            Box(80, name=5)]


def test_prob1():
    """ original form """
    boxes = __boxes1()
    fp = FloorPlan(boxes)
    fp.horizontal_orderings.append( [boxes[0], boxes[2], boxes[4]] )
    fp.horizontal_orderings.append( [boxes[1], boxes[2]] )
    fp.horizontal_orderings.append( [boxes[3], boxes[4]] )
    fp.vertical_orderings.append( [boxes[1], boxes[0], boxes[3]] )
    fp.vertical_orderings.append( [boxes[2], boxes[3]] )
    problem = fp.layout()
    print(problem.is_dcp(), problem.is_dqcp(), problem.is_qp(),
          problem.is_dgp(), problem.is_mixed_integer())

    problem.solve()
    for box in boxes:
        print(str(box))
    print(problem)
    print(problem.solution)
    fp.show()


def test_prob2():
    """ """
    boxes = __boxes1()
    fp = FloorPlan(boxes)

    adjacency = [[0, 2], [1, 3]]

    hmat = [[0, 2, 4], [1, 2], [3, 4]]
    vmat = [[1, 0, 3], [2, 3]]

    problem = fp.problem(hmat, vmat)
    print(problem.is_dcp(), problem.is_dqcp(), problem.is_qp(),
          problem.is_dgp(), problem.is_mixed_integer())
    print(problem)
    problem.solve()
    print(problem.solution)
    for box in boxes:
        print(str(box))

    fp.show()


def test_prob3():
    """ with better no-overlap constraints """
    boxes = __boxes1()
    fp = FloorPlan(boxes)

    problem = fp.problem2()
    print(problem.is_dcp(), problem.is_dqcp(), problem.is_qp(),
          problem.is_dgp(), problem.is_mixed_integer())
    print(problem)
    problem.solve()
    print(problem.solution)
    for box in boxes:
        print(str(box))
    fp.show()


if __name__ == '__main__':
    test_prob3()

