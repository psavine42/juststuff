from cvxpy import *
import math


def order_constraints(list_of_orders, boxes, horizontal, margin=0):
    constraints = []
    for boxes_ixs in list_of_orders:
        curr = boxes[boxes_ixs[0]]
        for box_ix in boxes_ixs[1:]:
            if horizontal:
                constraints.append(curr.right + margin <= boxes[box_ix].left)
            else:
                constraints.append(curr.top + margin <= boxes[box_ix].bottom)
            curr = boxes[box_ix]
    return constraints


def is_fully_within(box1, box2, margin=0):
    """ b1 is fully within box2 """
    return [box1.x - margin >= box2.x,
            box1.y - margin >= box2.y,
            box1.x + box1.width + margin <= box2.x + box2.width,
            box1.y + box1.height + margin <= box2.y + box2.height
            ]


# def aspect_constraints(boxes):
#     constraints = []
#     for box in boxes:
#         constraints += [
#             (1 / box.ASPECT_RATIO) * box.height <= box.width,
#             box.width <= box.ASPECT_RATIO * box.height
#         ]
#     return constraints


def min_area_constraints(boxes):
    return [geo_mean(vstack([box.width, box.height])) >= math.sqrt(box.min_area)
            for box in boxes]


def within_boundary_constraints(boxes, height, width, margin=0):
    """# Enforce that boxes lie in bounding box."""
    constraints = []
    for box in boxes:
        constraints += [
            box.bottom >= margin,
            box.top + margin <= height,
            box.left >= margin,
            box.right + margin <= width
        ]
    return constraints


def no_overlaps(boxes, w, h):
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
            or_vars = Variable(shape=4, boolean=True,
                               name='overlap_or({},{})'.format(b1.name, b2.name))
            constraints += [
                b1.right <= b2.x + w * or_vars[0],
                b2.right <= b1.x + w * or_vars[1],
                b1.top   <= b2.y + h * or_vars[2],
                b2.top   <= b1.y + h * or_vars[3],
                sum(or_vars) <= 3
            ]
            # expr1, expr2, expr3, expr4, exprs]
    return constraints


def is_adjacent(box1, box2, margin=0, h=1e3, w=1e3):
    """ todo - seems right, check latter
    """
    or_vars = [Variable(boolean=True, name='adj({},{},{})'.format(box1.name, box2.name, k))
               for k in range(4)]
    lft_rgt = box1.left + margin <= box2.right + w * or_vars[0]
    expr2 = box1.top + margin <= box2.bottom + h * or_vars[1]

    expr3 = box1.right + margin <= box2.left + w * or_vars[2]
    expr4 = box1.bottom + margin <= box2.top + h * or_vars[3]

    v1, v2, v3, v4 = or_vars
    exprs = v1 + v2 + v3 + v4 >= 2
    constraints = [lft_rgt, expr2, expr3, expr4, exprs]
    return constraints


def ___must_be_adjacent(box1, box2, margin=0, h=1e3, w=1e3):
    """ REA: """
    constraints = []
    nvars = 2 ** (len(box1) + len(box2))
    or_vars = Variable(shape=nvars, boolean=True,
                       name='adj_or({},{})'.format(box1.name, box2.name))
    cnt = 0
    for bx1 in iter(box1):
        for bx2 in iter(box2):
            constraints += [
                bx1.left + margin <= bx2.right + w * or_vars[cnt],
                bx1.top + margin <= bx2.bottom + h * or_vars[cnt + 1],

                bx1.right + margin <= bx2.left + w * or_vars[cnt + 2],
                bx1.bottom + margin <= bx2.top + h * or_vars[cnt + 3],
            ]
            cnt += 4
    constraints.append(sum(or_vars) >= 2)
    # constraints.append(sum(or_vars) >= 3)
    return constraints


def must_be_adjacent(box1, box2, margin=0, h=1e3, w=1e3):
    """

    """
    constraints = []
    nvars = 2 ** (len(box1) + len(box2))
    or_vars = Variable(shape=nvars, boolean=True,
                       name='adj_or({},{})'.format(box1.name, box2.name))
    cnt = 0
    for bx1 in iter(box1):
        for bx2 in iter(box2):
            constraints += [
                bx1.left - bx2.right <= w * or_vars[cnt],
                bx1.top - bx2.bottom <= h * or_vars[cnt+1],

                bx1.right - bx2.left <= w * or_vars[cnt+2],
                bx1.bottom - bx2.top <= h * or_vars[cnt+3],
            ]
            cnt += 4
    # constraints.append(sum(or_vars) >= 2)
    constraints.append(sum(or_vars) == 2)
    return constraints


def grouping_constraint(boxes, group_area, aspect=10):
    """ allows N boxes to make an irregular group """
    constraints = []
    num_boxes = len(boxes)
    for b in boxes:
        b.min_area = None
        b.aspect = aspect

    for i in range(num_boxes):
        for j in range(i+1, num_boxes):
            constraints += must_be_adjacent(boxes[i], boxes[j])
            constraints.append(boxes[i].area + boxes[j].area >= math.sqrt(group_area))
    return constraints


def on_line():
    pass


def on_point(box, point, tol=0.1):
    constraints = [
        box.top - point[1] <= tol,
        box.bottom - point[1] <= tol,
        box.right - point[0] <= tol,
        box.left - point[0] <= tol,
    ]
    pass

