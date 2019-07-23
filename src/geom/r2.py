import numpy as np
import math
import operator
from functools import reduce


def sort_cw(coords):
    center = tuple(map(operator.truediv,
                       reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    return sorted(coords, key=lambda coord: (-135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)


def deep_tup(x):
    """fully copies trees of tuples or lists to a tree of lists.
         deep_list( (1,2,(3,4)) ) returns [1,2,[3,4]]
         deep_list( (1,2,[3,(4,5)]) ) returns [1,2,[3,[4,5]]]"""
    if not isinstance(x, (tuple, list)):
        return x
    return tuple(list(map(deep_tup, x)))


def rot_mat(theta):
    return np.asarray([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


def trans_mat(vec):
    assert len(vec) == 2
    return np.asarray([
        [1, 0, vec[0]],
        [0, 1, vec[1]],
        [0, 0, 1]
    ])


def compose_mat(scale=None, shear=None, angle=None, translate=None):
    """Return transformation matrix from sequence of transformations.

    This is the inverse of the decompose_matrix function.

    Sequence of transformations:
        scale : vector of 3 scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        angles : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : perspective partition of matrix
    """
    M = np.identity(3)
    if translate is not None:
        T = np.identity(3)
        T[:2, 2] = translate[:2]
        M = np.dot(M, T)
    if angle is not None:
        R = rot_mat(angle)
        M = np.dot(M, R)
    M /= M[2, 2]
    return M


def transform(points, xform, dtype=None):
    assert xform.shape == (3, 3)
    xform_rot = xform[:2, :2]
    xform_trn = xform[:2, -1]
    pts_np = np.asarray(points)
    # pts_np = np.asarray([list(p) + [1] for p in points])
    res = np.dot(np.dot(pts_np, xform_rot), xform_trn)
    if dtype:
        res = res.astype(dtype)
    if isinstance(points, (list, tuple)):
        return deep_tup(res)
    return res


def translate(xs, vec):
    vec_np = np.asarray(vec)
    xs_np = np.asarray(xs)

    res = vec_np + xs_np
    if isinstance(xs, (list, tuple)):
        return deep_tup(res)
    return res


def to_homo(xs, dtype=None):
    vec_np = np.asarray([list(x) + [1] for x in xs], dtype=dtype)
    return vec_np


def from_homo(xs, dtype=None, round=None):
    if round:
        xs = np.round(xs, round)
    if dtype:
        xs = xs.astype(dtype=dtype)
    res = deep_tup(xs[:, 0:2].tolist())
    return res


def verts_to_edges(coords):
    base = [(coords[i], coords[i+1]) for i in range(len(coords)-1)]
    return base + [(coords[-1], coords[0])]


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2':
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if v1_u[0] * v2_u[1] - v1_u[1] * v2_u[0] < 0:
        angle = -angle
    return angle


def align_vec(vec_src, vec_tgt):
    """
    get the transformation matrix to project vec_src onto vec_tgt
    with rotation and translation
    """
    src = to_homo(vec_src)
    tgt = to_homo(vec_tgt)
    lin = tgt[0] - src[0]
    u1, u2 = src[1] - src[0], tgt[1] - tgt[0]
    theta = angle_between(u1[:2], u2[:2])
    return compose_mat(angle=theta, translate=lin)


def _align_vec(vec_src, vec_tgt):
    src_v = to_homo(vec_src)
    tgt_v = to_homo(vec_tgt)
    u1, u2 = src_v[1] - src_v[0], tgt_v[1] - tgt_v[0]
    v1_u = unit_vector(u1)
    v2_u = unit_vector(u2)
    print(v1_u, v2_u)
    src = np.eye(3)
    src[0, 2] = v1_u[0]
    src[1, 2] = v1_u[1]
    tgt = np.eye(3)
    tgt[0, 2] = v2_u[0]
    tgt[1, 2] = v2_u[1]
    trans = np.dot(np.linalg.inv(src), tgt)
    return trans


def centroid(pts):
    pts = np.asarray(pts)
    return pts.mean(axis=0).tolist()




