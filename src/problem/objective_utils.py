"""
Utility functions for generating problems and computing objectives

"""

import numpy as np
import skimage.morphology as skm
from skimage import measure


def max_boundnp(tensor):
    """ return bounding boxes for each plane
    Return
    ----
    size(6)
    """
    stat = []
    for i in range(tensor.shape[0]):
        inds = np.stack(np.nonzero(tensor[i]), -1)
        if inds.shape[0] == 0:
            stat.append(None)
            continue
        center = inds.mean(axis=0)
        xymin = np.min(inds, axis=0)
        xymax = np.max(inds, axis=0)
        stat.append(np.concatenate((center, xymin, xymax)))
    return stat


def convexity(mat, area=None):
    """ input - binary image """
    if area is None:
        area = mat.shape[0] * mat.shape[1]
    cvs = np.sum(skm.convex_hull_image(mat))
    return 1 - (cvs - area) / cvs


def aspect(bounds):
    xmin, ymin, xmax, ymax = bounds
    return min(xmax - xmin, ymax - ymin) / max(xmax - xmin, ymax - ymin)


def area(area, fp_area, v):
    return 1 - v - area / fp_area


def adjacency(rooms, dilation):
    if len(rooms) == 0:
        return 1
    n = 0
    # todo this should be a seperate features set and eval
    for other in rooms:
        adj = dilation + other
        n += 1 if np.where(adj == 2)[0].shape[0] >= 3 else 0
    return n / len(rooms)


def num_components(img):
    """
    number of connected components in a 2d binary image
    """
    _, n = measure.label(img, connectivity=1, return_num=True)
    return n
