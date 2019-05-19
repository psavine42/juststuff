import torch


def max_bound(tensor):
    """ return bounding boxes for each plane
    Return
    ----
    size(6)
    """
    num = tensor.size(0)
    amax = torch.argmax(tensor, 0)
    stat = []
    for i in range(num):
        inds = (amax == i).nonzero()
        xymin = inds.argmin(0)
        xymax = inds.argmax(0)
        stat.append(torch.cat((xymin, xymax)))
    return torch.stack(stat)


def maxbound2(tensor):
    idx = (tensor == 1).nonzero()
    pass


def diff_convex_hull():
    """
    http://gamma.cs.unc.edu/volccd/ghull/

    """
    pass


def approx_convex_eval(tensor, bounds=None):
    """
    area of convex hull appriximated by a box
    inputs:
        tensor: [C, w, h]

    Returns:
        tensor [C, 1]
        area of channel / area of convex_bound of channel
    """
    if bounds is None:
        bounds = max_bound(tensor)
    area_chan = tensor.sum(dim=(-1, -2))
    dx = bounds[:, 2] - bounds[:, 0]
    dy = bounds[:, 3] - bounds[:, 1]
    return area_chan / (dx * dy)


def dilation(tensor):
    # index-based dialation on gpu
    ix = (tensor == 1).nonzero()
    z = torch.zeros(4, *ix.size())
    z[0, :, 1:] = [ 1, 1]
    z[1, :, 1:] = [-1, 1]
    z[2, :, 1:] = [ 1, -1]
    z[3, :, 1:] = [-1, -1]


