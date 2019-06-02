import torch


def max_bound(tensor):
    """ return bounding boxes for each plane
    Return
    ----
    size(6)
    """
    num = tensor.size(0)
    amax = torch.argmax(tensor, 0)
    stat = torch.zeros(num, 4)
    for i in range(num):
        inds = (amax == i).nonzero()
        # print(inds)
        if len(inds) == 0:
            stat[i] = torch.zeros(4).long()
        else:
            xmin = inds[:, 0].min(0)[0]
            ymin = inds[:, 1].min(0)[0]
            xmax = inds[:, 0].max(0)[0]
            ymax = inds[:, 1].max(0)[0]
            stat[i] = torch.stack((xmin, ymin, xmax, ymax))
    return stat


def maxbound_diff(tensor):
    """ ensure backprop where the indices area min/max"""
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
    area_chan = tensor.sum(dim=(-1, -2)).float()
    dx = (bounds[:, 2] - bounds[:, 0]).float()
    dy = (bounds[:, 3] - bounds[:, 1]).float()
    res = (dx * dy) / area_chan
    res[res != res] = 0
    return res


def dilate(tnsr, nways=4):
    """

    :param tensor: LongTensor or ByteTensor of dim=3
    :param nways: 4 or 8 components
    :return: dilated version of tnsr
    """
    assert nways in [4, 8], 'num compnents must be 4 or 8, got {}'.format(nways)
    ix = (tnsr == 1).nonzero()  # [3, N]
    dilated = tnsr.clone().detach_()
    size_ = tnsr.size(-1) - 1
    if nways == 4:
        add_ = torch.tensor([[0, 0, 1],  [0, 1, 0],
                             [0, -1, 0], [0, 0, -1]])
    else:
        add_ = torch.tensor([[0, 1, 1],  [0, 1, -1],
                             [0, -1, 1], [0, -1, -1],
                             [0, 0, 1],  [0, 1, 0],
                             [0, -1, 0], [0, 0, -1]])
    for i in range(nways):
        z = (ix + add_[i]).clamp(min=0, max=size_).long()
        dilated[z[:, 0], z[:, 1], z[:, 2]] = 1
    return dilated


def box_aspect(bounds):
    bnd = bounds[:, 2:] - bounds[:, :2]
    res = bnd.min(dim=-1)[0] / bnd.max(dim=-1)[0]
    res[torch.isnan(res)] = 0
    return res


def adjacencies(rooms, dilations=None):
    """
        [ num_spaces, h, w ]
        [ num_spaces, h, w ]

    returns:
        - [ num_spaces, num_spaces ] adjacency matrix
    """
    if dilations is None:
        dilations = dilate(rooms)
    adjs = torch.eye(rooms.size(0))
    for i in range(len(rooms)):
        for j in range(i+1, len(rooms)):
            dx = ((dilations[j] + rooms[i]) > 1).nonzero()
            adjs[i, j] = dx.size(0)
    return adjs


def field_area(bbxs):
    if bbxs.dim() < 2:
        raise Exception('expected dim > 2')
    elif bbxs.dim() == 2:
        # single matrix
        return bbxs.sum(dim=(-1))
    return bbxs.sum(dim=(-1, -2))


# -----------------------------------------------------------
def box_area(boxes):
    if boxes.dim() == 1:
        boxes.unsqueeze_(0)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_bound(boxes):
    xmn = boxes[:, 0].min(keepdim=True)[0]
    ymn = boxes[:, 1].min(keepdim=True)[0]
    xmx = boxes[:, 2].max(keepdim=True)[0]
    ymx = boxes[:, 3].max(keepdim=True)[0]
    return torch.cat((xmn, ymn, xmx, ymx))


def approx_box_cvx(bounds):
    dx = (bounds[:, 2] - bounds[:, 0]).float()
    dy = (bounds[:, 3] - bounds[:, 1]).float()
    res = (dx * dy)
    # res[res != res] = 0
    return res


def box_intersection(bb1, bb2):
    """
    Calculate the Intersection of two bounding boxes.

    Parameters
    ----------
    bb1 : tensor
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : tensor
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    tensor
    """
    if bb1.dim() == 1:
        bb1 = bb1.unsqueeze(0)
    if bb2.dim() == 1:
        bb2 = bb2.unsqueeze(0)
    # determine the coordinates of the intersection rectangle
    # top, left, bottom, right
    x_lft = torch.max(torch.cat((bb1[:, 0], bb2[:, 0]), -1), dim=-1, keepdim=True)[0]
    y_top = torch.max(torch.cat((bb1[:, 1], bb2[:, 1]), -1), dim=-1, keepdim=True)[0]
    x_rgt = torch.min(torch.cat((bb1[:, 2], bb2[:, 2]), -1), dim=-1, keepdim=True)[0]
    y_btm = torch.min(torch.cat((bb1[:, 3], bb2[:, 3]), -1), dim=-1, keepdim=True)[0]
    # print(y_btm)

    # todo make this masky
    if x_rgt < x_lft or y_btm < y_top:
        return None
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    # intersection_area = (x_rgt - x_lft) * (y_btm - y_top)

    # compute the area of both AABBs
    # bb1_area = (bb1[: 2] - bb1[: 0]) * (bb1[: 3] - bb1[: 1])
    # bb2_area = (bb2[: 2] - bb2[: 0]) * (bb2[: 3] - bb2[: 1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    # iou = intersection_area / (bb1_area + bb2_area - intersection_area)
    # assert iou >= 0.0
    # assert iou <= 1.0
    return torch.cat((x_lft, y_top, x_rgt, y_btm), 0).unsqueeze(0)


def box_max(bb1, bb2):
    """
    Calculate the corner Union of two bounding boxes.
    """
    if bb1.dim() == 1:
        bb1.unsqueeze_(0)
    if bb2.dim() == 1:
        bb2.unsqueeze_(0)
    # determine the coordinates of the intersection rectangle
    x_lft = torch.min(torch.cat((bb1[:, 0], bb2[:, 0]), -1), dim=-1, keepdim=True)[0]
    y_top = torch.min(torch.cat((bb1[:, 1], bb2[:, 1]), -1), dim=-1, keepdim=True)[0]
    x_rgt = torch.max(torch.cat((bb1[:, 2], bb2[:, 2]), -1), dim=-1, keepdim=True)[0]
    y_btm = torch.max(torch.cat((bb1[:, 3], bb2[:, 3]), -1), dim=-1, keepdim=True)[0]
    return torch.cat((x_lft, y_top, x_rgt, y_btm), 0).unsqueeze(0)


def bbx_gt_(bb1, bb2):
    # print(bb1.shape, bb2.shape)
    # todo make this raetuyrn an bit-mask

    if bb1[:, 0] > bb2[:, 0]:
        return True
    elif bb1[:, 0] == bb2[:, 0] and bb1[:, 1] > bb2[:, 1]:
        return True
    elif bb1[:, 0] == bb2[:, 0] and bb1[:,1] == bb2[:,1] and bb1[:, 2] > bb2[:, 2]:
        return True
    elif bb1[:, 0] == bb2[:, 0] and bb1[:,1] == bb2[:,1] and bb1[:,2] == bb2[:, 2] and bb1[:, 3] > bb2[:,3]:
        return True
    return False


def intersection_kind(bb1, bb2):
    inters_kind = (bb1 == bb2).sum().item()
    return inters_kind


def sort_boxes(bbs):
    pass


def diff_areas(bb1, bb2):
    """ bb1 - bb2 -> (1:2) sub-boxes of bb1

        s.t. bb1 > bb2

    Returns
    -----------
        - if there is no overlap -> None
        - if there is regular overlap -> None
    """
    # if bb1[0] <= bb2[0]:


    if bb1.dim() == 1:
        bb1.unsqueeze_(0)
    if bb2.dim() == 1:
        bb2.unsqueeze_(0)

    b_gt = bbx_gt_(bb1, bb2)

    inters_kind = (bb1 == bb2).sum().item()
    inters = box_intersection(bb1, bb2)
    if inters is None:
        return None

    if inters_kind == 0:
        if bbx_gt_(bb1, bb2) is True:
            box1 = torch.cat((bb1[:, 0], bb1[:, 1], bb2[:, 0], bb1[:, 3]))
            box2 = torch.cat((bb2[:, 0], bb1[:, 1], bb1[:, 2], bb2[:, 1]))
            return box1, box2
        else:
            box1 = torch.cat((bb1[:, 0], bb2[:, 3], bb2[:, 2], bb1[:, 3]))
            box2 = torch.cat((bb2[:, 2], bb1[:, 1], bb1[:, 2], bb1[:, 3]))
            return box1, box2
    elif inters_kind == 1:
        if bbx_gt_(bb1, bb2) is True:
            pass

    elif inters_kind == 2:
        pass

    elif inters_kind == 3:
        pass

    return None


def shatter_with(bb1, bb2):
    """ shatter a box with the intersection """
    # inters_kind = (bb1 == inters)
    # if inters_kind.sum().item() == 1:
    #     # bb1[~inters_kind.byte()]
    #     t1 = torch.where((bb1 == inters), bb1, inters)
    inters = box_intersection(bb1, bb2)
    if inters is None:
        return None
    pts1 = bb1.view(-1, 2)
    intp = inters.view(-1, 2)
    rem_corner = (pts1 == intp)

    proj_corner = intp[rem_corner.rot90(2)]
    rot1 = intp[rem_corner.rot90(1)]
    rot2 = intp[rem_corner.rot90(-1)]

    keep_corner = pts1[rem_corner.rot90(2)]



def merge_areas(bb1, bb2):
    """ bb1 + bb2 -> (3) sub-boxes of bb1


    """

    inters = box_intersection(bb1, bb2)
    if inters is None:
        return None

    ikind = intersection_kind(bb1, bb2)
    if ikind == 0:
        bmax = box_max(bb1, bb2)
        print(bmax.shape, inters.shape)
        # box1 ok
        box1 = torch.cat((bmax[:, 0], bmax[:, 1], inters[:, 0], inters[:, 3]))
        # box2
        box2 = torch.cat((inters[:, 0], bmax[:, 1], inters[:, 2], bmax[:, 3]))
        # box3
        box3 = torch.cat((inters[:, 2], inters[:, 1], bmax[:, 2], bmax[:, 3]))
        return box1, box2, box3
    elif ikind == 1:
        # -> 1
        pass

    elif ikind == 2:
        # -> 1
        pass
    elif ikind == 3:
        pass



# ----------------------------------------
def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points, indices = points.sort()

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    return lower[:-1] + upper[:-1]


def stupid_box_adj(boxes):

    ngroup = len(boxes)
    adj_mat = torch.eye(ngroup)
    for i in range(ngroup):
        for j in range(i+1, ngroup):
            for j in range(i + 1, ngroup):
                pass

def box_objective(boxes):
    """
    input: list of tensor.size([ num_boxes, 4])
    apparentyly convex hull is its own thang

    https://github.com/Thidtc/PointerNetwork-PyTorch
    compute continuous:
        - dilations
        - areas
        - adjacencies
        - convexities
    """
    # precomputed
    areas = torch.stack((box_area(x).sum() for x in boxes))
    bounds = torch.stack([box_bound(x) for x in boxes])

    # loss components
    delta_aspect = torch.abs(box_aspect(bounds))
    delta_convex = approx_box_cvx(bounds)       # todo - make this better

    return torch.stack((areas, delta_convex, delta_aspect, delta_adjs))


def conv_objective(layout, constraints, adjs):
    """

        layoout     [ num_region, h , w ]
        constraints [ num_region, num_constraint ]
        adjs        [ num_region, num_region ]

    returns
        - auxs  [ num_region, num_constraint ]

    the delta targets , action results
    """
    area = layout.size(-1) * layout.size(-2)

    bounds = max_bound(layout)
    dilateds = dilate(layout)
    convexes = approx_convex_eval(layout)
    areas = layout.sum(dim=(-1, -2)) / area
    cur_adjs = adjacencies(layout, dilateds)

    # these are all vecs of size [ num_spaces ]
    delta_area = torch.abs(constraints[:, 0] - areas)       # ok

    # convex eval needs to backprop through
    delta_convex = torch.abs(constraints[:, 1] - convexes)
    delta_aspect = torch.abs(constraints[:, 2] - box_aspect(bounds))
    delta_adjs = torch.mul(adjs, cur_adjs).sum(dim=(-1))

    aux_mat = torch.stack((delta_area, delta_convex, delta_aspect, delta_adjs))

    return aux_mat


