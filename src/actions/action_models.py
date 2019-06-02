"""
conviniences wrappers for actions



box actions 5
    [index, xmin, ymin, xmax, ymax]

tupple actions
    (goal, geom)
"""
import torch
import enum
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict as odict
from functools import reduce
from src.probablistic.utils import *


class AType(enum.Enum):
    ONEHOT = 0
    POINT = 1
    LINE = 2


# --------------------------------------------------------------------------
# Policy Models : z -> action
# --------------------------------------------------------------------------
def mse_from_tgt_indicies(tnsr, targets):
    """
    tnsr is one_hot encoding
    targets are (0, 1)
    """
    y_ix = (targets * tnsr.size(-1)).round().long()
    tgt = torch.zeros_like(tnsr)
    tgt.scatter_(1, y_ix.view(-1, 1), 1)
    return F.mse_loss(tnsr, tgt)


def disc_cont_loss(predicted, targets):
    """
    used for case where prediction is a a tuple of
        - predicted tuple
            - logits    [ b, 1, S ]
            - geom      [ b, 4, N ]
        - targets [b , [index, x0, y0, x1, y1]]
    """
    yp, ap = predicted
    yt, at = targets
    loss_actn = F.mse_loss(yp, yt)
    loss_geom = F.mse_loss(ap, at)
    return loss_actn + loss_geom


def disc_disc_loss(predicted, targets):
    """
    predicted:
        - logits    [ b, 1, S ]
        - geom      [ b, 4, N ]
    targets:
        - FloatTensor([b, 5])
        [b , [index, x0, y0, x1, y1]]
    """
    yp, ap = predicted
    yt, at = targets
    loss_actn = F.mse_loss(yp, yt)
    loss_geom = F.mse_loss(ap, at)
    return loss_geom + loss_actn


def map_mse(predictions, targets, coefs=None):
    assert len(predictions) == len(targets)
    if coefs:
        assert len(coefs) == len(targets)
    loss = 0
    for p, t in zip(predictions, targets):
        loss = loss + F.mse_loss(p, t)
    return loss


def composite_action_to_cont_box(x):
    y, action_box = x
    idx = torch.max(y, dim=1, keepdim=True)[1]
    return torch.cat((idx.float()/y.size(-1), action_box), -1)


def disc_disc_action_to_cont_box(x):
    y, a = x
    y_idx = torch.argmax(y, dim=1, keepdim=True).float().unsqueeze(0) # / y.size(-1)
    a_idx = torch.argmax(a, dim=1, keepdim=True).float() # / a.size(-1)
    return torch.cat((y_idx, a_idx), -1)


# ------------------------------------------------------------
class ActionReg(object):
    """
    each action can be a list of sub actions

    """
    def __init__(self, shape, name=None):
        if not isinstance(shape, (tuple, list, dict, ActionReg, int)):
            raise Exception('shape must be a list or dict ')

        self._name = name
        self._def = []
        self._children = None

        if isinstance(shape, int):
            self._def = [shape]
        elif all([isinstance(x, int) for x in shape]):
            self._def = shape
        else:
            self._children = []
            for sub_shape in shape:
                if isinstance(sub_shape, ActionReg):
                    self._children.append(sub_shape)
                else:
                    self._children.append(ActionReg(sub_shape))

    # pytorchies -----------------------------------------------------
    def numel(self):
        if self._children:
            return reduce(lambda x, y: x + y, [x.numel() for x in self._children])
        return reduce(lambda a, x: a * x, self._def)

    def dim(self):
        if self._children:
            return [x.dim() for x in self._children]
        return len(self._def)

    def size(self):
        if self._children:
            return [x.size() for x in self._children]
        return self._def

    # -----------------------------------------------------
    def split(self, x):
        """ torch.tensor with batch dim 0
         Returns
            tensors of size
         """
        numel = self.numel()
        assert x.numel() % numel == 0, 'not broadcastable'

        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 2:
            pass
        else:
            x = flatten(x)

        ch = 0
        res = []
        for child in self._children:
            take = child.numel()
            res.append(x[:, ch:ch+take])
            ch += take
        return res

    def _loss_fn(self, pred, target):
        return

    # -------------------------------------------------------------
    def __len__(self):
        if self._children is None:
            return 0
        return len(self._children)

    def __str__(self):
        return '{}, {}: {}'.format(self.__class__.__name__, self._name, self.numel())

    def __repr__(self):
        return self.__str__()

    # -------------------------------------------------------------
    def apply(self, fs, xs):
        """
        params
            - xs: list of tensors
            - fs: list of functions of same structure
        """
        if self._children:
            res = []
            if isinstance(fs, (list, tuple)):
                assert len(fs) == len(self), \
                    'got {} children and {} fns'.format(len(fs), len(self))
                for child, f, x in zip(self._children, fs, xs):
                    res.append(child.apply(f, x))
                return res
            else:
                # single function to all nodes
                for child, x in zip(self._children, xs):
                    res.append(child.apply(fs, x))
                return res
        return fs(xs)

    @property
    def children(self):
        if self._children:
            return self._children

    def regularize(self, xs):
        # print('model reg', size(xs))
        if self._children:
            return [child.regularize(x) for child, x in zip(self._children, xs)]

    def flatten(self, tensor):
        assert self._check_shape(tensor) is True

    def loss(self, predicted, target):
        loss = 0
        if self._children:
            for child, p, t in zip(self._children, predicted, target):
                loss = loss + child.loss(p, t)
            return loss
        return self._loss_fn(predicted, target)

    def _check_shape(self, xs):
        if isinstance(xs, (list, tuple)):
            if self._children:
                for child, x in zip(self._children, xs):
                    if not child._check_shape(x):
                        return False
        elif torch.is_tensor(xs):
            if self._children:
                return False
            return self._def == list(xs.size())


class CoordVec(ActionReg):
    CanBuildFromInt = 0

    def from_vec(self, vec):
        if torch.is_tensor(vec):
            return torch.tensor(vec)
        elif isinstance(vec, np.ndarray):
            return torch.from_numpy(vec)

    def _loss_fn(self, predicted, target):
        return F.mse_loss(predicted, target)

    def regularize(self, x):
        # print(type(x), len(x))
        # if isinstance(x, list) and len(x) == 1:
        #     x = x[0]
        # print(type(x), size(x))
        return torch.clamp(x, 0, 1)


class OneHot(ActionReg):
    CanBuildFromInt = 1
    """ 
    
    """
    def from_vec(self, vec):
        if isinstance(vec, int) and vec < self._def[0]:
            z = torch.zeros(*self._def).long()
            z[vec] = 1
            return z

    def _loss_fn(self, predicted, target):
        if isinstance(target, (int)):
            return mse_from_tgt_indicies(predicted, torch)
        # elif target.size(-1) == 1:
        return F.mse_loss(predicted, target)

    def regularize(self, x):
        # print('ix size', size(x))
        return nn.Softmax(dim=-1)(x)



