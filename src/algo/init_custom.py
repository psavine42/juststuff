import torch
import torch.nn as nn
import numpy as np


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('LSTM') != -1:
        # weight_shape = list(m.weight.data.size())
        # fan_in = weight_shape[1]
        # fan_out = weight_shape[0]
        # w_bound = np.sqrt(6. / (fan_in + fan_out))
        # m.weight.data.uniform_(-w_bound, w_bound)
        m.bias_ih.data.fill_(0)
        m.bias_hh.data.fill_(0)


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


def init_hidden_cell(in_size, device):
    hx = torch.zeros(1, in_size).to(device)
    cx = torch.zeros(1, in_size).to(device)
    return hx, cx

