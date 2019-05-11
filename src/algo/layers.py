import torch
import torch.nn as nn
from torch.nn import Module, Parameter
from torch import tensor
import torch.nn.functional as F


class MultiBoxLayer(Module):
    """https://github.com/kuangliu/pytorch-ssd/blob/master/multibox_layer.py """
    num_classes = 21
    num_anchors = [4, 6, 6, 6, 4, 4]
    in_planes = [512, 1024, 512, 256, 256, 256]

    def __init__(self):
        super(MultiBoxLayer, self).__init__()

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for i in range(len(self.in_planes)):
            self.loc_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i]*4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i]*21, kernel_size=3, padding=1))

    def forward(self, xs):
        """
        Args:
          xs: (list) of tensor containing intermediate layer outputs.
        Returns:
          loc_preds: (tensor) predicted locations, sized [N,8732,4].
          conf_preds: (tensor) predicted class confidences, sized [N,8732,21].
        """
        y_locs = []
        y_confs = []
        for i, x in enumerate(xs):
            y_loc = self.loc_layers[i](x)
            N = y_loc.size(0)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
            y_loc = y_loc.view(N, -1, 4)
            y_locs.append(y_loc)

            y_conf = self.conf_layers[i](x)
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            y_conf = y_conf.view(N, -1, 21)
            y_confs.append(y_conf)

        loc_preds = torch.cat(y_locs, 1)
        conf_preds = torch.cat(y_confs, 1)
        return loc_preds, conf_preds


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class DotAttention(Module):
    def __init__(self, in_features=512, sigma0=0.5):
        Module.__init__(self)
        self.weight_key = Parameter(in_features,  )
        self.weight_query = Parameter(in_features, )
        self.weight_value = Parameter(in_features, )

    def forward(self, x, k, v):
        wq = torch.matmul(self.weight_query, x)
        wk = torch.matmul(self.weight_query, k)
        wv = torch.matmul(self.weight_query, v)

        scaled = torch.matmul(wq, wk.t()) / torch.sqrt(k.size(0))
        attn_out = torch.tensordot(torch.softmax(scaled, -1),  wv)


class ConvSelfAttention(Module):
    """ https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html """
    def __init__(self):
        Module.__init__(self)
        self.key = nn.Conv2d()
        self.value = nn.Conv2d()
        self.query = nn.Conv2d()
        self.softmx = nn.Softmax()

    def forward(self, x):
        """
        α_i,j = softmax(f(xi).t(), g(xj))

        o_j= ∑ αi,jh(xi)

        """
        kx = self.key(x)
        vx = self.value(x)
        qx = self.query(x)

        attention_map = self.softmx(torch.mm(kx.t(), qx))
        out = torch.mm(attention_map, vx)
        return out


class NoisyLinear(Module):
    """Factorised Gaussian NoisyNet"""

    def __init__(self, in_features, out_features, sigma0=0.5):
        Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(tensor(out_features, in_features))
        self.bias = nn.Parameter(tensor(out_features))
        self.noisy_weight = nn.Parameter(tensor(out_features, in_features))
        self.noisy_bias = nn.Parameter(tensor(out_features))
        self.noise_std = sigma0 / (self.in_features ** 0.5)

        self.reset_parameters()
        self.register_noise()

    def register_noise(self):
        in_noise = torch.FloatTensor(self.in_features)
        out_noise = torch.FloatTensor(self.out_features)
        noise = torch.FloatTensor(self.out_features, self.in_features)
        self.register_buffer('in_noise', in_noise)
        self.register_buffer('out_noise', out_noise)
        self.register_buffer('noise', noise)

    def sample_noise(self):
        self.in_noise.normal_(0, self.noise_std)
        self.out_noise.normal_(0, self.noise_std)
        self.noise = torch.mm(self.out_noise.view(-1, 1), self.in_noise.view(1, -1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.noisy_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Note: noise will be updated if x is not volatile
        """
        normal_y = F.linear(x, self.weight, self.bias)
        if self.training:
            # update the noise once per update
            self.sample_noise()

        noisy_weight = self.noisy_weight * self.noise
        noisy_bias = self.noisy_bias * self.out_noise
        noisy_y = F.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) + ', out_features=' + str(self.out_features) + ')'
