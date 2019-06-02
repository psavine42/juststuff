import math
import torch
import torch.nn as nn
from torch.nn import Module, Parameter
from torch import tensor
import torch.nn.functional as F
from src.actions.action_models import *
from src.probablistic.utils import flatten


class Noop(Module):
    def __init__(self):
        Module.__init__(self)

    def forward(self, input):
        return input


class CnvStack(Module):
    def __init__(self, in_size=2, batch_norm=False):
        Module.__init__(self)
        self.batch_norm = batch_norm
        self.in_channels = in_size
        self.conv1 = nn.Conv2d(in_size, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(16)

    @property
    def conv_mods(self):
        return [self.conv1, self.conv2, self.conv3]

    def out_size(self, w, h):
        z = self(torch.zeros(1, self.in_channels, w, h))
        return z.size(1) * z.size(2) * z.size(3)

    def forward(self, x):
        if self.batch_norm is True:
            x = F.leaky_relu(self.bn1(self.conv1(x)))
            x = F.leaky_relu(self.bn2(self.conv2(x)))
            x = F.leaky_relu(self.bn3(self.conv3(x)))
        else:
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
        return x


class ConvNormRelu(Module):
    def __init__(self, in_channels=2, out_channels=16, k=5, s=1,
                 p=0, batch_norm=True):
        Module.__init__(self)
        self.batch_norm = batch_norm
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=k, stride=s, padding=p)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def out_size(self, w, h, ndim=1):
        z = self(torch.zeros(1, self.in_channels, w, h))
        if ndim == 1:
            return z.size(1) * z.size(2) * z.size(3)
        else:
            return [z.size(2), z.size(3)]

    def forward(self, x):
        if self.batch_norm is True:
            x = F.leaky_relu(self.bn1(self.conv1(x)))
        else:
            x = F.leaky_relu(self.conv1(x))
        return x


class MLP2(Module):
    def __init__(self, in_size, out_size, activation=nn.ReLU):
        Module.__init__(self)
        self.l = nn.Sequential(nn.Linear(in_size, (in_size + out_size) // 2),
                               activation(),
                                nn.Linear((in_size + out_size) // 2, out_size),
                                # activation()
                               )

    def forward(self, x):
        return self.l(x)


class DeConvNormRelu(Module):
    def __init__(self, in_channels=2, out_channels=16, k=5, s=1,
                 p=0, u=1, d=False, batch_norm=True,
                 activation=None):
        Module.__init__(self)
        self.batch_norm = batch_norm
        self.in_channels = in_channels
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, k, stride=s,
                                        padding=p)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = activation if activation else F.leaky_relu
        self.u = True if u > 1 else False
        if u > 1:
            self.up_sample = nn.Upsample(scale_factor=u, mode='bilinear')

    def out_size(self, w, h, ndim=1):
        z = self(torch.zeros(1, self.in_channels, w, h))
        if ndim == 1:
            return z.size(1) * z.size(2) * z.size(3)
        else:
            return [z.size(2), z.size(3)]

    def forward(self, x):
        if self.batch_norm is True:
            x = self.bn1(self.conv1(x))
        else:
            x = self.conv1(x)
        # if self.activation is not None:
        x = self.activation(x)
        if self.u:
            return self.up_sample(x)
        return x


class EncodeState4(Module):
    def __init__(self, in_channels, zdim, residual=False):
        Module.__init__(self)
        _ly = [dict(k=6, s=1),
               dict(k=6, s=1),
               dict(k=4, s=2),
               dict(k=4, s=1)]
        self.residual = residual
        self.conv1 = ConvNormRelu(in_channels, 8, **_ly[0])
        self.conv2 = ConvNormRelu(8, 16, **_ly[1])
        self.conv3 = ConvNormRelu(16, 32, **_ly[2])
        self.conv4 = ConvNormRelu(32, zdim, **_ly[3], batch_norm=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return x4


class DecodeState4(Module):
    def __init__(self, in_channels, zdim, residual=False):
        Module.__init__(self)
        _ly = [dict(k=6, s=1),
               dict(k=6, s=1),
               dict(k=4, s=2),
               dict(k=4, s=1)]
        self.dec1 = DeConvNormRelu(zdim, 32, **_ly[3])
        self.dec2 = DeConvNormRelu(32, 16,   **_ly[2])
        self.dec3 = DeConvNormRelu(16, 8,    **_ly[1])
        self.dec4 = DeConvNormRelu(8, in_channels, **_ly[0])

    def forward(self, z, hidden=None):
        """
        todo - make this real
        img: size [batch_size, c, h, w]
        """
        x = self.dec1(z.unsqueeze(-1).unsqueeze(-1))
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        return x


# ----------------------------------------------------------
class _PolicyDecoderBase(Module):
    def __init__(self, zdim,
                 shape=[3, 20, 20],
                 action_fn=None,
                 geomtry_fn=None):
        Module.__init__(self)
        self._zdim = zdim
        self._state_shape = shape
        self.sigmoid = geomtry_fn if geomtry_fn else nn.Sigmoid()
        self.softmax = action_fn if action_fn else nn.Softmax(dim=-1)

    def loss(self, *args):
        raise NotImplemented('not implemented in base class ')

    def predict_box(self, *args):
        """ return [b , 5 ] in (0, 1) """
        raise NotImplemented('not implemented in base class ')


class PolicySimple(_PolicyDecoderBase):
    """
    DONE

    """
    def __init__(self, zdim, **kwargs):
        _PolicyDecoderBase.__init__(self, zdim, **kwargs)
        self.action = MLP2(zdim, 5)

    def predict_box(self, x):
        return x

    def loss(self, predicted, targets):
        return F.mse_loss(predicted, targets)

    def forward(self, x):
        return self.sigmoid(self.action(x))


# ----------------------------------------------------------
# LOGITS :[N, S] , GEOM: [ N, 4 ] - testing fully continuous policies
class PolicyDiscContIndependant(_PolicyDecoderBase):
    """
    todo DONE
    Assumes Independence
    ----
    action -> size(shape[0]) logits-num_spaces
    action -> size(4)        continuous
    """
    def __init__(self, zdim, **kwargs):
        _PolicyDecoderBase.__init__(self, zdim, **kwargs)
        self.action = MLP2(zdim, kwargs['shape'][0])
        self.geom = MLP2(zdim, 4)

    def predict_box(self, x):
        return composite_action_to_cont_box(x)

    def loss(self, predicted, targets):
        return disc_cont_loss(predicted, targets)

    def forward(self, z):

        logits = self.softmax(self.action(z))
        geom = self.sigmoid(self.geom(z))
        return logits, geom


class PolicyDiscContGA(_PolicyDecoderBase):
    """
    todo DONE
    Assumes 'ActionIndex' Depends on 'Geometry'

    """
    def __init__(self, zdim, **kwargs):
        _PolicyDecoderBase.__init__(self, zdim, **kwargs)
        self.action = MLP2(zdim + 4, kwargs['shape'][0])
        self.geom = MLP2(zdim, 4)

    def predict_box(self, x):
        return composite_action_to_cont_box(x)

    def loss(self, predicted, targets):
        return disc_cont_loss(predicted, targets)

    def forward(self, z):
        geom = self.geom(z)
        logits = self.action(torch.cat((z, geom), -1))
        return self.softmax(logits), self.sigmoid(geom)


class PolicyDiscContAG(_PolicyDecoderBase):
    """
    todo DONE
    Assumes 'ActionIndex' Depends on 'Geometry'
    ----
    action -> size(shape[0]) logits-num_spaces
    action -> size(4)        continuous

    """
    def __init__(self, zdim, **kwargs):
        _PolicyDecoderBase.__init__(self, zdim, **kwargs)
        self.action = MLP2(zdim, kwargs['shape'][0])
        self.geom = MLP2(zdim + kwargs['shape'][0], 4)

    def predict_box(self, x):
        return composite_action_to_cont_box(x)

    def loss(self, predicted, targets):
        return disc_cont_loss(predicted, targets)

    @property
    def out_size(self):
        return [self._state_shape[0], 4]

    def forward(self, z):
        logits = self.action(z)
        geom = self.geom(torch.cat((z, logits), -1))
        return self.softmax(logits), self.sigmoid(geom)


# ----------------------------------------------------------
#
class PolicyLogits4CoordIndep(_PolicyDecoderBase):
    """
    TODO THIS SHOULD BE THE CONF PRED MODELLLLL
        location_predictions   [N, 4]
    confidence_predictions [N, num_]

    """
    def __init__(self, zdim, **kwargs):
        _PolicyDecoderBase.__init__(self, zdim, **kwargs)
        self.action = MLP2(zdim, kwargs['shape'][0])
        self.geom = MLP2(zdim, 4)

    def loss(self, predicted, targets):
        return disc_cont_loss(predicted, targets)

    def forward(self, z):
        """
        Returns
            [b, S, 1], [b, N, 4 ]
        """
        y = self.softmax(self.action(z))
        action_box = self.sigmoid(self.action(z))
        return y, action_box


class PolicyAllLogitsIndependent(_PolicyDecoderBase):
    """

    TODO - losses

    action -> logits-num_spaces
    action -> logits-xdim
    action -> logits-ydim
    action -> logits-xdim
    action -> logits-ydim
    """

    def __init__(self, zdim, shape=[3, 20, 20], **kwargs):
        _PolicyDecoderBase.__init__(self, zdim, shape=shape, **kwargs)
        self.action = MLP2(zdim, shape[0])
        self.x0 = MLP2(zdim, shape[1])
        self.y0 = MLP2(zdim, shape[2])
        self.x1 = MLP2(zdim, shape[1])
        self.y1 = MLP2(zdim, shape[2])

    def loss(self, pred, targets):
        return disc_disc_loss(pred, targets)

    def predict_box(self, x):
        return disc_disc_action_to_cont_box(x)

    def forward(self, z):
        """ tuple of [N, actions ] ,
            [ b, 1, S ], [ b, 4, N ]
        """
        ac = self.softmax(self.action(z))
        x0 = self.softmax(self.x0(z))
        y0 = self.softmax(self.y0(z))
        x1 = self.softmax(self.x1(z))
        y1 = self.softmax(self.y1(z))
        return ac, torch.stack((x0, y0, x1, y1), 1)


class PolicyAllLogitsAG(_PolicyDecoderBase):
    """

    TODO - losses

    action -> logits-num_spaces
    action -> logits-xdim
    action -> logits-ydim
    action -> logits-xdim
    action -> logits-ydim
    """

    def __init__(self, zdim, shape=[3, 20, 20], **kwargs):
        _PolicyDecoderBase.__init__(self, zdim, shape=shape, **kwargs)
        self.action = MLP2(zdim, shape[0])
        self.x0 = MLP2(zdim + shape[0], shape[1])
        self.y0 = MLP2(zdim + shape[0], shape[1])
        self.x1 = MLP2(zdim + shape[0], shape[1])
        self.y1 = MLP2(zdim + shape[0], shape[1])

    def loss(self, pred, targets):
        return disc_disc_loss(pred, targets)

    def predict_box(self, x):
        return disc_disc_action_to_cont_box(x)

    @property
    def out_size(self):
        return [self._state_shape[0], [self._state_shape[1], 4]]

    def forward(self, z):
        """ tuple of [N, actions ] ,
            [ b, 1, S ], [ b, 4, N ]
        """
        ac = self.action(z)
        x0 = self.x0(torch.cat((z, ac), -1))
        y0 = self.y0(torch.cat((z, ac), -1))
        x1 = self.x1(torch.cat((z, ac), -1))
        y1 = self.y1(torch.cat((z, ac), -1))
        return self.softmax(ac), torch.stack((x0, y0, x1, y1), 1)


class PolicyAllLogitsRNN(_PolicyDecoderBase):
    """

    """
    def __init__(self, zdim, num_layer=1, **kwargs):
        _PolicyDecoderBase.__init__(self, zdim, **kwargs)
        shape = kwargs['shape']
        self.hs = shape[1]
        self.num_layer = num_layer
        self.action = MLP2(zdim, shape[0])
        self.geom = nn.RNN(zdim, shape[1], num_layer)

    def forward(self, z):

        nz = z.expand(4, z.size(0), z.size(-1))
        hx = torch.zeros(self.num_layer, z.size(0), self.hs)

        coord, hs = self.geom(nz, hx)

        ac = self.action(z )
        return


# ---------------------------------------------------------------------------------------
# OTHER LAYERS - EXTERNALS
# ---------------------------------------------------------------------------------------
class MultiBoxLayer(Module):
    """https://github.com/kuangliu/pytorch-ssd/blob/master/multibox_layer.py """
    # num_classes = 21
    num_anchors = [4, 6, 6, 6, 4, 4]
    in_planes = [512, 1024, 512, 256, 256, 256]

    def __init__(self, num_class):
        super(MultiBoxLayer, self).__init__()
        self.num_classes = num_class

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
    def forward(self, x):
        return flatten(x)


def squash(input, dims):
    for d in dims:
        input.squeeze_(d)
    return input


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
    def __init__(self, in_channels):
        Module.__init__(self)
        self.key = nn.Conv2d(in_channels, in_channels, stride=1, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, stride=1, kernel_size=1)
        self.query = nn.Conv2d(in_channels, in_channels, stride=1, kernel_size=1)
        self.softmx = nn.Softmax()

    def forward(self, x):
        """
        α_(i,j) = softmax(f(xi).t(), g(xj))

        o_j= ∑ α_(i,j) * h(xi)

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


def batchnorm(in_planes):
    "batch norm 2d"
    return nn.BatchNorm2d(in_planes, affine=True, eps=1e-5, momentum=0.1)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


def convbnrelu(in_planes, out_planes, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups,
                      bias=False),
            batchnorm(out_planes),
            nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups,
                      bias=False),
            batchnorm(out_planes))


class ResidualEncBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        Module.__init__(self)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        y = out + residual
        return self.relu(y)


class ResidualDecBlock(Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        Module.__init__(self)
        self.downsample = downsample
        self.stride = stride

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):

        return


class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv3x3(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x


stages_suffixes = {0: '_conv',
                   1: '_conv_relu_varout_dimred'}


class RCUBlock(Module):
    def __init__(self, in_planes, out_planes, n_blocks, n_stages):
        super(RCUBlock, self).__init__()
        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}{}'.format(i + 1, stages_suffixes[j]),
                        conv3x3(in_planes if (i == 0) and (j == 0) else out_planes,
                                out_planes, stride=1,
                                bias=(j == 0)))
        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = F.relu(x)
                x = getattr(self, '{}{}'.format(i + 1, stages_suffixes[j]))(x)
            x += residual
        return x








