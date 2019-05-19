import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import random
from collections import namedtuple
import numpy as np
from torch.nn.parameter import Parameter
from torch import tensor
from torch.nn import Module

from src.algo.init_custom import *
from src.probablistic.funcs import *
from src.algo.layers import *


def conv2d_size_out(in_size, mod, hw=0):
    k = mod.kernel_size[hw]
    s = mod.stride[hw]
    d = mod.dilation[hw]
    p = mod.padding[hw]
    return (in_size + 2 * p - d * (k - 1) - 1) // s + 1


def convs2d_size_out(size, convs, hw=0):
    res = size
    for conv in convs:
        res = conv2d_size_out(res, conv, hw=hw)
    return res


def out_size(modules, dims, all=False, d=False):
    dy = [int(_) for _ in dims]
    if d:
        print(dy)
    alls = [dy ]
    for m in modules:
        z = m(torch.zeros(1, *dy))
        dy = list(z.size())[1:]
        if d:
            print(dy)
        alls.append(dy)
    if all is True:
        return alls
    return dy


_map = {2: [[2, 16, 3, 2], [16, 32, 3, 2], [32, 32, 3, 1]],
        3: [[3, 16, 5, 2], [16, 32, 3, 1], [32, 32, 3, 1]],
        4: [[3, 16, 5, 2], [16, 32, 3, 1], [32, 32, 3, 1]]
        }


# Components -------------------------------------------
class FeatureNet(nn.Module):
    """ """
    def __init__(self, in_size, output_size):
        super(FeatureNet, self).__init__()
        self.linear1 = nn.Linear(in_size, 2 * in_size)
        self.linear2 = nn.Linear(2 * in_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x


class CnvController(Module):
    def __init__(self,
                 state_shape,
                 out_dim=256,
                 batch_norm=False):
        Module.__init__(self)
        self.conv_stack = CnvStack(state_shape[0], batch_norm=batch_norm)
        cw, ch = self.conv_stack.out_size(state_shape[1], state_shape[2])
        self.lstm = nn.LSTMCell(ch * cw, out_dim)
        self.feature_dim = out_dim

    def forward(self, x):
        x = self.conv_stack(x)
        x = x.view(x.size(0), -1)
        return self.lstm(x)


class CnvController2h(Module):
    def __init__(self,
                 state_shape,
                 hints_shape=2,
                 out_dim=256,
                 batch_norm=False):
        Module.__init__(self)
        self.conv_stack = CnvStack(state_shape[0])
        size_out = self.conv_stack.out_size(state_shape[1], state_shape[2])
        print(state_shape, size_out)
        self.features1 = nn.Linear(hints_shape, hints_shape * 2)
        self.features2 = nn.Linear(hints_shape * 2, hints_shape)

        self.lstm_in_size = size_out + hints_shape
        self.lstm = nn.LSTMCell(self.lstm_in_size, out_dim)
        self.feature_dim = out_dim
        self.apply(weights_init)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = init_hidden_cell(self.feature_dim, next(self.parameters()).device)
        s, hints = x
        # print(s.size(), hints.size())
        s = self.conv_stack(s)
        s = s.view(s.size(0), -1)
        h = F.tanh(self.features1(hints.view(hints.size(0), -1)))
        h = F.tanh(self.features2(h))
        v = torch.cat((h, s), -1)
        # print(v.size())
        return self.lstm(v, hidden)


class CnvController3h(Module):
    """ 3 headed input """
    def __init__(self,
                 state_shape,
                 hints_shape=2,
                 out_dim=256,
                 batch_norm=False):
        Module.__init__(self)
        self.conv_stack = CnvStack(state_shape[0])
        size_out = self.conv_stack.out_size(state_shape[1], state_shape[2])
        print(state_shape, size_out)
        self.features_cur = nn.Linear(hints_shape, hints_shape // 2)
        self.features_tgt = nn.Linear(hints_shape, hints_shape // 2)

        self.featuresf11 = nn.Linear(hints_shape, hints_shape // 2)
        self.featuresf12 = nn.Linear(hints_shape // 2, state_shape[0])

        self.lstm_in_size = size_out + state_shape[0]
        self.lstm = nn.LSTMCell(self.lstm_in_size, out_dim)
        self.feature_dim = out_dim
        self.apply(weights_init)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = init_hidden_cell(self.feature_dim, next(self.parameters()).device)
        s, hints, targets = x
        # print(s.size(), hints.size())
        s = self.conv_stack(s)
        s = s.view(s.size(0), -1)
        h1 = F.leaky_relu(self.features_cur(hints.view(hints.size(0), -1)))
        h2 = F.leaky_relu(self.features_tgt(hints.view(targets.size(0), -1)))
        h = F.leaky_relu(self.featuresf11(torch.cat((h1, h2), -1)))
        h = F.leaky_relu(self.featuresf12(h))
        v = torch.cat((h, s), -1)
        # print(v.size())
        return self.lstm(v, hidden)


# MODELS -------------------------------------------------------

class DQNC(Module):
    def __init__(self, h, w, outputs, in_size=2, feats_size=40):
        Module.__init__(self)
        # print(in_size)
        self.conv_stack = CnvStack(in_size)
        convw = convs2d_size_out(w, self.conv_stack.conv_mods, hw=0)
        convh = convs2d_size_out(h, self.conv_stack.conv_mods, hw=1)

        linear_input_size = convw * convh * 16 + feats_size
        # print('linear in size ', linear_input_size, convw, outputs)
        self.head1 = nn.Linear(linear_input_size, linear_input_size // 2)
        self.head2 = nn.Linear(linear_input_size // 2, outputs)
        self.features = FeatureNet(11, feats_size)

    def forward(self, inputs):
        """Called with either one element to determine next action, or a batch
            during optimization. Returns tensor([[left0exp,right0exp]...])."""
        x, feats = inputs
        x = self.conv_stack(x)

        x = x.view(x.size(0), -1)
        y = self.features(feats.view(-1).unsqueeze(0))
        xs = torch.cat((x, y), -1)
        xs = F.relu(self.head1(xs))
        xs = self.head2(xs)
        return F.softmax(xs, dim=1)


class LSTMDQN(nn.Module):
    def __init__(self, h, w, outputs, in_size=2, feats_in=11, feats_size=40, value=True):
        super(LSTMDQN, self).__init__()
        self.conv_stack = CnvStack(in_size)
        convw = convs2d_size_out(w, self.conv_stack.conv_mods, hw=0)
        convh = convs2d_size_out(h, self.conv_stack.conv_mods, hw=1)

        lstm_in_size = convw * convh * 16 + feats_size
        self.lstm_in = lstm_in_size
        # print('linear in size ', lstm_in_size, convw, outputs)
        self.features = FeatureNet(feats_in, feats_size)

        self.lstm = nn.LSTMCell(lstm_in_size, lstm_in_size)
        self.actor = nn.Linear(lstm_in_size, outputs)
        self._value = value
        if value is True:
            self.critic = nn.Linear(lstm_in_size, 1)

    def forward(self, inputs):
        """ Mashup of Models """
        x, feats, (hx, cx) = inputs
        x = self.conv_stack(x)
        x = x.view(x.size(0), -1)

        y = self.features(feats.view(-1).unsqueeze(0))
        x = torch.cat((x, y), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        if self._value:
            return self.critic(x), self.actor(x), (hx, cx)
        return self.actor(x), (hx, cx)


class DiscProbTransform(nn.Module):
    def __init__(self):
        super(DiscProbTransform, self).__init__()
        self.xform = nn.Linear()

    def forward(self, disc_probs):
        """
        given a tensor of size [(b), S, N, M] where sum([:, n, m]) = 1 aka a probablity dist.

        learn a transformation which will move those about so that max(X) at each i,j resolves
        constraints.

        What are we really doing? perturbing a probablility field

        the probabilities are all dependent

        1) A map which operates on each point and adjusts the probability distribution
        2) Something needs to tell the map roughly what to do.
            - Attention module - parametrize by f(X) -> mu_x, sigma_x, mu_y, sigma_y

        ------
        1) a function which outputs [S, 4] representing mu_x, sigma_x, mu_y, sigma_y of each S
        2) this is used to parametrize the final field on which agent shall act ????
        3)

        """
        x = self.xform(disc_probs)
        # is this a flow field ?? lets say S = 3
        # eg: x[:, 0, 0] -> [0.1 , 0.2, 0.7]
        # this is a unit vector. What does that mean?
        # this means ..
        return



class OptionCriticNet(nn.Module):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        # self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(x)
        q = self.fc_q(phi)
        beta = F.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        pi = F.softmax(pi, dim=-1)
        return {'q': q,
                'beta': beta,
                'log_pi': log_pi,
                'pi': pi}


class Optimization(Module):
    def __init__(self, action_dim):
        Module.__init__(self)

        self.X = nn.Parameter(torch.zeros(action_dim))
        self.B = nn.Parameter(torch.zeros(action_dim))

    def forward(self, Q):
        f = torch.mm(self.X.t(), torch.matmul(Q, self.X)) + self.B


class PredictionEncoded(Module):
    """ if the encoder takes the lstm common layer as the inu"""
    def __init__(self, input_shape, z_size):
        Module.__init__(self)
        self._state_action_encoder = nn.Linear(input_shape, z_size)
        self._state_encoder = None

    def forward(self, state, ):
        """
        Encoder(s, a)

        Encoder(s)

        """

        return


class GaussianActorCriticNet(Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 granular=False):
        Module.__init__(self)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.phi_body = phi_body
        feature_dim = self.phi_body.feature_dim

        self.actor_body = actor_body
        self.fc_action_cat = nn.Linear(feature_dim, state_dim[0])
        self.fc_action_loc = nn.Linear(feature_dim, action_dim)

        self.critic_body = critic_body
        self.fc_critic = layer_init(nn.Linear(feature_dim, 1), 1e-3)

        # auxilary value for predicting Himts
        self.fc_auxilary = layer_init(nn.Linear(feature_dim, 1), 1e-3)

        self.std = nn.Parameter(torch.zeros(action_dim))
        # todo https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/
        self.predictor_module = None
        self.granular = granular
        self.__init_weights()

    def __init_weights(self):
        self.critic_body.weight.data = normalized_columns_initializer(self.critic_body.weight.data, 1.0)
        self.fc_critic.weight.data = normalized_columns_initializer(self.fc_critic.weight.data, 1.0)

        self.actor_body.weight.data = normalized_columns_initializer(self.actor_body.weight.data,  0.01)
        self.fc_action_cat.weight.data = normalized_columns_initializer(self.fc_action_cat.weight.data, 0.01)
        self.fc_action_loc.weight.data = normalized_columns_initializer(self.fc_action_loc.weight.data, 0.01)

        self.actor_body.bias.data.fill_(0)
        self.critic_body.bias.data.fill_(0)
        self.fc_action_cat.bias.data.fill_(0)
        self.fc_action_loc.bias.data.fill_(0)
        self.fc_critic.bias.data.fill_(0)

    def forward(self, obs, hidden, action=None):
        # Controller
        hx, cx = self.phi_body(obs, hidden)     # conv+lstm
        phi = hx

        # bodies
        phi_a = self.actor_body(phi)
        value = self.critic_body(phi)

        # heads -----------------------------
        # value [n, 1]
        value = self.fc_critic(value)

        # discrete prediction [n, F ]
        # act = self.fc_action_cat(phi_a)

        # discrete prediction [n, S ]
        act = self.fc_action_cat(phi_a)

        # continuous prediction [n, 4 ]
        loc = self.fc_action_loc(phi_a)

        means = torch.tanh(loc)
        # print(means, F.softplus(self.std))
        pred_act = sample_normal(means, scale=F.softplus(self.std))
        # print(act )
        pred_opt = sample_categorical(F.softmax(act, dim=-1) )
        # print('ac sizes', pred_act['log_prob'].size(), pred_opt['log_prob'].size() )
        res = {'action_index': pred_opt['action'],
               'action': pred_act['action'],
               'hidden': (hx, cx),
               'mean': means,
               'value': value,
               'logits': act}

        if self.granular:
            res['log_pi_a_cont'] = pred_act['log_prob']
            res['log_pi_a_disc'] = pred_opt['log_prob']
            res['log_pi_a_cont'] = pred_act['entropy']
            res['log_pi_a_disc'] = pred_opt['entropy']
        else:
            # todo this is one way to hack the discrete problem
            alp = pred_act['log_prob']
            res['log_prob'] = torch.mean(torch.cat((alp.view(1, alp.size(1)), pred_opt['log_prob']), -1))
            ep = pred_act['entropy']
            res['entropy'] = torch.mean(torch.cat((ep.view(1, ep.size(1)), pred_opt['entropy']), -1))
        return res


class StagedActorCriticNet(Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 granular=False):
        Module.__init__(self)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.phi_body = phi_body
        feature_dim = self.phi_body.feature_dim

        self.actor_body = actor_body
        self.fc_action_cat = nn.Linear(feature_dim, state_dim[0])
        self.fc_action_loc = nn.Linear(feature_dim, action_dim)

        self.critic_body = critic_body
        self.fc_critic = layer_init(nn.Linear(feature_dim, 1), 1e-3)

        # auxilary value for predicting Himts
        self.fc_auxilary = layer_init(nn.Linear(feature_dim, 1), 1e-3)

        self.std = nn.Parameter(torch.zeros(action_dim))
        self.predictor_module = None
        self.granular = granular
        self.apply(weights_init)

    def forward(self, obs, hidden, action=None):
        """
        1) encode state
        - predict parameters writing DRAW grid
        2) predict boxes [N, M, N, M] ~ 4N
        3) assign (boxes, z)
        """
        # Controller
        hx, cx = self.phi_body(obs, hidden)     # conv+lstm
        phi = hx

        # bodies
        phi_a = self.actor_body(phi)
        value = self.critic_body(phi)

        # heads -----------------------------
        # value [n, 1]
        value = self.fc_critic(value)

        # discrete prediction [n, F ]
        # act = self.fc_action_cat(phi_a)

        # continuous prediction [n, 4, M]
        loc = self.fc_action_loc(phi_a)

        # means = torch.tanh(loc)
        # print(means, F.softplus(self.std))
        # pred_act = sample_normal(means, scale=F.softplus(self.std))
        # print(act )
        pred_geom = sample_categorical(F.softmax(loc, dim=-1) )

        # predict next_state

        # print('ac sizes', pred_act['log_prob'].size(), pred_opt['log_prob'].size() )
        res = {'action_index': pred_opt['action'],
               'action': pred_geom['action'],
               'hidden': (hx, cx),

               'value': value,
               'logits': act}

        return res


class StreamNet(Module):
    """
    Attention + Auxilary Streams
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 img_size=[20, 20],
                 num_spaces=3,
                 aux_code_size=4,
                 critic_body=None,
                 granular=False):
        Module.__init__(self)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.phi_body = CnvStack(img_size)
        self.attention = ConvSelfAttention()

        feature_dim = self.phi_body.feature_dim

        self.extra_layers = nn.Sequential(
            Flatten(),
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU()
        )
        # self.actor_body = actor_body
        # self.fc_action_cat = nn.Linear(feature_dim, state_dim[0])
        # self.fc_action_loc = nn.Linear(feature_dim, 2)

        self.critic_body = critic_body
        self.fc_critic = layer_init(nn.Linear(feature_dim, 1), 1e-3)

        # auxilary value for predicting Hints
        self.fc_auxilary = nn.ModuleList([nn.Linear(feature_dim, num_spaces)
                                          for x in range(aux_code_size)])
        self.fc_auxilary_attn = nn.Linear(feature_dim, num_spaces * aux_code_size)

        self.sigmoid = nn.Sigmoid()
        self.predictor_module = None
        self.granular = granular
        self.apply(weights_init)

    def forward(self, obs, hidden=None):
        """
        Auxilary task streams

            Predict adj_matrix
            Predict constraint-values
            Value (standard)

        :return:
        """
        img, target_code = obs
        # Controller
        x = self.phi_body(img)
        attn = self.attention(x)
        phi_att = 0.5 * attn + x
        phi = self.extra_layers(phi_att)

        # heads -----------------------------
        # value [n, 1]
        value = self.fc_critic(phi)

        # aux [n, F, S]
        aux_value = torch.stack(tuple([l(phi) for l in self.fc_auxilary]))
        aux_attn = self.fc_auxilary_attn(aux_value)

        # global attention - look at aux_values and
        phi_feats = torch.cat((aux_attn, phi))
        xf = self.global_attn(phi_feats)

        # these describe a window of the input
        sigmas = self.fc_sigma(xf)
        action = self.fc_action(xf)

        logits = self.fc_choice(xf)
        pred_act = sample_categorical(F.softmax(logits, dim=-1))
        return {'action_index': pred_act['action'],
                'action': action,
                'sigmas': sigmas,
                'log_prob': pred_act['log_prob'],
                'entropy': pred_act['entropy'],
                'aux': aux_value,
                'value': value,
                'logits': logits
                }


class StreamNetFull(Module):
    """
    Attention + Auxilary Streams
    """
    def __init__(self,
                 state_dim=[4, 20, 20],
                 zdim=100,
                 num_spaces=3,
                 aux_code_size=4,
                 debug=False):
        Module.__init__(self)
        self.state_dim = state_dim
        self.zdim = zdim
        _ly = [dict(k=6, s=1),
               dict(k=6, s=1),
               dict(k=4, s=2),
               dict(k=4, s=1)]

        self.conv1 = ConvNormRelu(state_dim[0], 8, **_ly[0])
        self.conv2 = ConvNormRelu(8, 16,    **_ly[1])
        self.conv3 = ConvNormRelu(16, 32,   **_ly[2])
        self.conv4 = ConvNormRelu(32, zdim, **_ly[3])

        dims1 = out_size([self.conv1, self.conv2, self.conv3, self.conv4],
                         state_dim, all=True, d=debug)
        self._inner_dim = dims1[-1]

        self.encode = nn.Sequential(
            Flatten(),
            nn.Linear(dims1[-1][0] * dims1[-1][1] * dims1[-1][2], zdim * 2),
            nn.LeakyReLU(),
            nn.Linear(zdim * 2, zdim),
            nn.LeakyReLU()
        )

        self.encode_targets = nn.Sequential(
            Flatten(),
            nn.Linear(aux_code_size * num_spaces,
                      min(zdim, aux_code_size * num_spaces) * 2),
            nn.LeakyReLU(),
            nn.Linear(min(zdim, aux_code_size * num_spaces) * 2, zdim),
            nn.LeakyReLU()
        )
        self.merge = nn.Bilinear(zdim, zdim, zdim)

        self.dec1 = DeConvNormRelu(zdim *2, 32, **_ly[3])
        self.dec2 = DeConvNormRelu(32*2, 16,   **_ly[2])
        self.dec3 = DeConvNormRelu(16*2, 8,    **_ly[1])
        self.dec4 = DeConvNormRelu(8*2, num_spaces, **_ly[0])

        self.fc_critic = nn.Linear(zdim, 1)
        self.fc_auxilary = nn.ModuleList([nn.Linear(zdim, num_spaces)
                                          for _ in range(aux_code_size)])
        self.softmax2 = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()
        self.predictor_module = None
        self.apply(weights_init)

    def forward(self, obs, hidden=None):
        """
        todo - make this real
        img: size [batch_size, c, h, w]
        """
        img, target_code = obs
        res = []

        # Stack of convs with features cache
        x = self.conv1(img)
        res.append(x)
        x = self.conv2(x)
        res.append(x)
        x = self.conv3(x)
        res.append(x)
        x = self.conv4(x)
        res.append(x)

        # encode for target and z
        tgt = self.encode_targets(target_code)

        z = self.merge(flatten(x), tgt)
        # z = z.view(z.size(0), )
        # --or-- Bilinear layyer ???
        # z = self.encode(x, tgt)

        # value for prediciont module
        value = self.fc_critic(z)
        auxs = torch.sigmoid(torch.stack([m(z) for m in self.fc_auxilary]).permute(1, 2, 0))

        # decode into an image of [b, c, h, w]
        # at each step there is a skip connection to diluted input
        x = self.dec1(torch.cat((z.unsqueeze(-1).unsqueeze(-1), res[-1]), 1))
        x = self.dec2(torch.cat((x, res[-2]), 1))
        x = self.dec3(torch.cat((x, res[-3]), 1))
        x = self.dec4(torch.cat((x, res[-4]), 1))
        # x = self.dec5(torch.cat((tgt, x, res[2])))

        # ?? final_state = softmax(probs + x)
        # ?? or --       = softmax(self.softmax2(x) + x)
        # so that i am just doing a giant residual step ????
        probs = self.softmax2(x)

        maxs = probs.max(dim=1)[0]
        kl_divergence = (probs * torch.log(probs / maxs)).sum(-1).mean(dim=(1, 2), keepdim=True).squeeze(-1)
        entropy = -(probs * torch.log(probs)).sum(-1).mean(dim=(1, 2), keepdim=True).squeeze(-1)

        return {'log_prob': kl_divergence,  # to maximize
                'entropy': entropy,         # to minimize
                'action': probs,
                'hidden': None,
                'value': value,
                'aux': auxs,
                'z': z,
                }


class StreamNet2(Module):
    """
    Attention + Auxilary Streams
    """

    def __init__(self,
                 state_dim=[4, 20, 20],
                 zdim=100,
                 num_spaces=3,
                 aux_code_size=4,
                 debug=False):
        Module.__init__(self)
        self.state_dim = state_dim
        self.zdim = zdim
        _ly = [dict(k=6, s=1),
               dict(k=6, s=1),
               dict(k=4, s=2),
               dict(k=4, s=1)]

        self.conv1 = ConvNormRelu(state_dim[0], 8, **_ly[0])
        self.conv2 = ConvNormRelu(8, 16, **_ly[1])
        self.conv3 = ConvNormRelu(16, 32, **_ly[2])
        self.conv4 = ConvNormRelu(32, zdim, **_ly[3], batch_norm=False)

        dims1 = out_size([self.conv1, self.conv2, self.conv3, self.conv4],
                         state_dim, all=True, d=debug)
        self._inner_dim = dims1[-1]

        self.encode = nn.Sequential(
            Flatten(),
            nn.Linear(dims1[-1][0] * dims1[-1][1] * dims1[-1][2], zdim * 2),
            nn.LeakyReLU(),
            nn.Linear(zdim * 2, zdim),
            nn.LeakyReLU()
        )

        self.encode_targets = nn.Sequential(
            Flatten(),
            nn.Linear(aux_code_size * num_spaces,
                      min(zdim, aux_code_size * num_spaces) * 2),
            nn.LeakyReLU(),
            nn.Linear(min(zdim, aux_code_size * num_spaces) * 2, zdim),
            nn.LeakyReLU()
        )
        self.merge = nn.Bilinear(zdim, zdim, zdim)

        self.dec1 = DeConvNormRelu(zdim * 2, 32, **_ly[3], batch_norm=False)
        self.dec2 = DeConvNormRelu(32 * 2, 16, **_ly[2])
        self.dec3 = DeConvNormRelu(16 * 2, 8, **_ly[1])
        self.dec4 = DeConvNormRelu(8 * 2, num_spaces, **_ly[0], activation=lambda x: x)

        self.fc_critic = nn.Linear(zdim, 1)
        self.fc_auxilary = nn.ModuleList([nn.Linear(zdim, num_spaces)
                                          for _ in range(aux_code_size)])
        self.softmax2 = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()
        self.predictor_module = None
        self.apply(weights_init)

    def forward(self, obs, hidden=None):
        """
        todo - make this real
        img: size [batch_size, c, h, w]
        """
        img, target_code = obs

        # Stack of convs with features cache
        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        # encode for target and z
        tgt = self.encode_targets(target_code)

        # todo attention layer
        z = self.merge(flatten(x4), tgt)

        # value for predicion module
        value = self.fc_critic(z)
        auxs = torch.sigmoid(torch.stack([m(z) for m in self.fc_auxilary]).permute(1, 2, 0))

        # decode into an image of [b, c, h, w]
        # at each step there is a skip connection to diluted input
        x = self.dec1(torch.cat((z.unsqueeze(-1).unsqueeze(-1), x4), 1))
        x = self.dec2(torch.cat((x, x3), 1))
        x = self.dec3(torch.cat((x, x2), 1))
        x = self.dec4(torch.cat((x, x1), 1))

        probs = self.softmax2(img[:, 0:x.size(1), :, :] + x) # self.softmax2(x))

        log_prob = log_prob_from_logits(x).mean()
        sf = self.softmax2(x)
        entropy = -(sf * torch.log(sf)).sum(-1).mean(dim=(1, 2), keepdim=True).squeeze(-1)
        return {'log_prob': log_prob,  # to maximize
                'entropy': entropy,  # to minimize
                'action': probs,
                'hidden': None,
                'value': value,
                'aux': auxs,
                'x': x.detach().squeeze().cpu(),
                'z': z,
                }





# ----------------------------------------------------------------------------------
class NextStatePred(Module):
    def __init__(self, z_size, base=None, pred=None):
        Module.__init__(self)
        self.base_net = base
        self.to_z = pred

    def target(self, next_obs):
        return

    def predictor(self, next_obs):
        return

    def forward(self, state, action):
        """
        transition tuiple (s_t, s_t+1, a_t)
        a) embed obseervations into representations φ(s_t )
        b) forward dynamics network F ( φ(s_t+1 ) | a_t) -> s_t+1

        r t = − log p(φ(x t+1 )|x t , a t ),
        """
        phi_s = self.base_net(state)
        # predict action
        # continuous prediction [n, 4 ]
        action = self.fc_action_loc(phi_s)
        pred_act = sample_normal(torch.tanh(action), scale=F.softplus(self.std))
        this_z = self.to_z(phi_s, pred_act)
        pred_z = self.to_z(phi_s, pred_act)

        # --------------------------------
        x = self.feature(state)
        policy = self.actor(x)
        value_ext = self.critic_ext(self.extra_layer(x) + x)
        value_int = self.critic_int(self.extra_layer(x) + x)

        return policy, value_ext, value_int

    def impression(self, problem):
        """ generate z of features """
        # sample a point in the problem space


class RNDModel(Module):
    def __init__(self, input_size, output_size):
        super(RNDModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        feature_output = 7 * 7 * 64
        self.predictor = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.target = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        )
        self.__init_weights()

    def __init_weights(self):
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)
        return predict_feature, target_feature


class GACLinearStds(GaussianActorCriticNet):
    """ try with parametrizing all stds --- but why ??? """
    def __init__(self, *args, **kwargs):
        GaussianActorCriticNet.__init__(self , *args, **kwargs)
        self.std = nn.Linear(self.feature_dim, self.action_dim)

    def cov_mat(self, x=None):
        return self.std(x)


class GaussianActorCriticNet2(Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        Module.__init__(self)
        # if phi_body is None: phi_body = DummyBody(state_dim)
        # if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        # if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action_x = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_action_y = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())

        self.controller = nn.LSTMCell(ch * cw, lstm_out_size)

        self.fc_sigma_x = nn.Linear()
        self.fc_sigma_y = nn.Linear()

        self.fc_mu_x = nn.Linear()
        self.fc_mu_y = nn.Linear()

        self.std_y = nn.Parameter(torch.zeros(action_dim))
        self.std_x = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs, hidden, action=None):
        """ model an action as discrete, continuos """
        phi = self.phi_body(obs)
        hx, cx = self.controller(phi, hidden)
        phi = hx

        phi_a = self.actor_body(phi)
        v = self.critic_body(phi)

        # [n, S + 4]
        mean = F.tanh(a)
        dist = D.Normal(mean, F.softplus(self.std_x))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        # return action, log_prob, entropy, mean

        # v = self.fc_critic(phi_v)
        return {'a': action,
                'log_pi_a': log_prob,
                'hidden': (hx, cx),
                'ent': entropy,
                'mean': mean,
                'v': v,
                }


# encoder and decoder modules
class EncoderRNN(Module):
    def __init__(self, enc_h_size, hp, device='cuda'):
        Module.__init__(self)
        self.hp = hp
        self.device = device
        self.enc_h_size = enc_h_size
        # bidirectional lstm:
        # self._is_cuda = hp.is_cuda
        self.lstm = nn.LSTM(5, enc_h_size, dropout=hp.dropout, bidirectional=True)

        # create mu and sigma from lstm's last output:
        self.fc_mu = nn.Linear(2 * enc_h_size, hp.Nz)
        self.fc_sigma = nn.Linear(2 * enc_h_size, hp.Nz)

        # active dropout:
        self.train()

    def forward(self, inputs, batch_size, hidden_cell=None):
        if hidden_cell is None:
            # then must init with zeros
            hidden = torch.zeros(2, batch_size, self.enc_h_size, device=self.device)
            cell = torch.zeros(2, batch_size, self.enc_h_size, device=self.device)
            hidden_cell = (hidden, cell)

        _, (hidden, cell) = self.lstm(inputs.float(), hidden_cell)
        # hidden is (2, batch_size, hidden_size), we want (batch_size, 2*hidden_size):
        hidden_forward, hidden_backward = torch.split(hidden, 1, 0)
        hidden_cat = torch.cat((hidden_forward.squeeze(0), hidden_backward.squeeze(0)), 1)
        # mu and sigma:
        mu = self.fc_mu(hidden_cat)
        sigma_hat = self.fc_sigma(hidden_cat)
        sigma = torch.exp(sigma_hat / 2.)

        # N ~ N(0,1)
        z_size = mu.size()
        N = torch.normal(torch.zeros(z_size), torch.ones(z_size)).to(self.device)
        z = mu + sigma * N
        # mu and sigma_hat are needed for LKL loss
        return z, mu, sigma_hat


class DecoderRNN(Module):
    def __init__(self, hp):
        Module.__init__(self)
        self.hp = hp
        # to init hidden and cell from z:
        self.fc_hc = nn.Linear(hp.Nz, 2 * hp.dec_hidden_size)
        # unidirectional lstm:
        self.lstm = nn.LSTM(hp.Nz + 5, hp.dec_hidden_size, dropout=hp.dropout)
        # create proba distribution parameters from hiddens:
        self.fc_params = nn.Linear(hp.dec_hidden_size, 6 * hp.M + 3)

    def forward(self, inputs, z, hidden_cell=None):
        """
        in this case, inputs can be the image/field, z is previous actions

        there is no reconsruction loss.

        action is a continuous function paramaterized by mus and sigmas
        """
        if hidden_cell is None:
            # then we must init from z
            hidden, cell = torch.split(F.tanh(self.fc_hc(z)), self.hp.dec_hidden_size, 1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        outputs, (hidden, cell) = self.lstm(inputs, hidden_cell)
        # in training we feed the lstm with the whole input in one shot
        # and use all outputs contained in 'outputs', while in generate
        # mode we just feed with the last generated sample:
        if self.training:
            y = self.fc_params(outputs.view(-1, self.hp.dec_hidden_size))
        else:
            y = self.fc_params(hidden.view(-1, self.hp.dec_hidden_size))
        # separate pen and mixture params:
        params = torch.split(y, 6, 1)
        params_mixture = torch.stack(params[:-1])  # trajectory
        params_pen = params[-1]  # pen up/down
        # identify mixture params:
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)

        # preprocess params::
        if self.training:
            len_out = self.hp.Nmax + 1
        else:
            len_out = 1

        pi = F.softmax(pi.transpose(0, 1).squeeze()).view(len_out, -1, self.hp.M)
        sigma_x = torch.exp(sigma_x.transpose(0, 1).squeeze()).view(len_out, -1, self.hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0, 1).squeeze()).view(len_out, -1, self.hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0, 1).squeeze()).view(len_out, -1, self.hp.M)
        mu_x = mu_x.transpose(0, 1).squeeze().contiguous().view(len_out, -1, self.hp.M)
        mu_y = mu_y.transpose(0, 1).squeeze().contiguous().view(len_out, -1, self.hp.M)
        q = F.softmax(params_pen).view(len_out, -1, 3)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell


class STNNet(Module):
    def __init__(self):
        super(STNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ----------------------------------------------------------------------------------
# hyperparameters
class HParams():
    def __init__(self):
        self.data_location = 'cat.npz'
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        self.Nz = 128
        self.M = 20
        self.dropout = 0.9
        self.batch_size = 100
        self.eta_min = 0.01
        self.R = 0.99995
        self.KL_min = 0.2
        self.wKL = 0.5
        self.lr = 0.001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001
        self.grad_clip = 1.
        self.temperature = 0.4
        self.max_seq_length = 200


# load and prepare data
def max_size(data):
    """larger sequence length in the data set"""
    return max([len(seq) for seq in data])


def purify(strokes):
    """removes to small or too long sequences + removes large gaps"""
    data = []
    for seq in strokes:
        if seq.shape[0] <= hp.max_seq_length and seq.shape[0] > 10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
    return data


def calculate_normalizing_scale_factor(strokes):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    data = []
    for i in range(len(strokes)):
        for j in range(len(strokes[i])):
            data.append(strokes[i][j, 0])
            data.append(strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)


def normalize(strokes):
    """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
    data = []
    scale_factor = calculate_normalizing_scale_factor(strokes)
    for seq in strokes:
        seq[:, 0:2] /= scale_factor
        data.append(seq)
    return data


def make_batch(data, batch_size, Nmax, device='cuda'):
    """# function to generate a batch:"""
    batch_idx = np.random.choice(len(data), batch_size)
    batch_sequences = [data[idx] for idx in batch_idx]
    strokes = []
    lengths = []
    indice = 0
    for seq in batch_sequences:
        len_seq = len(seq[:, 0])
        new_seq = np.zeros((Nmax, 5))
        new_seq[:len_seq, :2] = seq[:, :2]
        new_seq[:len_seq - 1, 2] = 1 - seq[:-1, 2]
        new_seq[:len_seq, 3] = seq[:, 2]
        new_seq[(len_seq - 1):, 4] = 1
        new_seq[len_seq - 1, 2:4] = 0
        lengths.append(len(seq[:, 0]))
        strokes.append(new_seq)
        indice += 1
    batch = torch.from_numpy(np.stack(strokes, 1)).to(device).float()
    return batch, lengths


def lr_decay(optimizer, hp):
    """ adaptive lr Decay learning rate by a factor of lr_decay"""
    for param_group in optimizer.param_groups:
        if param_group['lr'] > hp.min_lr:
            param_group['lr'] *= hp.lr_decay
    return optimizer


class Canny(nn.Module):
    def __init__(self, threshold=10.0, use_cuda=False):
        super(Canny, self).__init__()
        from scipy.signal import gaussian
        self.threshold = threshold
        self.use_cuda = use_cuda

        filter_size = 5
        generated_filters = gaussian(filter_size, std=1.0).reshape([1, filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

    def forward(self, img):
        img_r = img[:,0:1]
        img_g = img[:,1:2]
        img_b = img[:,2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES

        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
        grad_orientation += 180.0
        grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)

        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width
        pixel_range = torch.FloatTensor([range(pixel_count)])
        if self.use_cuda:
            pixel_range = torch.cuda.FloatTensor([range(pixel_count)])

        indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(1,height,width)

        indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1,height,width)

        channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=0)

        thin_edges = grad_mag.clone()
        thin_edges[is_max == 0] = 0.0

        # THRESHOLD
        thresholded = thin_edges.clone()
        thresholded[thin_edges<self.threshold] = 0.0

        early_threshold = grad_mag.clone()
        early_threshold[grad_mag<self.threshold] = 0.0

        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()

        return blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold

