import torch
import torch.nn as nn
from src.algo.init_custom import *


class SOM(nn.Module):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """

    def __init__(self, m, n, dim, niter, alpha=None, sigma=None):
        super(SOM, self).__init__()
        self.m = m
        self.n = n
        self.dim = dim
        self.niter = niter
        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = max(m, n) / 2.0
        else:
            self.sigma = float(sigma)

        self.weights = torch.randn(m * n, dim)
        self.locations = torch.LongTensor(np.array(list(self.neuron_locations())))
        self.pdist = nn.PairwiseDistance(p=2)

    def get_weights(self):
        return self.weights

    def get_locations(self):
        return self.locations

    def neuron_locations(self):
        for i in range(self.m):
            for j in range(self.n):
                yield np.array([i, j])

    def map_vects(self, input_vects):
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self.weights))],
                            key=lambda x: np.linalg.norm(vect - self.weights[x]))
            to_return.append(self.locations[min_index])

        return to_return

    def forward(self, x, it):

        dists = self.pdist(torch.stack((x for i in range(self.m * self.n))), self.weights)

        _, bmu_index = torch.min(dists, 0)
        bmu_loc = self.locations[bmu_index, :]
        bmu_loc = bmu_loc.squeeze()

        learning_rate_op = 1.0 - it / self.niter
        alpha_op = self.alpha * learning_rate_op
        sigma_op = self.sigma * learning_rate_op

        bmu_distance_squares = torch.sum(
            torch.pow(self.locations.float() -
                      torch.stack((bmu_loc for i in range(self.m * self.n))).float(), 2), 1)

        neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma_op ** 2)))

        learning_rate_op = alpha_op * neighbourhood_func

        learning_rate_multiplier = torch.stack(
            (learning_rate_op[i:i + 1].repeat(self.dim) for i in range(self.m * self.n))
        )
        delta = torch.mul(learning_rate_multiplier, (torch.stack((x for i in range(self.m * self.n))) - self.weights))
        new_weights = torch.add(self.weights, delta)
        self.weights = new_weights


class ActorCritic(nn.Module):
    """ Original Actor Critic - github
        DO NOT FUCK WITH
    """
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 3 * 3)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
