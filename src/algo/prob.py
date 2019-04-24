
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
import torch.distributions
import torch.nn as nn
import torch.optim as optim

pyro.set_rng_seed(101)


def scale(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.0))
    return pyro.sample("measurement", dist.Normal(weight, 0.75))


def test1():
    guess = torch.tensor(8.5)

    pyro.clear_param_store()
    svi = pyro.infer.SVI(model=conditioned_scale,
                         guide=scale_parametrized_guide,
                         optim=pyro.optim.SGD({"lr": 0.001, "momentum": 0.1}),
                         loss=pyro.infer.Trace_ELBO())

    losses, a, b = [], [], []
    num_steps = 2500
    for t in range(num_steps):
        X = svi.step(guess)
        losses.append(X)
        a.append(pyro.param("a").item())
        b.append(pyro.param("b").item())

    plt.plot(losses)
    plt.title("ELBO")
    plt.xlabel("step")
    plt.ylabel("loss")
    print('a = ', pyro.param("b").item())
    print('b = ', pyro.param("a").item())



def lpath():
    num_items = 4



def test2():
    """
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
    #  n1 = mixture.NormalDistribution(-2,0.4)
    #  n2 = mixture.NormalDistribution(2,0.6)
    #  m = mixture.MixtureModel(2,[0.5,0.5], [n1,n2])
    Something like this
    """
    num_items = 2

    psx = [dist.Normal(0.2, 0.3)]
    psy = [dist.Normal(0.5, 0.2), dist.Normal(0.3, 0.2), dist.Normal(0.7, 0.2)]
    mixture1 = dist.GaussianScaleMixture(2 , psx, psy)

    psx2 = [dist.Normal(0.2, 0.3)]
    psy2 = [dist.Normal(0.5, 0.2), dist.Normal(0.3, 0.2), dist.Normal(0.7, 0.2)]
    mixture2 = dist.GaussianScaleMixture(2, psx2, psy2)

    dists = [mixture1, mixture2]
    cats = dist.Categorical(num_items, [])

    placements = []
    for i in range(100):
        placement = []
        # expirement
        for j in range(num_items):

            # this | previous placements
            to_place = cats.sample()
            dist_to_use = dists[to_place]

            # generate locations
            loc_x, loc_y = dist_to_use.sample(), dist_to_use.sample()

            # generate bounds
            bound_x, bound_y = random.random(), random.random()

            # resulting distribution from 'placement' s.t. other | this -> 0
            new_distx, new_disty = place(loc_x, loc_y, bound_x, bound_y)
            dists[0] = update_mix(dists[0], new_distx)
            dists[1] = update_mix(dists[1], new_disty)

            placement.extend([to_place, loc_x, loc_y, bound_x, bound_y])
        lossfn(placement)

        placements.append(placement)


def test_opt_tree(searcher, costfn, problem):
    """ search over sequences of optimizations

    """
    from scipy.spatial import Voronoi
    solver = None
    g = problem.to_graph()
    # create a voronoi diagram of the space
    V = Voronoi(g.nodes)

    faces = V.ridge_dict
    init_state = Domain(faces)

    actions = searcher.get_actions(init_state)




def test3(num_iter=100):
    """ concept

    extractor: an encoder s.t.:
        F(w) -> num_items x 4 s.t. [x, y, u, v]

        learn to encode sample space into

        the point is to generate some stupid tests and not deal with decoding

        optimization is done
    """
    num_items = 2
    width = 40
    height = 30

    # create a discrete world
    world = torch.zeros(num_items, width, height, requires_grad=True)

    # precondition using distributions, normalize
    world = world

    # setup extractor
    model = nn.Linear(num_items * width * height, num_items * 4)

    # setup optimizers
    opt = optim.Adam(model)
    for i in range(num_iter):
        opt.zero_grad()
        X = model(world)
        loss = lossfn(X)
        loss.backward()
        opt.step()






# distributions = [
#     {"type": np.random.normal, "kwargs": {"loc": -3, "scale": 2}},
#     {"type": np.random.uniform, "kwargs": {"low": 4, "high": 6}},
#     {"type": np.random.normal, "kwargs": {"loc": 2, "scale": 1}},
# ]
# coefficients = np.array([0.5, 0.2, 0.3])
# coefficients /= coefficients.sum()      # in case these did not add up to 1
# sample_size = 100000
#
# num_distr = len(distributions)
# data = np.zeros((sample_size, num_distr))
# for idx, distr in enumerate(distributions):
#     data[:, idx] = distr["type"](size=(sample_size,), **distr["kwargs"])
# random_idx = np.random.choice(np.arange(num_distr), size=(sample_size,), p=coefficients)
# sample = data[np.arange(sample_size), random_idx]
# plt.hist(sample, bins=100, density=True)
# plt.show()