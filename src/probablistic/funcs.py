import numpy as np
import torch
import torch.distributions as D
import scipy.ndimage as nd


def sample_(dist, action=None):
    if action is None:
        action = dist.sample()
    log_prob = dist.log_prob(action).unsqueeze(-1)
    entropy = dist.entropy().unsqueeze(-1)
    return {'action': action,
            'log_prob': log_prob,
            'entropy': entropy}


def sample_categorical(logits, **kwargs):
    return sample_(D.Categorical(logits, **kwargs))


def sample_normal(means, **kwargs):
    return sample_(D.Normal(means, **kwargs))


def sample_multinomial(*args, **kwargs):
    return sample_(D.Multinomial(*args, **kwargs))


def sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, temp=1, greedy=False):
    # inputs must be floats
    if greedy:
        return mu_x, mu_y
    mean = [mu_x, mu_y]
    sigma_x *= np.sqrt(temp)
    sigma_y *= np.sqrt(temp)
    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y], \
           [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def get_action_log_prob(batch_mu, batch_log_sigma, min_val):
    # batch_mu, batch_log_sigma = self.policy_net(state)
    batch_sigma = torch.exp(batch_log_sigma)
    dist = torch.normal(batch_mu, batch_sigma)
    z = dist.sample()
    action = torch.tanh(z)
    log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + min_val)
    return action, log_prob, z  # , batch_mu, batch_log_sigma


def update(model, optimizer, replay_buffer, batch_size):
    if batch_size > len(replay_buffer):
        return
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    q_value = model(state)
    q_value = q_value.gather(1, action.unsqueeze(1)).squeeze(1)

    next_q_value = model(next_state).max(1)[0]
    expected_q_value = (reward + 0.99 * next_q_value * (1 - done)).detach()

    loss = (q_value - expected_q_value).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def gae(storage, R, args):
    policy_loss = 0
    value_loss = 0
    gae = torch.zeros(1, 1, device=args.device)

    for i in reversed(range(len(storage))):
        R = args.gamma * R + storage.rewards[i]
        advantage = R - storage.values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)

        # Generalized Advantage Estimation
        delta_t = storage.rewards[i] + args['gamma'] * storage.values[i + 1] - storage.values[i]
        gae = gae * args['gamma'] * args['gae_lambda '] + delta_t
        policy_loss = policy_loss - storage.log_probs[i] * gae.detach() - \
                      args['entropy_coef '] * storage.entropies[i]

    return policy_loss + args['value_loss_coef'] * value_loss


class LKLLoss:
    def __init__(self, args):
        self.wKL = args.wkl
        # self.eta_step = args.eta_step
        self.batch = args.batch

    def kullback_leibler_loss(self, KL_min, sigma, mu, eta, size):
        LKL = -0.5 * torch.sum(1 + sigma - mu ** 2 - torch.exp(sigma)) / size
        # / float(self.hp.Nz * self.hp.batch_size)

        KL_min = torch.FloatTensor([KL_min], device=self.device).detach()
        return self.wKL * eta * torch.max(LKL, KL_min)


def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4,
              end='inception_4c/output', clip=True, **step_params):
    """https://github.com/google/deepdream/blob/master/dream.ipynb """
    # prepare base images for all octaves
    def preprocess(net, img):
        return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

    def deprocess(net, img):
        return np.dstack((img + net.transformer.mean['data'])[::-1])

    def objective_L2(dst):
        dst.diff[:] = dst.data

    def make_step(net, step_size=1.5, end='inception_4c/output',
                  jitter=32, clip=True, objective=objective_L2):
        '''Basic gradient ascent step.'''

        src = net.blobs['data']  # input image is stored in Net's 'data' blob
        dst = net.blobs[end]

        ox, oy = np.random.randint(-jitter, jitter + 1, 2)
        src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)  # apply jitter shift

        net.forward(end=end)
        objective(dst)  # specify the optimization objective
        net.backward(start=end)
        g = src.diff[0]
        # apply normalized ascent step to the input image
        src.data[:] += step_size / np.abs(g).mean() * g

        src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)  # unshift image

        if clip:
            bias = net.transformer.mean['data']
            src.data[:] = np.clip(src.data, -bias, 255 - bias)


    octaves = [preprocess(net, base_img)]
    for i in range(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1])  # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        src.reshape(1, 3, h, w)  # resize the network's input image size
        src.data[0] = octave_base + detail
        for i in range(iter_n):
            make_step(net, end=end, clip=clip, **step_params)

            # visualization
            vis = deprocess(net, src.data[0])
            if not clip:  # adjust image contrast if clipping is disabled
                vis = vis * (255.0 / np.percentile(vis, 99.98))
            # showarray(vis)
            print(octave, i, end, vis.shape)
            # clear_output(wait=True)

        # extract details produced on the current octave
        detail = src.data[0] - octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

