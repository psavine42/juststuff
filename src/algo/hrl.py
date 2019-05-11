import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from src.model.storage import Buffer, Sample, Storage
import torch.distributions as D
from torch import tensor


class HLR:
    """
    https://arxiv.org/abs/1901.01365
    Input: Number of options O, size of on-policy buffer
    Initialize: Replay buffer D R , on-policy buffer D on ,
        network parameters η, θ, w, θ target , w target
    repeat
        for t = 0 to t = T do
            Draw an option for a given s by following Equation 17: o ∼ π(o|s)
            Draw an action a ∼ β(a|s, o) = μ o θ (s) + e
            Record a data sample (s, a, r, s' )
            Aggregate the data in D R and D on
            if the on-policy buffer is full then
                Update the option network by minimizing Equation (7) for samples in D on
                Clear the on-policy buffer D on
            end if
            Sample a batch D batch ∈ D R
            Update the Q network parameter w
            if t mod d then
                Estimate p(o|s i , a i ) for (s i , a i ) ∈ D batch using the option network
                Assign samples (s i , a i ) ∈ D batch to the option o ∗ = arg max p(o|s i , a i )
                Update the option policy networks μ o θ (s) for o = 1, ..., O with Equation (19)
                Update the target networks: w target ← τ w +(1−τ )w target , θ target ← τ θ +(1−τ )θ target
            end if
        end for
    until the convergence
    return θ

    """
    def __init__(self, env, policy_class, option_class, num_options,
                 state_dim=(20, 20),
                 batch_size=64,
                 mem_size=10000):
        self.env = env
        self.option = option_class(state_dim, num_options)
        self.policy = policy_class(state_dim, num_options)  # μ o θ (s)

        self.policy_target = policy_class(state_dim, num_options)
        self.option_target = option_class(state_dim, num_options)

        self.state_dim = state_dim
        self.batch_size = batch_size
        self.num_options = num_options
        self.update_theta = 10

        self.__buffer_replay = Buffer(mem_size)
        self.__buffer_policy = Buffer(mem_size)

    def env_step(self, state, option, action):
        pass

    def optimize_option(self):
        return

    def step(self, state, time_step):
        option = self.option(state)
        action = self.policy(state, option)

        action_data = self.env.step(state, option, action)
        # sample = Sample(state.detach(), action.detach(), reward, next_state.detach())
        self.__buffer_policy.push(sample)
        self.__buffer_replay.push(sample)

        if self.__buffer_policy.is_full:
            self.optimize_option()
            self.__buffer_policy.reset()

        # optimize by sample_batches

        if time_step % self.update_theta == 0:
            batches = self.__buffer_replay.sample(self.batch_size)
            self.option()
            self.policy_target.load_state_dict(self.policy.state_dict())


class OptionCriticAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.worker_index = tensor(np.arange(config.num_workers)).long()

        self.states = self.config.state_normalizer(self.task.reset())
        self.is_initial_states = tensor(np.ones((config.num_workers))).byte()
        self.prev_options = self.is_initial_states.clone().long()

    def sample_option(self, prediction, epsilon, prev_option, is_intial_states):
        with torch.no_grad():
            q_option = prediction['q']
            pi_option = torch.zeros_like(q_option).add(epsilon / q_option.size(1))
            greedy_option = q_option.argmax(dim=-1, keepdim=True)
            prob = 1 - epsilon + epsilon / q_option.size(1)
            prob = torch.zeros_like(pi_option).add(prob)
            pi_option.scatter_(1, greedy_option, prob)

            mask = torch.zeros_like(q_option)
            mask[:, prev_option] = 1
            beta = prediction['beta']
            pi_hat_option = (1 - beta) * mask + beta * pi_option

            dist = D.Categorical(probs=pi_option)
            options = dist.sample()
            dist = D.Categorical(probs=pi_hat_option)
            options_hat = dist.sample()

            options = torch.where(is_intial_states, options, options_hat)
        return options

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length, ['beta', 'o', 'beta_adv', 'prev_o', 'init', 'eps'])

        for _ in range(config.rollout_length):
            prediction = self.network(self.states)
            epsilon = config.random_option_prob(config.num_workers)
            options = self.sample_option(prediction, epsilon, self.prev_options, self.is_initial_states)
            prediction['pi'] = prediction['pi'][self.worker_index, options]
            prediction['log_pi'] = prediction['log_pi'][self.worker_index, options]
            dist = D.Categorical(probs=prediction['pi'])
            actions = dist.sample()
            entropy = dist.entropy()

            next_states, rewards, terminals, info = self.task.step(to_np(actions))
            self.record_online_return(info)
            next_states = config.state_normalizer(next_states)
            rewards = config.reward_normalizer(rewards)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         'o': options.unsqueeze(-1),
                         'prev_o': self.prev_options.unsqueeze(-1),
                         'ent': entropy.unsqueeze(-1),
                         'a': actions.unsqueeze(-1),
                         'init': self.is_initial_states.unsqueeze(-1).float(),
                         'eps': epsilon})

            self.is_initial_states = tensor(terminals).byte()
            self.prev_options = options
            self.states = next_states

            self.total_steps += config.num_workers
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        with torch.no_grad():
            prediction = self.target_network(self.states)
            storage.placeholder()
            betas = prediction['beta'][self.worker_index, self.prev_options]
            ret = (1 - betas) * prediction['q'][self.worker_index, self.prev_options] + \
                  betas * torch.max(prediction['q'], dim=-1)[0]
            ret = ret.unsqueeze(-1)

        for i in reversed(range(config.rollout_length)):
            ret = storage.r[i] + config.discount * storage.m[i] * ret
            adv = ret - storage.q[i].gather(1, storage.o[i])
            storage.ret[i] = ret
            storage.adv[i] = adv

            v = storage.q[i].max(dim=-1, keepdim=True)[0] * (1 - storage.eps[i]) + storage.q[i].mean(-1).unsqueeze(-1) * storage.eps[i]
            q = storage.q[i].gather(1, storage.prev_o[i])
            storage.beta_adv[i] = q - v + config.termination_regularizer

        q, beta, log_pi, ret, adv, beta_adv, ent, option, action, initial_states, prev_o = \
            storage.cat(['q', 'beta', 'log_pi', 'ret', 'adv', 'beta_adv', 'ent', 'o', 'a', 'init', 'prev_o'])

        q_loss = (q.gather(1, option) - ret.detach()).pow(2).mul(0.5).mean()
        pi_loss = -(log_pi.gather(1, action) * adv.detach()) - config.entropy_weight * ent
        pi_loss = pi_loss.mean()
        beta_loss = (beta.gather(1, prev_o) * beta_adv.detach() * (1 - initial_states)).mean()

        self.optimizer.zero_grad()
        (pi_loss + q_loss + beta_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()


