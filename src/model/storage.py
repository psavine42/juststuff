from collections import namedtuple
import random
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

Sample = namedtuple('Sample', ('state',      'features',
                               'action',     'reward',
                               'entropy',    'reward_pred',
                               'next_state', 'next_features',
                               'hidden1',    'hidden2',
                               'term_mask',  'eos_mask'))



def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Sample(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def generator(self, batch_size):
        transitions = self.sample(batch_size)
        batch = Sample(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                      device=self.device,
                                      dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # hidden states
        hidden_batch = torch.cat(batch.hidden)
        return non_final_next_states, state_batch, action_batch, reward_batch, hidden_batch

    def __len__(self):
        return len(self.memory)


class Buffer(object):
    def __init__(self,
                 capacity,
                 state_size, hidden_size, action_size, feats_size):
        self._state_size = state_size
        self._hidden_size = hidden_size
        self._action_size = action_size
        self._feats_size = feats_size
        self.capacity = capacity

        self.state = None
        self.feats = None
        self.actions = None
        self.next_state = None
        self.next_feats = None
        self.rewards = None
        self.hidden1 = None
        self.hidden2 = None
        self.term_mask = None
        self.eos_mask = None
        self.entropy = None
        self.value_pred = None
        self.__len = 0

    def reset(self):
        self.state = torch.zeros(self.capacity, *self._state_size)
        self.feats = torch.zeros(self.capacity, *self._feats_size)

        self.next_state = torch.zeros(self.capacity, *self._state_size)
        self.next_feats = torch.zeros(self.capacity, *self._feats_size)

        self.actions = torch.zeros(self.capacity, *self._action_size)
        self.rewards = torch.zeros(self.capacity, 1)

        self.hidden1 = torch.zeros(self.capacity, *self._hidden_size)
        self.hidden2 = torch.zeros(self.capacity, *self._hidden_size)

        self.term_mask = torch.zeros(self.capacity, 1)
        self.eos_mask = torch.zeros(self.capacity, 1)
        self.__len = 0

    def push(self, sample):
        """Saves a transition."""
        self.state[self.__len] = sample.state
        self.feats[self.__len] = sample.feature
        self.actions[self.__len] = sample.action
        self.next_state[self.__len] = sample.next_state
        self.next_feats[self.__len] = sample.next_feature
        self.rewards[self.__len] = sample.rewards
        self.hidden1[self.__len] = sample.hidden1
        self.hidden2[self.__len] = sample.hidden2
        self.term_mask[self.__len] = sample.term_mask
        self.eos_mask[self.__len] = sample.eos_mask
        # self.position = (self.position + 1) % self.capacity
        self.__len += 1

    def sample(self, batch_size):
        start_idx = torch.randint(0, self.__len - batch_size)
        inputs = []
        # states are sampled in order

        return

    @property
    def is_full(self):
        return self.__len >= self.capacity


class Storage:
    def __init__(self, size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['s', 'a', 'r', 'm',
                       'v', 'q', 'pi', 'log_pi', 'ent',
                       'adv', 'ret', 'q_a', 'log_pi_a',
                       'mean']
        self.keys = keys
        self.size = size
        self.reset()

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return getattr(self, item)

    def add(self, data):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)



class RolloutStorage(object):
    def __init__(self,
                 num_steps,
                 num_processes,
                 obs_shape,
                 action_space,
                 recurrent_hidden_state_size):

        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)

        self.returns = torch.zeros(num_steps + 1, num_processes, 1)

        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs,
               recurrent_hidden_states, actions, action_log_probs,
               value_preds,
               rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] \
                            * self.masks[step + 1] - self.value_preds[step]

                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):

                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] \
                            * self.masks[step + 1] - self.value_preds[step]

                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)

        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch =         _flatten_helper(T, N, torch.stack(obs_batch, 1))
            actions_batch =     _flatten_helper(T, N, torch.stack(actions_batch, 1))
            value_preds_batch = _flatten_helper(T, N, torch.stack(value_preds_batch, 1))
            return_batch =      _flatten_helper(T, N, torch.stack(return_batch, 1))
            masks_batch =       _flatten_helper(T, N, torch.stack(masks_batch, 1))
            old_action_log_probs_batch = _flatten_helper(T, N, torch.stack(old_action_log_probs_batch, 1))
            adv_targ =          _flatten_helper(T, N, torch.stack(adv_targ, 1))

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

