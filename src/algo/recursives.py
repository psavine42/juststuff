import torch
import torch.nn as nn
from torch.nn import Module, Parameter
from src.probablistic.funcs import *
from src.model.storage import Storage
from src.algo.layers import *

class SMC(Module):
    """
    Bengio et all

    https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=29&ved=2ahUKEwi18_6X8qPiAhVjxFkKHUl1CMo4FBAWMAh6BAgBEAI&url=https%3A%2F%2Fwww.aclweb.org%2Fanthology%2FW18-3020&usg=AOvVaw1GFZuztn5GrI8MV0_Wmg67

    """
    def __init__(self, size, input_size):
        Module.__init__(self)
        self.counter = 0    # nsteps
        self.stack = []     # fuckit. its a list

        self.split_f1 = nn.LSTMCell(size, size)     # F1
        self.split_f2 = nn.LSTMCell(size, size)     # F2

        self.recurrent = nn.LSTMCell(size, size)    # F3

        self.merge_mlp = nn.Sequential(nn.Linear(2*size, size))
        self.merge_lstm = nn.LSTMCell(size, size)   # F4

        self.policy = nn.Linear(input_size, 3)

    def split(self, x, prev_hx):
        h_down, _ = self.split_f1(prev_hx, x)
        h_l, _ = self.split_f2(prev_hx, x)
        self.stack.append(h_down)
        return h_l

    def merge(self, x, prev_hx):
        if self.stack:      # not sure how to deal with merge when no stack
            h_down = self.stack.pop(-1)
            h_m = self.merge_mlp(prev_hx, h_down)
            _, h_l = self.merge_lstm(x, h_m)
            return h_l
        return prev_hx

    def recur(self, x, prev_hx):
        h_l, x_hat = self.recurrent(prev_hx, x)
        self.counter += 1
        return x_hat, h_l

    def forward(self, x, hidden, L=10):
        """REINFORCE algorithm using − log p(y_t |C) as a reward,
        where y_t is the task target (i.e. the next word in language modeling),
        and C is the representation learnt by the model up until time t.
        Maximize log_liekyhood of observations
        ---------------------------------------------------------

        """
        logits = self.policy(x, hidden)
        action = sample_categorical(logits)
        action['logits'] = logits

        ix = logits.argmax()
        if ix == 0:
            new_hidden = self.split(x, hidden)
        elif ix == 1:
            new_hidden = self.merge(x, hidden)
        else:
            x_hat, new_hidden = self.recur(x, hidden)
            action['action'] = x_hat
        action['hidden'] = new_hidden
        return action


class GBP(Module):
    def __init__(self, enc, dec, action_size, z_size, target_size,
                 shared_size=None):
        Module.__init__(self)
        shared_size = shared_size if shared_size else 2 * action_size

        self.encode_state = enc
        self.decode_state = dec

        self.encode_target = MLP2(target_size, shared_size)
        self.encode_action = MLP2(action_size, shared_size)
        self.decode_action = MLP2(shared_size, action_size)

        #
        self.merge_saz = MLP2(2 * shared_size + z_size, z_size)
        self.pred_reward = MLP2(2 * shared_size + z_size, 1)

        self.policy = MLP2(z_size, action_size)
        self.sigmoid = nn.Sigmoid()

    def encode(self, state):
        img, feats = state
        zs = self.encode_state(img)
        zf = self.sigmoid(self.encode_target(feats))
        return torch.cat((zs, zf))

    def transition(self, state, action):
        zs = self.encode_state(state)
        za = self.encode_action(action)

        z = self.merge_saz(torch.cat(zs, za))
        next_state = self.decode_state(z)
        return next_state

    def reward(self, state, action):
        zs = self.encode_state(state)
        za = self.encode_action(action)

        z = self.merge_saz(torch.cat(zs, za))
        reward = self.sigmoid(self.pred_reward(z))
        return reward

    def policy(self, state):
        zs = self.encode_state(state)
        action = self.policy(zs)
        return action

    def forward(self, state):
        """ for supervised only - runs all modules
            action
            predict_next

        """
        zs = self.encode_state(state)
        action = self.policy(zs)

        za = self.encode_action(action)
        z = self.merge_saz(torch.cat(zs, za))

        next_state = self.decode_state(z)
        reward = self.sigmoid(self.pred_reward(z))
        return action, next_state, reward


class PseudoKD(Module):
    """η lets try decoding the requirements directly """
    def __init__(self, policy):
        Module.__init__(self)
        self.policy = policy

    def forward(self, x, hidden):
        action = self.policy(x, hidden)


def train_SMC(env, model, episodes, nsteps):
    for ep in episodes:
        state = env.initialize()
        storage = Storage(nsteps, [])
        hidden = None
        for i in nsteps:

            prediction = model(state, hidden)
            storage.add(**prediction)
            hidden = prediction['hidden']
            if 'action' in prediction:
                state = env.step(prediction)


