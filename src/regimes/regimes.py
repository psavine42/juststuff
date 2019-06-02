import torch
from src.algo.grg import *
import torch.distributions as D
import torch.autograd
from src.probablistic.utils import *




class GBPTrainer(Trainer):
    def __init__(self, env, optimizer=None, model=None, argobj=None, **kwargs):
        Trainer.__init__(self, env, **kwargs)
        self.model = model.to(self.device)
        self._episode = 0
        self._step = 1
        # self._actions = [0] * self.env.state.shape[0]
        # self._instances = []
        self._solutions = []
        self.args = argobj
        self._lr = argobj.train.lr
        self._title = argobj.title
        self.make_meters()
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self._lr)
        self.model.train()

        import torch.backends.cudnn
        torch.backends.cudnn.deterministic = True
        self._img = None
        self.softmax = nn.Softmax()
        self.action_model = None

    def make_meters(self):
        self.M = SupervisedMeter(
            detail_every=self.args.train.detail_every,
            log_every=self.args.train.log_every,
            arg_obj=self.args,
            title=self._title,
            env=None,
        )

    def state_to_observation(self, state_data):
        keys = ['state', 'code']  # , 'feats'
        return self._to_tensor(tuple([state_data[k] for k in keys]))

    def prediction_to_action(self, prediction):
        """

        """
        ac = prediction['action']
        if isinstance(ac, tuple):
            pass
        return prediction['action'].squeeze(0).cpu().numpy()

    def _action_sig(self, tnsr_action):
        return torch.clamp(tnsr_action, min=0, max=1)

    def gbp_step(self, state, args):
        """ algorithm 1 GBP

        rollouts are simulated giving


        Returns
            Action sequence A_k and state trajectory S_k for which R k is largest.
        """
        # result.diagonal(dim1=1, dim2=2).normal_().mul_(sigma)
        # diagonal proirs https://github.com/pytorch/pytorch/pull/11178
        # result_seq = Storage(args.num_rollouts, ['action', 'reward', 'state'])
        # from src.actions.action_models import CoordVec, OneHot

        init_state = state
        best_states = []  # [init_state] + [None] * args.num_steps
        best_action = []  # [None] * args.num_steps
        best_reward = -1e6

        # initialize noise
        action_noise = []
        for i in range(len(self.action_model)):
            child = self.action_model.children[i]
            sz = child.size()
            # if isinstance(child, CoordVec):
            am = D.Normal(torch.zeros(sz).fill_(0.5), torch.zeros(sz).fill_(args.sigma ** 2))
            action_noise.append(am)
            # elif isinstance(child, OneHot):
            #   am = D.Categorical(torch.zeros(sz).fill_(0.5), torch.zeros(sz).fill_(args.sigma ** 2))
            #    action_noise.append(am)

        # action_noise = D.Normal(torch.zeros(5).fill_(0.5), torch.zeros(5).fill_(args.sigma ** 2))
        for k in range(1, args.num_rollouts):

            # sample with a list so that each x has its own grad ...
            xs = [[an.sample(torch.Size([1])).clone().detach().requires_grad_(True).to(self.device)
                   for an in action_noise]
                  for _ in range(args.num_steps)]

            # optimize the noisy actions in for some grad steps
            for i in range(1, args.num_grad_steps):

                states = [init_state] + [None] * args.num_steps
                action = [None] * args.num_steps
                reward = [None] * args.num_steps

                for t in range(1, args.num_steps):
                    action[t] = self.action_model.regularize(xs[t])
                    states[t] = (self.model.transition(states[t-1], self.action_model.regularize(xs[t])),
                                 init_state[1])
                    reward[t] = self.model.reward(states[t-1], self.action_model.regularize(xs[t]))

                # calculate final reward - this can be better ???
                reward = torch.cat(reward[1:]).sum()

                # finish grad step - advantages etc
                for t in reversed(range(1, args.num_steps)):
                    # x t ← x_t − η (∂R / ∂x)
                    # ã t ← σ( x t )
                    # eta * partial derivative of reward w.r.t. x_t
                    # an update to the noise and to the action
                    if torch.is_tensor(xs[t]):
                        xs[t] = xs[t] - self._lr * dfdx(reward, xs[t])
                    else:
                        for j in range(len(xs[t])):
                            xs[t][j] = xs[t][j] - self._lr * dfdx(reward, xs[t][j])

                    action[t] = self.action_model.regularize(xs[t])

                # return option with best reward
                if reward.item() > best_reward:
                    best_states = states
                    best_action = action

        # todo
        # log optimized action reward best
        #
        return best_states, best_action

    def train_distGBP(self, dataset, args):
        """
        Algorithm 2
        train to predict just the value
        """
        for episode in range(args.action_steps):
            # action optimization (TR) episode
            sample = self.state_to_observation(dataset[episode])
            best_states, best_actions = self.gbp_step(sample, args)
            for j in range(1, args.num_steps):
                dataset.append({'state': best_states[j], 'action': best_actions[j]})

        print('starting policy_training ')
        for step in range(args.policy_steps):
            # transition-reward (TR) episode
            sample = dataset[step]
            state = self.state_to_observation(sample)
            target_action = self._to_tensor(sample['action'])

            action, ns, nr = self.model(state)
            # print(size(action))

            self.optimizer.zero_grad()
            loss = self.model.policy.loss(action, target_action)
            loss.backward()
            self.optimizer.step()

        return dataset

    def tr_supervised_episode(self, dataset, args=None, step=0):
        steps = len(dataset)
        if args is not None and 'steps' in args:
            steps = args.steps

        for i in range(steps):
            datum = dataset[i]
            state = self.state_to_observation(datum)

            # ack = self.io_model.
            action = self._to_tensor(datum['action'])
            next_state = self._to_tensor(datum['next_state'])
            reward = self._to_tensor(datum['reward'])

            # forward
            reward_hat = self.model.reward(state, action)
            next_state_hat = self.model.transition(state, action)

            # loss
            loss_r = F.mse_loss(reward_hat, reward)
            loss_s = F.mse_loss(next_state_hat, next_state)
            loss = loss_s + loss_r

            # backward
            # todo add grad clip ?
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.M.log_step(step, 0, loss_r, loss_s, action, action)
            step += 1

    def train_vanilla_supervised(self, dataset, args):
        """ vanilla version of all actions trainer"""
        step = 0
        steps = len(dataset)
        if args is not None and 'steps' in args:
            steps = args.steps
        print(self._title, len(dataset))
        for epoch in range(args.episodes):
            for i in range(steps):
                datum = dataset[i]
                state = self.state_to_observation(datum)
                action = self._to_tensor(datum['action'])
                next_state = self._to_tensor(datum['next_state'])
                reward = self._to_tensor(datum['reward'])

                # forward
                action_hat, next_state_hat, reward_hat = self.model(state)

                # loss
                # print(size(action), size(action_hat))
                loss_a = self.model.policy.loss(action_hat, action)
                loss_r = F.mse_loss(reward_hat, reward)
                loss_s = F.mse_loss(next_state_hat, next_state)

                # backward
                self.optimizer.zero_grad()
                (loss_a + loss_r + loss_s).backward()
                self.optimizer.step()

                self.M.log_step(step, loss_a, loss_r, loss_s, action_hat, action)
                step += 1

        self.save_model()

    def back_prop_loss_step(self, datum, step):
        state = self.state_to_observation(datum)
        action = self._to_tensor(datum['action'])
        next_state = self._to_tensor(datum['next_state'])
        reward = self._to_tensor(datum['reward'])

        # forward
        action_hat, next_state_hat, reward_hat = self.model(state)

        # loss
        # print(size(action), size(action_hat))
        loss_a = self.model.policy.loss(action_hat, action)
        loss_r = F.mse_loss(reward_hat, reward)
        loss_s = F.mse_loss(next_state_hat, next_state)

        # backward
        self.optimizer.zero_grad()
        (loss_a + loss_r + loss_s).backward()
        self.optimizer.step()
        self.M.log_step(step, loss_a, loss_r, loss_s, action_hat, action)

    def train_tr_supervised(self, dataset, args):
        """ vanilla version of all actions trainer"""
        step = 0
        print(self._title, len(dataset))
        for epoch in range(args.episodes):
            self.tr_supervised_episode(dataset)
        self.save_model()

    # ----------------------------------------------------------
    def train_semi_supervised(self, dataset, args):
        """
        vanilla version of all actions trainer

        """
        step = 0
        print(self._title, len(dataset))
        for epoch in range(args.episodes):
            for datum in dataset:

                state = self.state_to_observation(datum)
                action = self._to_tensor(datum['action'])
                next_state = self._to_tensor(datum['next_state'])
                reward = self._to_tensor(datum['reward'])

                # forward
                action_hat, next_state_hat, reward_hat = self.model(state)

                # loss
                # static eval to compute next state
                # static eval to compute best reward given a[0]
                #

                loss_a = self.model.policy.loss(action_hat, action)
                loss_r = F.mse_loss(reward_hat, reward)
                loss_s = F.mse_loss(next_state_hat, next_state)

                # backward
                self.optimizer.zero_grad()
                (loss_a + loss_r + loss_s).backward()
                self.optimizer.step()

                self.M.log_step(step, loss_a, loss_r, loss_s, self.model.policy.predict_box(action_hat), action)
                step += 1

        self.save_model()

    def train(self, dataset, epochs, args):
        trials = [1, 2, 5, 10, 20, 40]
        for epoch in range(epochs):
            dataset = self.train_distGBP(dataset, args)

            # save_model()

