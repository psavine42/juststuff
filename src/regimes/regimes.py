import torch
from src.algo.grg import *
import torch.distributions as D


class GBPTrainer(Trainer):
    def __init__(self, env, optimizer=None, model=None, argobj=None, **kwargs):
        Trainer.__init__(self, env, **kwargs)
        self.model = model.to(self.device)
        self._episode = 0
        self._step = 1
        self._actions = [0] * self.env.state.shape[0]
        self._instances = []
        self._solutions = []
        self.arg_obj = argobj
        self.make_meters()
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=kwargs['lr'])
        self.model.train()

        import torch.backends.cudnn
        torch.backends.cudnn.deterministic = True
        self._img = None
        self.softmax = nn.Softmax()

    def make_meters(self):
        self.M = SupervisedMeter(
            detail_every=self.arg_obj.train.detail_every,
            log_every=self.arg_obj.train.log_every,
            arg_obj=self.arg_obj,
            title=self._title,
            env=None,
        )

    def optimize_model(self, R, storage, args):
        policy_loss = 0
        value_loss = 0
        # advantage = 0
        gae = torch.zeros(1, 1, device=self.device)

        for i in reversed(range(len(storage.reward))):
            R = args['gamma'] * R + storage.reward[i]
            advantage = R - storage.value[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # td loss

            # prediction loss

            # Generalized Advantage Estimation
            delta_t = storage.reward[i] + args['gamma'] * storage.value[i + 1] - storage.value[i]
            gae = (gae * args['gamma'] * args['gae_lambda'] + delta_t).detach()

            # todo STORAGE.LOG_PROB IS TOO DAMN HIGh!!
            # todo SHOULD BE BETWEEN 0-1 AND LOW
            policy_loss = policy_loss - storage.log_prob[i] * gae - args['entropy_coef'] * storage.entropy[i]

        # additional losses
        feats_tgt = torch.from_numpy(np.stack(storage.feats)).to(self.device).float()
        # print(feats_tgt.size(), torch.cat(storage.aux).size())
        auxilary_loss = F.mse_loss(feats_tgt, torch.cat(storage.aux)[:-1])

        self.optimizer.zero_grad()
        loss = policy_loss + args['value_loss_coef'] * value_loss + args['aux_loss_coef'] * auxilary_loss
        # print(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), args['max_grad_norm'])
        self.optimizer.step()

        # -------------------
        self.M['loss'].add(loss.item())
        self.M['entropy'].add(torch.stack(storage.entropy).mean().item())
        self.M['aux_loss'].add(auxilary_loss.item())
        self.M['advantage'].add(advantage.item())
        self.M['log_prob'].add(torch.stack(storage.log_prob).mean().item())
        self.M['gae'].add(gae.item())
        self.M['policy_loss'].add(policy_loss.item())
        self.M['value_loss'].add(value_loss.item())

    def state_to_observation(self, state_data):
        keys = ['state', 'code']  # , 'feats'
        return self._to_tensor(tuple([state_data[k] for k in keys]))

    def prediction_to_action(self, prediction):
        # prediction['action_index'],
        return prediction['action'].squeeze(0).cpu().numpy()

    def gbp_step(self, state, args):
        """ algorithm 1 GBP

        rollouts are simulated giving


        Returns
            Action sequence A_k and state trajectory S_k for which R k is largest.
        """
        # result.diagonal(dim1=1, dim2=2).normal_().mul_(sigma)
        # diagonal proirs https://github.com/pytorch/pytorch/pull/11178
        # result_seq = Storage(args.num_rollouts, ['action', 'reward', 'state'])

        init_state = state
        best_states = [init_state] + [None] * args.num_steps
        best_action = [None] * args.num_steps
        best_reward = torch.zeros(1, 1, device=self.device)

        action_noise = D.Independent(D.Normal(0, args.sigma), self.action_size)
        for k in range(1, args.num_rollouts):

            prev_state = init_state.copy().detach()
            for i in range(1, args.num_grad_steps):

                states = [prev_state] + [None] * args.num_steps
                action = [None] * args.num_steps
                reward = torch.zeros(args.num_steps, device=self.device)
                noise = action_noise.sample()

                for t in range(1, args.num_steps):
                    action[t] = self.softmax(noise)
                    states[t] = self.model.transition(prev_state, action[t])
                    reward[t] = self.model.reward(prev_state, action[t])
                    prev_state = states[t].detach()

                reward = reward.sum()
                # finish grad step - advantages etc
                for t in reversed(range(args.num_steps)):
                    # xt  = noises_x[t] -0    # todo
                    action[t] = self.softmax(action[t])

                # return option with best reward
                if reward > best_reward:
                    best_states = states
                    best_action = action

        return best_states, best_action

    def train_vanilla_supervised(self, dataset, args):
        """ vanilla version of all actions trainer"""
        step = 0
        for epoch in args.episodes:
            for datum in dataset:

                state = self.state_to_observation(datum)
                action = self._to_tensor(datum['action'])
                next_state = self._to_tensor(datum['next_state'])
                reward = self._to_tensor(datum['reward'])

                #
                action_hat, next_state_hat, reward_hat = \
                    self.model.transition(state, action)

                loss_a = F.mse_loss(action, action_hat)
                loss_r = F.mse_loss(reward, reward_hat)
                loss_s = F.binary_cross_entropy(next_state, next_state_hat)
                self.optimizer.zero_grad()
                (loss_a + loss_r + loss_s).backward()
                self.optimizer.step()

                self.M.log_step(step, loss_a, loss_r, loss_s, action_hat, action)
                step += 1
            print('epoch {}'.format(epoch))

    def train_distGBP(self, dataset, args):
        """
        Algorithm 2
        train to predict just the value
        """
        opt_dataset = Storage(args.num_steps * args.episodes)
        for episode in range(args.episodes):
            sample = dataset.sample()
            best_states, best_actions = self.gbp_step(sample['state'], args)
            for j in range(args.num_steps):
                dataset.add({'state':best_states[j], 'action': best_actions[j]})

        for step in range(args.steps):
            sample = dataset.sample()
            action = self.model(sample['state'])
            self.optimizer.zero_grad()
            loss = F.mse_loss(sample['action'], action)
            loss.backward()
            self.optimizer.step()
        return dataset

    def train(self, epochs, args):
        dataset = self.build_dataset()
        trials = [1, 2, 5, 10, 20, 40]
        for epoch in range(epochs):
            dataset = self.train_distGBP(dataset, args)

            # save_model()

