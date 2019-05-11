import torchnet as tnt
import visdom
import numpy as np
import collections
from skimage.transform import resize


class BestValueMeter(object):
    def __init__(self, max=True):
        super(BestValueMeter, self).__init__()
        self._max = max
        self.val = 0
        self.reset()

    def add(self, value, n=1):
        self.val = max(self.val, value) if self._max else min(self.val, value)

    def value(self):
        return (self.val, )

    def reset(self):
        self.val = -np.inf if self._max else np.inf


class CounterMeter(object):
    def __init__(self, num=0):
        self._num = num
        self._val = collections.OrderedDict([(x, 0) for x in range(num + 1)])
        self.step = 0
        self.reset()

    def add(self, value, n=1):
        self.step += 1
        if value in self._val:
            self._val[value] += 1
        else:
            self._val[-1] += 1

    def value(self):
        return [v / self.step for v in self._val.values()]

    def reset(self):
        self.step = 0
        self._val = {x: 0 for x in range(self._num + 1)}


class AllMeters(object):
    def __init__(self, env=None, title='', log_every=100):
        self._title = title
        self._env = env
        self.log_every = log_every
        self.detail_every = 500
        self._img = None
        self.viz = visdom.Visdom()

        self.duratn_logger = tnt.logger.VisdomPlotLogger(
            'line', env=self._env, opts={'title': self._title + ': duration'}
        )
        self.losses_logger = tnt.logger.VisdomPlotLogger(
            'line', env=self._env, opts={'title': self._title + ': loss'}
        )
        self.reward_logger = tnt.logger.VisdomPlotLogger(
            'line', env=self._env, opts={'title': self._title + ': reward'}
        )
        self.action_logger = tnt.logger.VisdomPlotLogger(
            'line', env=self._env, opts={'title': self._title + ': action'}
        )
        self._mdict = {}
        self._ldict = {'action': self.action_logger,
                       'losses': self.losses_logger,
                       'duration': self.duratn_logger,
                       'reward': self.reward_logger}

    def __getitem__(self, item):
        return self._mdict[item]

    def reset(self):
        for v in self._mdict.values():
            v.reset()

    def values(self, *names):
        return [self._mdict[n].value()[0] for n in names]

    def add_meters(self, *names, cls=tnt.meter.AverageValueMeter):
        for n in names:

            self._mdict[n] = cls()

    def add_loggers(self, *names, type='line'):
        for n in names:
            self._ldict[n] = tnt.logger.VisdomPlotLogger(
                type, opts={'title': self._title + ': ' + n}
        )

    def log(self, episode, chart, meters):
        self._ldict[chart].log(episode, (self.values(meters)))
        return

    def log_detailed_episode(self, storage, env, show=True):
        for i in range(0, len(storage.state)):
            action = storage.action_index[i]
            acc = None if storage.action[i] is None else np.round(storage.action[i][1], 2)
            feats = np.sum(storage.feats[i], -1)
            print(action, acc, storage.legal[i], feats, storage.reward[i])

        batch = np.stack([resize(env.to_image(x), (3, 100, 100), anti_aliasing=False) for x in storage.state])
        self._img = self.viz.images(batch, win=self._img)

    def log_episode(self, episode, last_step, last_state, storage, env):
        self['duration'].add(last_step)
        self['done'].add(int(last_state['done']))

        for i in range(0, len(storage.state)):
            self['legal'].add(int(storage[i]['legal']))
            self['reward'].add(storage[i]['reward'])
            self['best'].add(storage[i]['reward'])
            if storage[i]['action_index'] is not None:
                self['actions'].add(storage[i]['action_index'])

        # log codes - which constraints are being solved
        # Final state objectives achieved
        # for i, k in enumerate(self.env._objective.keys):
        #    self.M['a_' + k ].add(np.mean(state_data['feats'][:, i] ))
        if episode % self.detail_every == 0:
            self.log_detailed_episode(storage, env)

        if episode % self.log_every == 0:
            self.losses_logger.log(episode, self.values('loss', 'policy_loss', 'value_loss', 'entropy'))
            self.reward_logger.log(episode, self.values('best', 'best_avg', 'reward', 'legal'))
            self.action_logger.log(episode, self['actions'].value())

            # self.M.duratn_logger.log(self._episode, self.M['duration'].value())
            print('episode {}, best: {}, {}'.format(episode, self.values('best'), self['actions'].value()))
            # self._actions = [0] * self.action_size
            # self._step = 0
            self.reset()

