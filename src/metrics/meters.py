import torchnet as tnt
import visdom
import numpy as np
import collections
from skimage.transform import resize
from src.model.arguments import Arguments
import matplotlib.pyplot as plt
import os


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


# -----------------------------------------------------------------------------------
class AllMeters(object):
    def __init__(self, env=None, title='', log_every=100, detail_every=500, arg_obj=None):
        self._title = title
        self._env = env
        self.log_every = log_every
        self.detail_every = detail_every
        self._img = None
        self._img2 = None
        self._img_dir = './data/img/{}/'.format(self._title)
        self.viz = visdom.Visdom()
        if isinstance(arg_obj, Arguments):
            # self._testing = True
            if arg_obj.train.testing is False:

                arg_txt = arg_obj.print()
                self.viz.text(arg_txt, opts=dict(title=self._title))

        if not os.path.exists(self._img_dir):
            os.makedirs(self._img_dir)
        self._mdict = {}
        self._meter_to_logger = {}
        self._ldict = {}

    def __init_loggers(self):
        self.duratn_logger = tnt.logger.VisdomPlotLogger(
            'line', env=self._env, opts={'title': self._title + ': duration'}
        )
        self._ldict['losses']  = tnt.logger.VisdomPlotLogger(
            'line', env=self._env, opts={'title': self._title + ': loss',
                                         'layoutopts': {'plotly': {
                                                 'yaxis': {
                                                     # 'type': 'log',
                                                      'range': [-5, 5],
                                                     'autorange': True,
                                                 }
                                             }
                                         }
                                         }
        )
        self._ldict['reward'] = tnt.logger.VisdomPlotLogger(
            'line', env=self._env, opts={'title': self._title + ': reward'}
        )
        self._ldict['action'] = tnt.logger.VisdomPlotLogger(
            'line', env=self._env, opts={'title': self._title + ': action'}
        )

    def __getitem__(self, item):
        # if item not in self._mdict:

        return self._mdict[item]

    def reset(self):
        for v in self._mdict.values():
            v.reset()

    def values(self, *names):
        return [self._mdict[n].value()[0] for n in names]

    def add_meters(self, *names, cls=tnt.meter.AverageValueMeter):
        for n in names:
            self._mdict[n] = cls()

    def add_logger(self, name, type='line', data=[]):
        self._meter_to_logger[name] = data
        self._ldict[name] = tnt.logger.VisdomPlotLogger(
            type,
            opts={'title': self._title + ': ' + name,
                  'legend': data }
        )

    @property
    def M(self):
        return self._mdict

    @property
    def L(self):
        return self._ldict

    def log(self, episode, chart, meters):
        self._ldict[chart].log(episode, (self.values(meters)))
        return

    def log_detailed_episode(self, storage, env, show=True):
        # for i in range(0, len(storage.state)):
        #     action = storage.action_index[i]
        #     acc = None if storage.action[i] is None else np.round(storage.action[i][1], 2)
        #     feats = np.sum(storage.feats[i], -1)
        #     print(action, acc, storage.legal[i], feats, storage.reward[i])
        #
        # batch = [resize(env.to_image(x), (3, 100, 100), anti_aliasing=False) for x in storage.state]
        # deltas = []
        # for i in range(1, len(batch)):
        #     delta = np.abs(batch[i] - batch[i+1])
        #     deltas.append(delta * (1 / np.max(delta)))
        #
        # # batch = np.stack(batch)
        # # deltas =
        # batch = np.concatenate([np.stack(batch), np.stack(deltas)], -1) # cat along Y axis
        # self._img = self.viz.images(batch, win=self._img)
        return

    def log_episode(self, episode, last_step, last_state, storage, env):
        self['duration'].add(last_step)
        self['done'].add(int(last_state['done']))

        for i in range(0, len(storage.state)):
            self['legal'].add(int(storage['legal'][i]))
            self['reward'].add(storage['reward'][i])
            self['best'].add(storage['reward'][i])
            if storage[i]['action_index'] is not None:
                self['actions'].add(storage['action_index'][i])

        self['best_avg'].add(storage['reward'].max())
        # log codes - which constraints are being solved
        # Final state objectives achieved
        # for i, k in enumerate(self.env._objective.keys):
        #    self.M['a_' + k ].add(np.mean(state_data['feats'][:, i] ))
        if episode % self.detail_every == 0:
            self.log_detailed_episode(storage, env)

        if episode % self.log_every == 0:
            self.losses_logger.log(
                episode, self.values('loss', 'policy_loss', 'value_loss', 'entropy')
            )
            self.reward_logger.log(episode, self.values('best', 'best_avg', 'reward', 'legal'))
            self.action_logger.log(episode, self['actions'].value())

            # self.M.duratn_logger.log(self._episode, self.M['duration'].value())
            print('episode {}, best: {}, {}'.format(episode, self.values('best'), self['actions'].value()))
            # self._actions = [0] * self.action_size
            # self._step = 0
            self.reset()


# ----------------------------------------------------------------------------------------
class SupervisedMeter(AllMeters):
    def __init__(self, **kwargs):
        AllMeters.__init__(self, **kwargs)
        meters = ['policy_loss', 'reward_loss', 'recon_loss', 'loss']
        meters_acc = ['error_index', 'error_geom']
        self.add_meters(*meters + meters_acc, cls=tnt.meter.AverageValueMeter)
        self.add_logger('loss', data=meters)
        self.add_logger('action', data=meters_acc)

    def log_detailed_episode(self, step, data):
        print(step, data)

    def np(self, xs):
        if isinstance(xs, (list, tuple)):
            return np.concatenate([self.np(x) for x in xs])
        elif isinstance(xs, (float, int)):
            return xs
        return xs.cpu().detach().squeeze().numpy()

    def log_step(self, step, loss_a, loss_r, loss_s, action_hat, action):
        # print(action_hat.shape)
        # action_hat = self.np(action_hat)
        # action_tgt = self.np(action)
        # action_hat[1:] *= 20
        # action_tgt[1:] *= 20
        # action_hat[0] *= 3
        # action_tgt[0] *= 3
        # a_tgt, a_pred = action_tgt.astype(int), action_hat.astype(int)

        # acc_action = np.abs(action_hat - action)
        # self.M['error_index'].add(1 if acc_action[0] > 0 else 0)
        # self.M['error_geom'].add(acc_action[1:].sum())

        self.M['policy_loss'].add(self.np(loss_a))
        self.M['reward_loss'].add(self.np(loss_r))
        self.M['recon_loss'].add(self.np(loss_s))
        self.M['loss'].add(self.np(loss_a + loss_r + loss_s))

        #if step % self.detail_every == 0:
        #    self.log_detailed_episode(step, [a_tgt, a_pred])

        if step % self.log_every == 0:
            self.L['loss'].log(step, self.values('policy_loss', 'reward_loss', 'recon_loss', 'loss'))
            self.L['action'].log(step, self.values('error_index', 'error_geom'))
            self.reset()


class Meters2(AllMeters):
    def __init__(self, **kwargs):
        AllMeters.__init__(self, **kwargs)
        self.add_logger('stats', data=['advantage', 'gae', 'log_prob'])
        self.add_logger('losses', data=['loss', 'policy_loss', 'aux_loss', 'value_loss', 'entropy'])
        self.add_logger('reward', data=['best', 'best_avg', 'reward'])

    def log_detailed_episode(self, episode, storage, env, show=True):
        """ todo
            log details at each step every ~500 episodes
            What were the activations
        """
        size = (3, 100, 100)
        batch = [resize(env.to_image(storage.action[i]), size, anti_aliasing=False) for i in range(len(storage.action))]
        # deltas = []
        deltas = []
        deltas_hard = []
        for i in range(len(batch)):
            x = storage.x[i]
            delta = ((x + x.min())/ (x.max()+ x.min())).numpy()
            deltas.append(resize(delta, size, anti_aliasing=False))
            deltas_hard.append(resize(env.to_image(delta), size, anti_aliasing=False))
            # deltas.append(np.multiply(delta , (1 / np.max(delta))))
            # feats = None
            if i < len(storage.feats):
                feats = np.round(np.sum(storage.feats[i], -1), 3)
                print(feats, np.round(storage.reward[i], 4), x.min().item(), x.mean().item(), x.max().item())

        batch = np.stack(batch)
        deltas = np.stack(deltas)
        deltas_hard = np.stack(deltas_hard)
        print(batch.shape, deltas.shape)
        s = batch.shape
        buf = np.ones((s[0], s[1], 10, s[3]))
        batch = np.clip(np.concatenate([batch, buf, deltas, buf, deltas_hard], -2), 0, 1)  # cat along Y axis
        self._img = self.viz.images(batch, win=self._img, opts=dict(title='state-' +self._title), nrow=batch.shape[0])
        # todo save image
        # img = np.zeros((3, batch.shape[0]*batch.shape[2], batch.shape[3]))
        # img = batch.reshape((-1, batch.shape[3], batch.shape[1]))
        # print(batch.shape, img.shape, )
        # plt.imsave(self._img_dir + 'ep_{}'.format(episode), img)

    def log_episode(self, episode, last_step, last_state, storage, env):
        self['best_avg'].add(max(storage['reward']))
        self['done'].add(int(last_state['done']))
        self['duration'].add(last_step)

        for i in range(0, len(storage.state)):
            self['legal'].add(int(storage['legal'][i]))
            self['reward'].add(storage['reward'][i])
            self['best'].add(storage['reward'][i])

        if episode % self.detail_every == 0:
            self.log_detailed_episode(episode, storage, env)

        if episode % self.log_every == 0:
            self.L['losses'].log(episode, self.values('loss', 'policy_loss', 'aux_loss', 'value_loss', 'entropy'))
            self.L['reward'].log(episode, self.values('best', 'best_avg', 'reward'))
            self.L['stats'].log(episode,  self.values('advantage', 'gae', 'log_prob'))
            print('episode {}, best: {}'.format(episode, self.values('best')))
            self.reset()



