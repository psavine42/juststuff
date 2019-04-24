import torchnet as tnt
import visdom


class AllMeters(object):
    def __init__(self, title=''):
        self._title = title
        self.viz = visdom.Visdom()

        self.advantage_meter = tnt.meter.AverageValueMeter()
        self.qval_meter = tnt.meter.AverageValueMeter()
        self.loss_meter = tnt.meter.AverageValueMeter()
        self.reward_meter = tnt.meter.AverageValueMeter()
        self.duration_meter = tnt.meter.AverageValueMeter()
        self.improved_meter = tnt.meter.AverageValueMeter()
        self.explore_exploit = tnt.meter.AverageValueMeter()
        self.max_reward_meter = tnt.meter.AverageValueMeter()
        self.fail_meter = tnt.meter.AverageValueMeter()

        self.duratn_logger = tnt.logger.VisdomPlotLogger(
            'line', opts={'title': self._title + ': duration'}
        )
        self.losses_logger = tnt.logger.VisdomPlotLogger(
            'line', opts={'title': self._title + ': loss'}
        )
        self.reward_logger = tnt.logger.VisdomPlotLogger(
            'line', opts={'title': self._title + ': reward'}
        )
        self.action_logger = tnt.logger.VisdomPlotLogger(
            'line', opts={'title': self._title + ': action'}
        )

    def reset(self):
        self.loss_meter.reset()
        self.reward_meter.reset()
        self.explore_exploit.reset()
        self.improved_meter.reset()
        self.duration_meter.reset()
        self.max_reward_meter.reset()
        self.qval_meter.reset()
        self.advantage_meter.reset()
        self.fail_meter.reset()
