import logging
from ..util import AverageMeter
from .base_hook import Hook


class LogHook(Hook):
    """ LogHook, print log info in training
    """
    def __init__(self, freq, rank):
        """ Create a LogHook object

            Args:
            freq: step interval
            rank: work rank in ddp
        """

        self.freq = freq
        self.rank = rank

    def before_train_epoch(self, task, epoch):
        task.log_buffer.clear()

    def after_train_iter(self, task, step, epoch):
        """ Print log info after every training step
        """

        if self.rank != 0:
            return
        if step == 0 or (step + 1) % self.freq == 0:
            time_cost = task.log_buffer['time_cost']
            logging.info("Epoch {} / {}, batch {} / {}, {:.4f} sec/batch".format(
                epoch + 1, task.epoch_num, step + 1, task.step_per_epoch, time_cost))

            log_str = " " * 25
            for k, v in task.log_buffer.items():
                if k == 'time_cost':
                    continue
                if isinstance(v, list):
                    s = ', '.join(['%.6f' % x.val for x in v])
                elif isinstance(v, AverageMeter):
                    s = '%.6f' % v.val
                else:
                    s = str(v)
                log_str += '%s = [%s] ' % (k, s)
            print(log_str)
