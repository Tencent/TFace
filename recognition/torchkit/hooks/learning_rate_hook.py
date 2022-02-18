import logging
from bisect import bisect
from .base_hook import Hook


def set_optimizer_lr(optimizer, lr):
    if isinstance(optimizer, dict):
        backbone_opt, head_opts = optimizer['backbone'], optimizer['heads']
        for param_group in backbone_opt.param_groups:
            param_group['lr'] = lr
        for _, head_opt in head_opts.items():
            for param_group in head_opt.param_groups:
                param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def warm_up_lr(step, warmup_step, init_lr, optimizer):
    """ Warm up learning rate when batch step below warmup steps
    """

    lr = step * init_lr / warmup_step
    if step % 500 == 0:
        logging.info("Current step {}, learning rate {}".format(step, lr))

    set_optimizer_lr(optimizer, lr)


def adjust_lr(epoch, learning_rates, stages, optimizer):
    """ Decay the learning rate based on schedule
    """

    pos = bisect(stages, epoch)
    lr = learning_rates[pos]
    logging.info("Current epoch {}, learning rate {}".format(epoch + 1, lr))

    set_optimizer_lr(optimizer, lr)


class LearningRateHook(Hook):
    """ LearningRate Hook, adjust learning rate in training
    """
    def __init__(self,
                 learning_rates,
                 stages,
                 warmup_step):
        """ Create a ``LearningRateHook`` object

            Args:
            learning_rates: all learning rates value
            stages: learning rate adjust stages value
            warmup_step: step num of warmup
        """

        self.learning_rates = learning_rates
        self.stages = stages
        if len(self.learning_rates) != len(self.stages) + 1:
            raise RuntimeError("Learning_rates size should be one larger than stages size")
        self.init_lr = self.learning_rates[0]
        self.warmup_step = warmup_step

    def before_train_iter(self, task, step, epoch):
        global_step = epoch * task.step_per_epoch + step
        if self.warmup_step > 0 and global_step <= self.warmup_step:
            warm_up_lr(global_step, self.warmup_step, self.init_lr, task.opt)

    def before_train_epoch(self, task, epoch):
        adjust_lr(epoch, self.learning_rates, self.stages, task.opt)
