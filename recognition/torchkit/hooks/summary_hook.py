from torch.utils.tensorboard.writer import SummaryWriter
from ..util import AverageMeter
from .base_hook import Hook


class SummaryHook(Hook):
    """ SummaryHook, write tensorboard summery in training
    """
    def __init__(self, log_root, freq, rank):
        """ Create A SummaryHook object

            Args:
            log_root: tensorboard summary root path
            freq: step interval
            rank: gpu rank
        """

        self.writer = SummaryWriter(log_root) if rank == 0 else None
        self.freq = freq

    def after_train_iter(self, task, step, epoch):
        if self.writer is None:
            return
        if step == 0 or (step + 1) % self.freq == 0:
            global_step = step + epoch * task.step_per_epoch
            scalars = task.summary.get('scalars', {})
            for k, v in scalars.items():
                if isinstance(v, list):
                    for i, x in enumerate(v):
                        self.writer.add_scalar('%s_%d' % (k, i), x.val,
                            global_step=global_step)
                elif isinstance(v, AverageMeter):
                    self.writer.add_scalar(k, v.val, global_step=global_step)

            histograms = task.summary.get('histograms', {})
            for k, v in histograms.items():
                if isinstance(v, list):
                    for i, x in enumerate(v):
                        self.writer.add_histogram('%s_%d' % (k, i), x.val,
                            global_step=global_step)
                elif isinstance(v, AverageMeter):
                    self.writer.add_histogram(k, v.val, global_step=global_step)

    def after_run(self, *args):
        if self.writer:
            self.writer.close()
