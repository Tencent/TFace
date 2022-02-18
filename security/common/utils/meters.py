class AverageMeter(object):
    '''
        The Class AverageMeter record the metrics during the training process
        Examples:
            >>> acces = AverageMeter('_Acc', ':.5f')
            >>> acc = (prediction == labels).float().mean()
            >>> acces.update(acc)
    '''
    def __init__(self, name='metric', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    '''
        The ProgressMeter to record all AverageMeter and print the results
        Examples:
            >>> acces = AverageMeter('_Acc', ':.5f')
            >>> progress = ProgressMeter(epoch_size, [acces]) 
            >>> progress.display(iterations)
    '''
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
