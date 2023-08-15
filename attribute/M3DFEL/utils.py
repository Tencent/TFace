import torch.nn as nn
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from einops import rearrange


def build_scheduler(args, optimizer, n_iter_per_epoch):
    """Build the scheduler
    The CosineLRScheduler schedules every iter in every epoch, so it needs n_iter_per_epoch instead of epochs

    Args:
        optimizer
        n_iter_per_epoch: 
    """
    num_steps = int(args.epochs * n_iter_per_epoch)
    warmup_steps = int(args.warmup_epochs * n_iter_per_epoch)

    # use the Cosine LR Scheduler for training
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=args.min_lr,
        warmup_lr_init=args.warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    return lr_scheduler


class DMIN(nn.Module):
    """The Dynamic Multi Instance Normalization Module
    This module dynamicly normlizes the instances and bags

    Args:
        optimizer
        n_iter_per_epoch:
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super(DMIN, self).__init__()

        # init the DMIN module
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))

        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):

        x = rearrange(x, 'b c n -> b n c')
        # calculate the mean and var of input
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        # caculated the weighted mean and var according to the learned weights
        mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
        var = var_weight[0] * var_in + var_weight[1] * var_ln

        # regulize x according to the calculated results
        x = (x-mean) / (var+self.eps).sqrt()
        x = x * self.weight + self.bias
        x = rearrange(x, 'b n c -> b c n')

        return x
