import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import ReduceOp


class DistCrossEntropyFunc(torch.autograd.Function):
    """ CrossEntropy loss is calculated in parallel, allreduce all logits into single gpu and calculate loss.
        Implemented of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    @staticmethod
    def forward(ctx, logit_part, part_labels):
        ctx.batch_size = logit_part.size(0)
        # for numerical stability
        logit_part_max, _ = torch.max(logit_part, dim=1, keepdim=True)
        dist.all_reduce(logit_part_max, ReduceOp.MAX)
        logit_part = logit_part - logit_part_max

        # get exp sum
        exp_logit = torch.exp(logit_part)
        exp_sum = torch.sum(exp_logit, dim=1, keepdim=True)
        torch.distributed.all_reduce(exp_sum, ReduceOp.SUM)
        log_exp_sum = torch.log(exp_sum)
        log_softmax = logit_part - log_exp_sum
        ctx.log_softmax = log_softmax
        index = torch.where(part_labels != -1)[0]
        label_mask = torch.zeros(index.size()[0],
                                 log_softmax.size()[1],
                                 device=log_softmax.device)
        label_mask.scatter_(1, part_labels[index], 1)
        ctx.label_mask = label_mask
        ctx.index = index

        loss = torch.zeros(log_softmax.size()[0], 1, device=log_softmax.device)
        loss[index] = log_softmax[index].gather(1, part_labels[index])
        torch.distributed.all_reduce(loss, ReduceOp.SUM)
        loss = -torch.mean(loss)

        return loss

    @staticmethod
    def backward(ctx, loss_grad):
        """
        Args:
            loss_grad (torch.Tensor): gradient backward by last layer
        Returns:
            gradients for each input in forward function
            `None` gradients for one-hot label
        """
        logit_grad = torch.exp(ctx.log_softmax)
        logit_grad[ctx.index] -= ctx.label_mask
        logit_grad = loss_grad.item() * logit_grad / ctx.batch_size
        return logit_grad, None


class DistCrossEntropy(nn.Module):
    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(self, logit_part, label_part):
        return DistCrossEntropyFunc.apply(logit_part, label_part)
