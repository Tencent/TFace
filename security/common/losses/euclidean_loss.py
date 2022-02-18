import torch
import torch.nn as nn


class EuclideanLoss(nn.Module):
    '''Compute euclidean distance between two tensors
    '''

    def __init__(self, reduction=None):
        super(EuclideanLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):

        n = x.size(0)
        m = n
        d = x.size(1)
        y = y.unsqueeze(0).expand(n, d)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        if self.reduction == 'mean':
            return torch.pow(x - y, 2).mean()

        elif self.reduction == 'sum':
            return torch.pow(x - y, 2).sum()
