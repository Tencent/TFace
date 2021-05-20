import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """ Implementaion of "https://arxiv.org/abs/1708.02002"
    """
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.func = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.func(input, target)
        prob = torch.exp(-logp)
        loss = (1 - prob) ** self.gamma * logp
        return loss.mean()
