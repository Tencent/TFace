import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """ Distilling the Knowledge in a Neural Network, NIPSW 2015
    """
    def __init__(self, t=4):
        super().__init__()
        self.t = t

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.t, dim=1)
        p_t = F.softmax(y_t/self.t, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.t**2) / y_s.shape[0]
        return loss
