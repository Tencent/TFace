import torch.nn as nn


class HintLoss(nn.Module):
    """ Fitnets: hints for thin deep nets, ICLR 2015
    """
    def __init__(self):
        super().__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss
