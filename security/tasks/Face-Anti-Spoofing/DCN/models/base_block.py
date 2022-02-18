import os
import sys
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..'))
from common.utils.model_init import init_weights


class Conv2dBlock(nn.Module):
    '''
        Args:
            First_block (bool): 
                'True': indicate the modules is the first block and the input channels is 64
    '''

    def __init__(self, First_block=False):
        super(Conv2dBlock, self).__init__()
        self.net = []
        if First_block:
            self.net += [nn.Conv2d(64, 128, 3, 1, 1, bias=True)]
        else:
            self.net += [nn.Conv2d(128, 128, 3, 1, 1, bias=True)]

        self.net += [nn.ELU()]
        self.net += [nn.BatchNorm2d(128)]

        self.net += [nn.Conv2d(128, 196, 3, 1, 1, bias=True)]
        self.net += [nn.ELU()]
        self.net += [nn.BatchNorm2d(196)]

        self.net += [nn.Conv2d(196, 128, 3, 1, 1, bias=True)]
        self.net += [nn.ELU()]
        self.net += [nn.BatchNorm2d(128)]

        self.net += [nn.AvgPool2d(2, stride=2)]

        self.net = nn.Sequential(*self.net)
        init_weights(self.net, init_type='kaiming')

    def forward(self, x):
        return self.net(x)
