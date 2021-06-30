import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


class UncertaintyNet_conv(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()

        # self.z_dim = z_dim
        # self.convf_dim = convf_dim
        self.use_bn = use_bn
        self.depth = 512
        self.conv_ = nn.Sequential(
            nn.Conv2d(self.depth, self.depth, (3, 3), stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.depth, self.depth, (3, 3), stride=1, bias=False),
            nn.BatchNorm2d(self.depth), 
            nn.ReLU(inplace=True),
            nn.Conv2d(self.depth, self.depth, (3, 3), stride=1, bias=False),
            nn.BatchNorm2d(self.depth), 
            nn.AdaptiveAvgPool2d((1))
            )
        self._log_kappa = nn.Sequential(
                        nn.Linear(self.depth, self.depth // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.depth // 2, self.depth // 4),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.depth // 4, 1),
                        #nn.Softplus())
                        #nn.Sigmoid(),
        )

    def forward(self, convf):
        #kappa = self._kappa(convf) + 8528 #3991472 + 8528
        convf = self.conv_(convf)
        convf = convf.view(convf.shape[0], -1) 
        log_kappa = self._log_kappa(convf)

        log_kappa = torch.log(1e-6 + torch.exp(log_kappa))

        #kappa = self._kappa(convf) * 3790353 # exp(20) / (radius*2)

        return log_kappa #kappa
