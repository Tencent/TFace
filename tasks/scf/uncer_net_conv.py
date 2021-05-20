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

        """
        self.fc1 = nn.Linear(convf_dim, convf_dim//2)
        self.bn1 = nn.BatchNorm1d(convf_dim//2, eps=1e-3)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(convf_dim//2, convf_dim//4)
        self.bn2 = nn.BatchNorm1d(convf_dim//4, eps=1e-3)
        self.fc3 = nn.Linear(convf_dim//4, 1)
        """

    def forward(self, convf):
        #kappa = self._kappa(convf) + 8528 #3991472 + 8528
        convf = self.conv_(convf)
        convf = convf.view(convf.shape[0], -1) 
        log_kappa = self._log_kappa(convf)
       
        """
        x = self.fc1(convf)
        if self.use_bn: x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        if self.use_bn: x = self.bn2(x)
        x = self.act(x)
        log_kappa = self.fc3(x)
        """

        log_kappa = torch.log(1e-6 + torch.exp(log_kappa))

        #kappa = self._kappa(convf) * 3790353 # exp(20) / (radius*2)

        return log_kappa #kappa
