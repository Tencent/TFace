import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CC(nn.Module):
    """ Correlation Congruence for Knowledge Distillation. ICCV 2019
    """
    def __init__(self, gamma=0.4, P_order=2):
        super().__init__()
        self.gamma = gamma
        self.P_order = P_order

    def forward(self, feat_s, feat_t):
        corr_mat_s = self.get_correlation_matrix(feat_s)
        corr_mat_t = self.get_correlation_matrix(feat_t)
        loss = F.mse_loss(corr_mat_s, corr_mat_t)
        return loss

    def get_correlation_matrix(self, feat):
        feat = F.normalize(feat)
        sim_mat = torch.matmul(feat, feat.t())
        corr_mat = torch.zeros_like(sim_mat)
        for p in range(self.P_order+1):
            corr_mat += math.exp(-2*self.gamma) * (2*self.gamma)**p / \
                        math.factorial(p) * torch.pow(sim_mat, p)
        return corr_mat
