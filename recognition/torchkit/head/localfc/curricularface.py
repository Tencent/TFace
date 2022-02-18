from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.nn import Parameter
import math
from torchkit.util.utils import l2_norm
from torchkit.head.localfc.common import calc_logits


class CurricularFace(nn.Module):
    """ Implement of CurricularFace (https://arxiv.org/abs/2004.00288)
    """

    def __init__(self,
                 in_features,
                 out_features,
                 scale=64.0,
                 margin=0.5,
                 alpha=0.1):
        """ Args:
            in_features: size of each input features
            out_features: size of each output features
            scale: norm of input feature
            margin: margin
        """
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.scale = scale
        self.alpha = alpha
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embeddings, labels):
        cos_theta, origin_cos = calc_logits(embeddings, self.kernel)
        target_logit = cos_theta[torch.arange(0, embeddings.size(0)), labels].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * self.alpha + (1 - self.alpha) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, labels.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.scale

        return output, origin_cos * self.scale
