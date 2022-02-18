from __future__ import print_function
from __future__ import division
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torchkit.head.localfc.common import calc_logits


class ArcFace(nn.Module):
    """ Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf)
    """
    def __init__(self,
                 in_features,
                 out_features,
                 scale=64.0,
                 margin=0.5,
                 easy_margin=False):
        """ Args:
            in_features: size of input features
            out_features: size of output features
            scale: scale of input feature
            margin: margin
        """
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.scale = scale
        self.margin = margin

        self.kernel = Parameter(torch.FloatTensor(in_features, out_features))
        # nn.init.xavier_uniform_(self.kernel)
        nn.init.normal_(self.kernel, std=0.01)
        # init.kaiming_uniform_(self.kernel, a=math.sqrt(5))

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        cos_theta, origin_cos = calc_logits(embeddings, self.kernel)
        target_logit = cos_theta[torch.arange(0, embeddings.size(0)), labels].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(target_logit > self.th, cos_theta_m, target_logit - self.mm)

        cos_theta.scatter_(1, labels.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.scale

        return output, origin_cos * self.scale
