from __future__ import print_function
from __future__ import division
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torchkit.head.localfc.common import calc_logits


class CosFace(nn.Module):
    """ Implement of CosFace (https://arxiv.org/abs/1801.09414)

    """
    def __init__(self,
                 in_features,
                 out_features,
                 scale=64.0,
                 margin=0.40):
        """ Args:
            in_features: size of each input features
            out_features: size of each output features
            scale: norm of input feature
            margin: margin
        """
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.scale = scale
        self.margin = margin

        self.kernel = Parameter(torch.FloatTensor(in_features, out_features))
        # nn.init.xavier_uniform_(self.kernel)
        nn.init.normal_(self.kernel, std=0.01)
        # init.kaiming_uniform_(self.kernel, a=math.sqrt(5))

    def forward(self, embeddings, labels):
        cos_theta, origin_cos = calc_logits(embeddings, self.kernel)
        target_logit = cos_theta[torch.arange(0, embeddings.size(0)), labels].view(-1, 1)

        final_target_logit = target_logit - self.margin

        cos_theta.scatter_(1, labels.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.scale

        return output, origin_cos * self.scale
