from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.distributed as dist
import math
from torchkit.util.utils import all_gather_tensor
from torchkit.head.localfc.common import calc_logits


class Cifp(nn.Module):
    """ Implement of  (CVPR2021 Consistent Instance False Positive Improves Fairness in Face Recognition)
    """

    def __init__(self,
                 in_features,
                 out_features,
                 scale=64.0,
                 margin=0.35):
        """ Args:
            in_features: size of each input features
            out_features: size of each output features
            scale: norm of input feature
            margin: margin
        """
        super(Cifp, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embeddings, label):
        cos_theta, origin_cos = calc_logits(embeddings, self.kernel)
        cos_theta_, _ = calc_logits(embeddings, self.kernel.detach())

        mask = torch.zeros_like(cos_theta)
        mask.scatter_(1, label.view(-1, 1).long(), 1.0)

        sample_num = embeddings.size(0)
        tmp_cos_theta = cos_theta - 2 * mask
        tmp_cos_theta_ = cos_theta_ - 2 * mask
        target_cos_theta = cos_theta[torch.arange(0, sample_num), label].view(-1, 1)
        target_cos_theta_ = cos_theta_[torch.arange(0, sample_num), label].view(-1, 1)

        target_cos_theta_m = target_cos_theta - self.margin

        far = 1 / (self.out_features - 1)
        # far = 1e-4
        topk_mask = torch.greater(tmp_cos_theta, target_cos_theta)
        topk_sum = torch.sum(topk_mask.to(torch.int32))
        dist.all_reduce(topk_sum)
        far_rank = math.ceil(far * (sample_num * (self.out_features - 1) * dist.get_world_size() - topk_sum))
        cos_theta_neg_topk = torch.topk((tmp_cos_theta - 2 * topk_mask.to(torch.float32)).flatten(), k=far_rank)[0]
        cos_theta_neg_topk = all_gather_tensor(cos_theta_neg_topk.contiguous())
        cos_theta_neg_th = torch.topk(cos_theta_neg_topk, k=far_rank)[0][-1]

        cond = torch.mul(torch.bitwise_not(topk_mask), torch.greater(tmp_cos_theta, cos_theta_neg_th))
        _, cos_theta_neg_topk_index = torch.where(cond)
        cos_theta_neg_topk = torch.mul(cond.to(torch.float32), tmp_cos_theta)
        cos_theta_neg_topk_ = torch.mul(cond.to(torch.float32), tmp_cos_theta_)

        cond = torch.greater(target_cos_theta_m, cos_theta_neg_topk)
        cos_theta_neg_topk = torch.where(cond, cos_theta_neg_topk, cos_theta_neg_topk_)
        cos_theta_neg_topk = torch.pow(cos_theta_neg_topk, 2)
        times = torch.sum(torch.greater(cos_theta_neg_topk, 0).to(torch.float32), dim=1, keepdim=True)
        times = torch.where(torch.greater(times, 0), times, torch.ones_like(times))
        cos_theta_neg_topk = torch.sum(cos_theta_neg_topk, dim=1, keepdim=True) / times
        target_cos_theta_m = target_cos_theta_m - (1 + target_cos_theta_) * cos_theta_neg_topk

        cos_theta.scatter_(1, label.view(-1, 1).long(), target_cos_theta_m)
        output = cos_theta * self.scale

        return output, origin_cos * self.scale
