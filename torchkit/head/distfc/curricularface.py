import math
import torch
from torch.distributed import ReduceOp
from torchkit.util.utils import l2_norm
from .common import CommonFace


class CurricularFace(CommonFace):
    """ Implement of CurricularFace (https://arxiv.org/abs/2004.00288)
    """
    def __init__(self,
                 in_features,
                 gpu_index,
                 weight_init,
                 class_split,
                 scale=64.0,
                 margin=0.5,
                 alpha=0.1):
        """ Args:
            in_features: size of input features
            gpu_index: gpu worker index for model parallel
            weight_init: weight initial value
            class_split: class num shards
            scale: scale of input feature
            margin: margin
            alpha: alpha
        """
        super(CurricularFace, self).__init__(in_features, gpu_index, weight_init, class_split)
        self.scale = scale
        self.margin = margin
        self.alpha = alpha
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.register_buffer('t', torch.zeros(1))

    def forward(self, embeddings, labels):
        index, part_labels, cos_theta, original_logits = self._calc_logits(embeddings, labels)
        target_logit = torch.zeros(embeddings.size(0), device=embeddings.device)

        target_logit[index] = cos_theta[index, part_labels[index].view(-1)]

        torch.distributed.all_reduce(target_logit, ReduceOp.SUM)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)

        hard_sample_mask = cos_theta > cos_theta_m.view(-1, 1)
        # print('hard_sample_mask', hard_sample_mask.size())
        hard_example = cos_theta[hard_sample_mask]
        final_target_logit = torch.where(target_logit > self.theta,
                                         cos_theta_m,
                                         target_logit - self.sinmm)
        with torch.no_grad():
            self.t = target_logit.mean() * self.alpha + (1 - self.alpha) * self.t
        cos_theta[hard_sample_mask] = hard_example * (self.t + hard_example)
        cos_theta[index, part_labels[index].view(-1)] = final_target_logit[index]
        cos_theta = cos_theta * self.scale
        return cos_theta, part_labels, original_logits * self.scale
