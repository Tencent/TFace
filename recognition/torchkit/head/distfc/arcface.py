import math
import torch
from torchkit.util.utils import l2_norm
from .common import CommonFace


class ArcFace(CommonFace):
    """ Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf)
    """
    def __init__(self,
                 in_features,
                 gpu_index,
                 weight_init,
                 class_split,
                 scale=64.0,
                 margin=0.5,
                 easy_margin=False):
        """ Args:
            in_features: size of input features
            gpu_index: gpu worker index for model parallel
            weight_init: weight initial value
            class_split: class num shards
            scale: scale of input feature
            margin: margin
        """
        super(ArcFace, self).__init__(in_features, gpu_index, weight_init, class_split)
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        index, part_labels, cos_theta, original_logits = self._calc_logits(embeddings, labels)
        target_logit = cos_theta[index, part_labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(target_logit > 0, cos_theta_m,
                                             target_logit)
        else:
            final_target_logit = torch.where(target_logit > self.theta,
                                             cos_theta_m,
                                             target_logit - self.sinmm)

        cos_theta[index, part_labels[index].view(-1)] = final_target_logit
        cos_theta = cos_theta * self.scale

        return cos_theta, part_labels, original_logits * self.scale
