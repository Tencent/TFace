import torch
from torchkit.util.utils import l2_norm
from .common import CommonFace


class CosFace(CommonFace):
    """ Implement of CosFace (https://arxiv.org/abs/1801.09414)
    """
    def __init__(self,
                 in_features,
                 gpu_index,
                 weight_init,
                 class_split,
                 scale=64.0,
                 margin=0.4):
        """ Args:
            in_features: size of input features
            gpu_index: gpu worker index for model parallel
            weight_init: weight initial value
            class_split: class num shards
            scale: scale of input feature
            margin: margin
        """
        super(CosFace, self).__init__(in_features, gpu_index, weight_init, class_split)

        self.scale = scale
        self.margin = margin

    def forward(self, embeddings, labels):
        index, part_labels, cos_theta, original_logits = self._calc_logits(embeddings, labels)
        target_logit = cos_theta[index, part_labels[index].view(-1)]

        final_target_logit = target_logit - self.margin

        cos_theta[index, part_labels[index].view(-1)] = final_target_logit
        cos_theta = cos_theta * self.scale

        return cos_theta, part_labels, original_logits * self.scale
