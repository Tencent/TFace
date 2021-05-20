from itertools import accumulate
import logging
import torch
import torch.nn as nn
from torch.nn import Parameter
from torchkit.util.utils import l2_norm


class CommonFace(nn.Module):
    """ CommonFace head
    """
    def __init__(self,
                 in_features,
                 gpu_index,
                 weight_init,
                 class_split):
        """ Args:
            in_features: size of input features
            gpu_index: gpu worker index for model parallel
            weight_init: weight initial value
            class_split: class num shards
        """
        super(CommonFace, self).__init__()
        self.in_features = in_features
        self.gpu_index = gpu_index
        self.out_features = class_split[gpu_index]
        self.shard_start = []
        self.shard_start.append(0)
        self.shard_start.extend(accumulate(class_split))
        logging.info('FC Start Point: {}'.format(self.shard_start))

        select_weight_init = weight_init[:, self.shard_start[self.gpu_index]:
                                         self.shard_start[self.gpu_index + 1]]

        self.kernel = Parameter(select_weight_init.clone())

    def _calc_logits(self, embeddings, labels):
        """ calculate original logits
        """
        embeddings = l2_norm(embeddings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        with torch.no_grad():
            original_logits = cos_theta.clone()
        labels = labels.view(-1, 1)
        part_labels = self._generate_part_labels(labels)
        index = torch.where(part_labels != -1)[0]
        return index, part_labels, cos_theta, original_logits

    def _generate_part_labels(self, labels):
        with torch.no_grad():
            part_labels = labels.clone()
        shad_start = self.shard_start[self.gpu_index]
        shad_end = self.shard_start[self.gpu_index + 1]
        label_mask = torch.ge(part_labels, shad_start) & torch.lt(part_labels, shad_end)

        part_labels[~label_mask] = -1
        part_labels[label_mask] -= shad_start

        return part_labels

    def forward(self, embeddings, labels):
        raise NotImplementedError()
