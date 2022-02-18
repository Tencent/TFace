# based on
# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/partial_fc.py
import os
import logging
from itertools import accumulate
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.functional import normalize


class PartialFC(nn.Module):
    """ Implement of PartialFC (https://arxiv.org/abs/2010.05222)
    """
    def __init__(self,
                 in_features,
                 gpu_index,
                 weight_init,
                 class_split,
                 scale=64.0,
                 margin=0.40):
        """
        Args:
            in_features: size of input features
            gpu_index: gpu worker index for model parallel
            weight_init: weight initial value
            class_split: class num shards
            scale: scale of input feature
            margin: margin
        """
        super().__init__()
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = gpu_index
        self.device = torch.device("cuda:{}".format(self.local_rank))
        self.num_classes = class_split[rank]
        self.shards = []
        self.shards.append(0)
        self.shards.extend(accumulate(class_split))
        self.shard_start = self.shards[rank]
        self.shard_end = self.shards[rank + 1]
        logging.info('FC Start Point: {}'.format(self.shards))
        logging.info('Rank: {}, shards ranges [{}, {}]'.format(rank, self.shard_start, self.shard_end))

        shard_weight_init = weight_init[:, self.shard_start: self.shard_end]
        self.weight = shard_weight_init.cuda(self.local_rank)
        self.weight_mom = torch.zeros_like(self.weight)

        self.s = scale
        self.m = margin
        self.embedding_size = in_features
        self.sample_rate = 0.1
        self.num_sample = int(self.sample_rate * self.num_classes)
        self.index = None
        self.sub_weight = Parameter(torch.empty((0, 0)).cuda(self.local_rank))
        self.sub_weight_mom = None

    def load_pretrain_weight(self, weight):
        if weight.size() != self.weight.size():
            raise RuntimeError("Size not equal bewteen pretrain weight and partialfc weight",
                               weight.size(), self.weight.size())
        self.weight = weight.cuda(self.local_rank)

    @torch.no_grad()
    def update(self):
        self.weight_mom[:, self.index] = self.sub_weight_mom
        self.weight[:, self.index] = self.sub_weight

    @torch.no_grad()
    def sample(self, labels):
        with torch.no_grad():
            part_labels = labels.clone()
        index_pos = (self.shard_start <= part_labels) & (part_labels < self.shard_end)
        part_labels[~index_pos] = -1
        part_labels[index_pos] -= self.shard_start
        positive = torch.unique(part_labels[index_pos], sorted=True)
        if self.num_sample - positive.size(0) >= 0:
            perm = torch.rand(size=[self.num_classes], device=self.device)
            perm[positive] = 2
            index = torch.topk(perm, k=self.num_sample)[1]
            index = index.sort()[0]
        else:
            index = positive
        self.index = index
        part_labels[index_pos] = torch.searchsorted(index, part_labels[index_pos])
        self.sub_weight = Parameter(self.weight[:, index])
        self.sub_weight_mom = self.weight_mom[:, index]
        return part_labels

    def calc_logit(self, features):
        norm_feature = normalize(features, dim=1)
        norm_weight = normalize(self.sub_weight, dim=0)
        logits = torch.mm(norm_feature, norm_weight)
        return logits

    def margin_softmax(self, cosine, label):
        """ Cosine face for partialfc
        """
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret

    def forward(self, features, labels, optimizer):
        part_labels = self.sample(labels)
        optimizer.state.pop(optimizer.param_groups[-1]['params'][0], None)
        optimizer.param_groups[-1]['params'][0] = self.sub_weight
        optimizer.state[self.sub_weight]['momentum_buffer'] = self.sub_weight_mom
        logits = self.calc_logit(features)
        with torch.no_grad():
            raw_logits = logits.clone()
        logits = self.margin_softmax(logits, part_labels)
        part_labels = part_labels.view(-1, 1)
        return logits, part_labels, raw_logits * self.s
