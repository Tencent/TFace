import math
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler


def create_label2index(dataset):
    """ create label2index map for BalancedBatchSampler,
        dataset is a SingleDataset object
    """
    label2index = defaultdict(list)
    for i, sample in enumerate(dataset.inputs):
        label = sample[-1]
        label2index[label].append(i)
    return label2index


class BalancedBatchSampler(Sampler):
    """ BalancedBatchSampler class
        Each batch includes `general_batch_size` general samples and `balanced_batch_size` balanced samples,
        general samples are directly randomly sampled, But balanced samples means picking `subsample_size`
        samples for each label.
    """
    def __init__(self, labels2index, general_batch_size,
                 balanced_batch_size, world_size,
                 total_size, subsample_size=4):
        self.general_batch_size = general_batch_size
        self.balanced_batch_size = balanced_batch_size
        self.subsample_size = subsample_size
        self.labels2index = labels2index
        self.total_size = total_size
        self.world_size = world_size

        self.samples = []
        for _, inds in self.labels2index.items():
            self.samples.extend(inds)

    def __iter__(self):
        for _ in range(self.__len__()):
            inds = []
            # random sample
            random_samples = random.sample(self.samples, self.general_batch_size)
            inds.extend(random_samples)

            # balanced sample
            sample_labels = random.sample(self.labels2index.keys(),
                                          self.balanced_batch_size // self.subsample_size)
            for each_label in sample_labels:
                each_label_indexes = self.labels2index[each_label]
                if len(each_label_indexes) > self.subsample_size:
                    each_label_inds = random.sample(each_label_indexes, self.subsample_size)
                elif len(each_label_indexes) == self.subsample_size:
                    each_label_inds = each_label_indexes
                else:
                    repeat_inds = each_label_indexes * math.ceil(self.subsample_size / len(each_label_indexes))
                    each_label_inds = repeat_inds[:self.subsample_size]
                inds.extend(each_label_inds)
            yield inds

    def __len__(self):
        return self.total_size // self.general_batch_size // self.world_size
