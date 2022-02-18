import logging
import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from .dataset import MultiDataset


class MultiDistributedSampler(Sampler):
    """ MultiDistributedSampler
    """
    def __init__(self, dataset, batch_sizes, init_seed=0) -> None:
        """ Create a ``MultiDistributedSampler`` object
            A ``MultiDistributedSampler`` object will generate MultiDataset indices,

            Args:
                dataset: MultiDataset object
                batch_sizes: batch sizes for all datasets

        """
        if not dist.is_available():
            raise RuntimeError("Requirse distributed package to be available")

        if not isinstance(dataset, MultiDataset):
            raise RuntimeError("Dataset should be a MultiDataset object")

        if len(batch_sizes) != dataset.dataset_num:
            raise RuntimeError("Batch_sizes num should be equal with dataset_num")

        if len(dataset.sample_nums) == 0:
            raise RuntimeError('Please call dataset make_dataset function first')

        self.dataset = dataset
        self.batch_sizes = dict(zip(self.dataset.names, batch_sizes))
        self.init_seed = init_seed

        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()

        self.total_sizes = dict()  # sample num of all workers in one epoch
        self.mine_sizes = dict()  # sample num of one worker in one epoch
        self.batch_nums = dict()  # batch num of one worker in one epoich
        self.indices = dict()
        self.current_index = dict()
        self.current_epoch = dict()

        for name in self.dataset.names:
            self.total_sizes[name], self.mine_sizes[name], self.batch_nums[name] = self.get_total_size(name)
            self.current_epoch[name] = -1
            self.current_index[name] = 0

        self.max_batch_num = max(self.batch_nums.values())
        logging.info("MultiDistributedSampler max_batch_num %d" % (self.max_batch_num))

    def get_total_size(self, name):
        if self.dataset.is_shard:
            sample_num = self.dataset.total_sample_nums[name]
        else:
            sample_num = self.dataset.sample_nums[name]
        batch_num = math.ceil(sample_num / self.num_replicas / self.batch_sizes[name])
        total_size = batch_num * self.num_replicas * self.batch_sizes[name]
        mine_size = batch_num * self.batch_sizes[name]
        logging.info("Dataset %s, total_size %d, mine_size %d, batch_num %d" % (
            name, total_size, mine_size, batch_num))
        return total_size, mine_size, batch_num

    def get_shuffle_indices(self, name):
        self.current_epoch[name] += 1
        g = torch.Generator()
        seed = self.init_seed + self.current_epoch[name]
        if self.dataset.is_shard:
            # For dataset with shard, each worker shuffles indices with different seed
            seed += self.rank

        g.manual_seed(seed)
        indices = torch.randperm(self.dataset.sample_nums[name], generator=g).tolist()

        # add extra sample to make it evenly divisible
        if self.dataset.is_shard:
            padding_size = self.mine_sizes[name] - len(indices)
        else:
            padding_size = self.total_sizes[name] - len(indices)
        assert padding_size >= 0 and padding_size < len(indices)
        indices += indices[:padding_size]

        # subsample
        if self.dataset.is_shard is False:
            indices = indices[self.rank: self.total_sizes[name]: self.num_replicas]
        assert len(indices) == self.mine_sizes[name]
        self.current_index[name] = 0
        return indices

    def __iter__(self):
        for _ in range(self.max_batch_num):
            for name in self.dataset.names:
                current_index = self.current_index[name]
                current_epoch = self.current_epoch[name]
                batch_size = self.batch_sizes[name]
                mine_size = self.mine_sizes[name]
                if current_epoch < 0 or current_index + batch_size > mine_size:
                    self.indices[name] = self.get_shuffle_indices(name)
                for i in range(self.batch_sizes[name]):
                    index = self.indices[name][self.current_index[name]]
                    self.current_index[name] += 1
                    yield (name, index)

    def __len__(self):
        return self.max_batch_num * sum(self.batch_sizes.values())
