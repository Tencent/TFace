import os
import logging
import torch.distributed as dist
from torch.utils.data import Dataset
from .parser import IndexParser, ImgSampleParser, TFRecordSampleParser


class SingleDataset(Dataset):
    """ SingleDataset
    """
    def __init__(self, data_root, index_root, name, transform, **kwargs) -> None:
        """ Create a ``SingleDataset`` object

            Args:
            data_root: image or tfrecord data root path
            index_root: index file root path
            name: dataset name
            transform: transform for data augmentation
        """

        super().__init__()
        self.data_root = data_root
        self.index_root = index_root
        self.name = name
        self.index_parser = IndexParser()
        if 'TFR' not in name and 'TFR' not in kwargs:
            self.sample_parser = ImgSampleParser(transform)
        else:
            self.sample_parser = TFRecordSampleParser(transform)

        self.inputs = []
        self.is_shard = False
        self.class_num = 0
        self.sample_num = 0

    def make_dataset(self):
        self._build_inputs()

    def _build_inputs(self):
        """ Read index file and saved in ``self.inputs``
        """

        index_path = os.path.join(self.index_root, self.name + '.txt')
        with open(index_path, 'r') as f:
            for line in f:
                sample = self.index_parser(line)
                self.inputs.append(sample)
        self.class_num = self.index_parser.class_num + 1
        self.sample_num = len(self.inputs)
        logging.info("Dataset %s, class_num %d, sample_num %d" % (
            self.name, self.class_num, self.sample_num))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        """ Parse image and label data from index
        """

        sample = list(self.inputs[index])
        sample[0] = os.path.join(self.data_root, sample[0])  # data_root join
        image, label = self.sample_parser(*sample)
        return image, label


class MultiDataset(Dataset):
    """ MultiDataset, which contains multiple dataset and should be used
        together with ``MultiDistributedSampler``
    """
    def __init__(self, data_root, index_root, names, transform, **kwargs) -> None:
        """ Create a ``MultiDataset`` object

            Args:
            data_root: image or tfrecord data root path
            index_root: index file root path
            name: dataset names for multiple dataset
            transform: transform for data augmentation
        """

        super().__init__()
        self.data_root = data_root
        self.index_root = index_root
        self.names = names
        self.index_parser = IndexParser()
        if 'TFR' not in names[0] and 'TFR' not in kwargs:
            self.sample_parser = ImgSampleParser(transform)
        else:
            self.sample_parser = TFRecordSampleParser(transform)

        self.inputs = dict()
        self.is_shard = False
        self.class_nums = dict()
        self.sample_nums = dict()

    @property
    def dataset_num(self):
        return len(self.names)

    @property
    def class_num(self):
        class_nums = []
        for name in self.names:
            class_nums.append(self.class_nums[name])
        return class_nums

    def make_dataset(self, shard=False):
        """ If shard is True, the inputs are shared into all workers for memory efficiency.
            If shard is False, each worker contain the total inputs.
        """

        if shard:
            self.is_shard = True
            if not dist.is_available():
                raise RuntimeError("Requirse distributed package to be available")
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            self.total_sample_nums = dict()
            self._build_inputs(world_size, rank)
        else:
            self._build_inputs()

    def _build_inputs(self, world_size=None, rank=None):
        """ Read index file and saved in ``self.inputs``
            If ``self.is_shard`` is True, ``total_sample_nums`` > ``sample_nums``
        """

        for i, name in enumerate(self.names):
            index_file = os.path.join(self.index_root, name + ".txt")
            self.index_parser.reset()
            self.inputs[name] = []
            with open(index_file, 'r') as f:
                for line_i, line in enumerate(f):
                    sample = self.index_parser(line)
                    if self.is_shard is False:
                        self.inputs[name].append(sample)
                    else:
                        if line_i % world_size == rank:
                            self.inputs[name].append(sample)
                        else:
                            pass
            self.class_nums[name] = self.index_parser.class_num + 1
            self.sample_nums[name] = len(self.inputs[name])
            if self.is_shard:
                self.total_sample_nums[name] = self.index_parser.sample_num
                logging.info("Dataset %s, class_num %d, total_sample_num %d, sample_num %d" % (
                    name, self.class_nums[name], self.total_sample_nums[name], self.sample_nums[name]))
            else:
                logging.info("Dataset %s, class_num %d, sample_num %d" % (
                    name, self.class_nums[name], self.sample_nums[name]))

    def __getitem__(self, index):
        """ Parse image and label data from index,
            the index size should be equal with 2
        """

        if len(index) != 2:
            raise RuntimeError("MultiDataset index size wrong")
        else:
            name, index = index
            sample = list(self.inputs[name][index])
            sample[0] = os.path.join(self.data_root, sample[0])  # data_root join
            image, label = self.sample_parser(*sample)
        return image, label
