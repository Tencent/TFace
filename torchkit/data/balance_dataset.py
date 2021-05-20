import os
import math
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data.sampler import Sampler
from torchkit.data.index_tfr_dataset import IndexTFRDataset
from torchkit.data.datasets import FaceDataset


class IDBalanceDataset(FaceDataset):
    """ IDBalanceDataset, generates batch data which each class contains fixed num samples.
    """
    def __init__(self, root_dir, index_file, transform):
        """ Create a ``IDBalanceDataset`` object
            Args:
                root_dir: image data dir
                index_file: image list file, each line format ``xxx.jpg\t 0``
                transform: image transform
        """
        super().__init__(root_dir, index_file, transform, train=True)

    def get_person_info(self, label):
        if label in self.label2index:
            return len(self.label2index[label])
        else:
            raise RuntimeError('Cannot find label {}'.format(label))

    def __getitem__(self, index):
        label, offset = index
        sample_index = self.label2index[label][offset]
        path, _ = self.imgs[sample_index]
        sample = Image.open(path)
        sample = sample.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.label_list)


class IDBalanceTFRDataset(IndexTFRDataset):
    """ IDBalanceDataset for TFRDataset, generates batch data which each class contains fixed num samples.
    """
    def __init__(self, tfrecord_dir, index_file, transform):
        """ Create a ``IDBalanceTFRDataset`` object
            Args:
                tfrecord_dir: tfrecord saved dir
                index_file: each line format ``tfr_name\t tfr_record_index \t tfr_record_offset \t label
                                            tfr_name\t tfr_record_index \t tfr_record_offset \t label``
                transform: image transform
        """
        super().__init__(tfrecord_dir, index_file, transform)
        self.label_set, self.id_dict = self._build_id_dict()

    def _build_id_dict(self):
        label_set = set()
        id_dict = {}
        for label, record, offset in zip(self.labels, self.records, self.offsets):
            label_set.add(label)
            if label not in id_dict:
                id_dict[label] = []
            id_dict[label].append((record, offset))
        return label_set, id_dict

    def get_person_info(self, label):
        if label in self.id_dict:
            return len(self.id_dict[label])
        else:
            raise RuntimeError('Cannot find person id {}'.format(label))

    def __len__(self):
        return len(self.label_set)

    def __getitem__(self, index):
        label, inner_offset = index
        record, offset = self.id_dict[label][inner_offset]
        record_file = os.path.join(self.root_dir, record)
        return self._get_record(record_file, offset), label


class DistributedIDBalanceSampler(Sampler):
    def __init__(self, dataset, person_num, init_seed=0, num_replicas=None, rank=None):
        """ Create a ``DistributedIDBalanceSampler`` object
            A ``DistributedIDBalanceSampler`` object will generate IDBalanced batch, which
            each class contains fixed num samples

            Args:
                dataset: torch.dataset object
                person num: sample num of each class
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError('Requirse distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError('Requirse distributed package to be available')
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset)*1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.init_seed = init_seed
        self.epoch_max_iter = 0
        self.person_num = person_num
        self.epoch_max_pic = 0

        # get max_iter person, can calc for every card and allreducemax
        for person_id in range(len(self.dataset)):
            pic_num = self.dataset.get_person_info(person_id)
            person_iter = int(math.ceil(pic_num * 1.0 / self.person_num)) + 1
            if person_iter > self.epoch_max_iter:
                self.epoch_max_iter = person_iter
        self.epoch_max_pic = self.epoch_max_iter * self.person_num

    def __iter__(self):
        gen = torch.Generator()
        gen.manual_seed(self.epoch + self.init_seed)
        indices = torch.randperm(len(self.dataset), generator=gen).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        shuffle_pic_indices = []
        person_offset_indices = {}
        for person_id in indices:
            person_offset_indices[person_id] = torch.randperm(
                self.dataset.get_person_info(person_id),
                generator=gen).tolist()
            extern_times = int(math.ceil(self.epoch_max_pic * 1.0 / self.dataset.get_person_info(person_id)))
            person_offset_indices[person_id] = person_offset_indices[person_id] * extern_times
            extra_num = self.epoch_max_pic - len(person_offset_indices[person_id])
            person_offset_indices[person_id] += person_offset_indices[person_id][:extra_num]

        for person_iter in range(self.epoch_max_iter):
            for person_id in indices:
                person_id_indices = person_offset_indices[person_id]
                iter_person_indices = person_id_indices[person_iter:self.epoch_max_pic:self.epoch_max_iter]
                for i in range(self.person_num):
                    shuffle_pic_indices.append((person_id, iter_person_indices[i]))
        return iter(shuffle_pic_indices)

    def __len__(self):
        return self.epoch_max_pic * self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
