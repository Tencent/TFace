import os
import queue as Queue
import threading
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


def read_index_file(root_dir, index_file, train):
    """ parse index file, each lines format ``image_name \t label`` or ``image_name``
    """
    samples = []
    classes = set()
    names = []
    label2index = defaultdict(list)
    with open(index_file, "r") as ifs:
        for index, line in enumerate(ifs):
            line = line.rstrip().split('\t')
            if train and len(line) < 2:
                raise RuntimeError('Label is missing')
            elif len(line) == 1:
                image_name = line[0]
                label = 0
            else:
                image_name, label = line[0], line[1]
            label = int(label)
            names.append(image_name)
            image_path = os.path.join(root_dir, image_name)
            if not os.path.exists(image_path):
                raise RuntimeError('Img %s not exits' % image_path)
            samples.append((image_path, label))
            classes.add(label)
            label2index[label].append(index)
    return samples, classes, names, label2index


class FaceDataset(Dataset):
    """ Local Facedatset
    """
    def __init__(self, root_dir, index_file, transform, train=True):
        """ Create a ``FaceDataSet`` object
            A ``FaceDataSet`` object will read the content of index_file, and parse
            each line into a [image_path, label] list, then converted to a sample by
            Dataloader.

            Args:
                root_dir: image data dir
                index_file: image list file, each line format ``xxx.jpg\t 0``
                transform: image transform
        """
        super(FaceDataset, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.train = train
        self.imgs, self.classes, self.names, self.label_to_indexes = read_index_file(
            root_dir, index_file, train=train)
        self.class_num = max(self.classes) + 1
        self.sample_num = len(self.imgs)
        print("Number of Sampels:{} Number of Classes: {}".format(self.sample_num, self.class_num))

    def __getitem__(self, index):
        path, label = self.imgs[index]
        sample = Image.open(path)
        sample = sample.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.train:
            return sample, label
        else:
            return sample, label, self.names[index]

    def __len__(self):
        return len(self.imgs)


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class IterationDataloader(object):
    """ Create a ``IterationDataloader`` object
        A ``IterationDataloader`` object will iterate a dataloader infinitely until to max iteration step

        Args:
            data_loader: torch DataLoader object
            max_iter: max iteration step
            current_iter: current iteration step
    """
    def __init__(self, dataloader, max_iter, current_iter):
        self.dataloader = dataloader
        self.max_iter = max_iter
        self.iter = current_iter
        self.epoch = int(self.iter * 1.0 / len(self.dataloader))

    def __iter__(self):
        while self.iter < self.max_iter:
            if self.dataloader.sampler and hasattr(self.dataloader.sampler, 'set_epoch'):
                self.epoch += 1
                self.dataloader.sampler.set_epoch(self.epoch)
            for _ in self.dataloader:
                yield _
                self.iter += 1
                if self.iter > self.max_iter:
                    break

    def __len__(self):
        return self.max_iter - self.iter

    @property
    def step_per_epoch(self):
        return len(self.dataloader)
