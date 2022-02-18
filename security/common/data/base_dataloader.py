import torch.utils.data as data


def create_base_dataloader(args, dataset, split):
    """Base data loader

    Args:
        args: Dataset config args
        split (string): Load "train", "val" or "test"

    Returns:
        [dataloader]: Corresponding Dataloader
    """
    sampler = None
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    shuffle = True if sampler is None and split == 'train' else False
    batch_size = getattr(args, split).batch_size
    num_workers = args.num_workers if 'num_workers' in args else 8

    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 sampler=sampler,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True)
    return dataloader
