import os
import sys
import albumentations as alb

from .dataset import FASDataset

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..'))
from common.data import create_base_transforms, create_base_dataloader


def create_data_transforms(args, split='train'):
    '''
        define the transforms accoding to different parameters
        Args:
            args ([type]): contain the specific parmaters
            split (str, optinal):
                'train': to generate the transforms for training
                'val': to generate the transforms for validation
                'test': to generaate the transforms for testing
    '''
    base_transform = create_base_transforms(args, split=split)
    if split == 'train':
        aug_transform = alb.Compose([
            alb.HueSaturationValue(p=0.1),
        ])
        data_transform = alb.Compose([*aug_transform, *base_transform])

    else:
        data_transform = base_transform

    return data_transform


def create_dataloader(args, split='train', category=None, print_info=False):
    kwargs = getattr(args.dataset, args.dataset.name)
    transform = create_data_transforms(args.transform, split=split)
    dataset = eval(args.dataset.name)(transform=transform,
                                      split=split,
                                      category=category,
                                      print_info=print_info,
                                      **kwargs)
    dataloader = create_base_dataloader(args, dataset, split=split)
    return dataloader
