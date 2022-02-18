import os
import sys
import albumentations as alb

from .DG_dataset import DG_Dataset

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..'))
from common.data import create_base_transforms, create_base_dataloader


def create_dg_data_transforms(args, split='train'):
    '''
        define the domain generalization transforms accoding to different parameters
        Args:
            args ([type]): contain the specific parmaters
            split (str, optinal):
                'train': to generate the domain generalization transforms for training
                'val': to generate the domain generalization transforms for validation
                'test': to generaate the domain generalization transforms for testing
    '''
    base_transform = create_base_transforms(args, split=split)
    if split == 'train':
        dg_aug_transform = alb.Compose([
            alb.HueSaturationValue(p=0.1),
        ])
        data_transform = alb.Compose([*dg_aug_transform, *base_transform])

    else:
        data_transform = base_transform

    return data_transform


def create_dataloader(args, split='train', category=None, print_info=False):
    kwargs = getattr(args.dataset, args.dataset.name)
    dg_transform = create_dg_data_transforms(args.transform, split=split)
    dg_dataset = eval(args.dataset.name)(transform=dg_transform,
                                      split=split,
                                      category=category,
                                      print_info=print_info,
                                      **kwargs)
    dg_dataloader = create_base_dataloader(args, dg_dataset, split=split)
    return dg_dataloader
