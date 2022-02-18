import os
import sys
import albumentations as alb

from .ffpp import FaceForensics
from .celeb_df import CelebDF
from .wild_deepfake import WildDeepfake

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..'))

from common.data import create_base_transforms, create_base_dataloader


def create_data_transforms(args, split='train'):
    """Create data transofrms

    Args:
        args: data transforms configs
        split (str, optional): split for train, val or test. Defaults to 'train'.

    Returns:
        albumentation: pytorch data augmentations
    """
    base_transform = create_base_transforms(args, split=split)

    if split == 'train':
        aug_transform = alb.Compose([
            alb.Rotate(limit=30),
            alb.Cutout(1, 25, 25, p=0.1),
            alb.RandomResizedCrop(256, 256, scale=(0.5, 1.0), p=0.5),
            alb.Resize(args.image_size, args.image_size),
            alb.HorizontalFlip(),
            alb.ToGray(p=0.1),
            alb.GaussNoise(p=0.1),
            alb.OneOf([
                alb.RandomBrightnessContrast(),
                alb.FancyPCA(),
                alb.HueSaturationValue(),
            ], p=0.7),
            alb.GaussianBlur(blur_limit=3, p=0.05),
        ])
        data_transform = alb.Compose([*aug_transform, *base_transform])

    elif split == 'val':
        data_transform = base_transform

    elif split == 'test':
        data_transform = base_transform

    return data_transform


def create_dataloader(args, split='train'):
    """create data loader

    Args:
        args: data loader configs
        split (str, optional): split for train, val or test. Defaults to 'train'.

    Returns:
        [type]: [description]
    """
    dataset_params = getattr(args.dataset, args.dataset.name)
    transform = create_data_transforms(args.transform, split=split)
    dataset = eval(args.dataset.name)(transform=transform, split=split, **dataset_params)
    dataloader = create_base_dataloader(args, dataset, split=split)
    return dataloader
