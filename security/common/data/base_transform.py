import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2


def create_base_transforms(args, split='train'):
    """Base data transformation

    Args:
        args: Data transformation args
        split (str, optional): Defaults to 'train'.

    Returns:
        [transform]: Data transform
    """
    num_segments = args.num_segments if 'num_segments' in args else 1
    additional_targets = {}
    for i in range(1, num_segments):
        additional_targets[f'image{i}'] = 'image'

    if split == 'train':
        base_transform = alb.Compose([
            alb.HorizontalFlip(),
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    elif split == 'val':
        base_transform = alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    elif split == 'test':
        base_transform = alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    return base_transform
