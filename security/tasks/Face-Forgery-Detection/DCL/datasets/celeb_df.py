import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

try:
    from .utils import RandomPatch
except Exception:
    from utils import RandomPatch


class CelebDF(Dataset):
    def __init__(self,
                 data_root,
                 data_types,
                 num_frames,
                 split,
                 transform=None,
                 pair=False,
                 random_patch=None,
                 sample=0):
        """Load Celeb-DF dataset

        Args:
            data_root: Root for Celeb-DF.
            data_types: Different version of dataset.
            num_frames: Frame numbers for each video.
            split: ["train","val","test"]
            transform (optional): Data transform. Defaults to None.
            pair (bool, optional): Return images with q&k or single image. Defaults to False.
            random_patch (int, optional):  Defaults to None.
            sample (int, optional): Defaults to 0.
        """
        self.data_root = Path(data_root)
        self.data_types = data_types
        self.num_frames = num_frames
        self.split = split
        self.transform = transform
        self.pair = pair
        self.sample = sample

        self.real_items = []
        self.fake_items = []
        if random_patch != None:
            self.random_patch = RandomPatch(grid_size=random_patch)
        else:
            self.random_patch = None
        self._read_videos()
        self._read_images()

        pos_len = len(self.real_items)
        neg_len = len(self.fake_items)
        print(f'Total number of data: {pos_len+neg_len} | pos: {pos_len}, neg: {neg_len}')

        np.random.seed(1234)
        if self.sample > 0:
            self.real_items = np.random.choice(self.real_items, neg_len // sample, replace=False).tolist()
            self.fake_items = np.random.choice(self.fake_items, pos_len // sample, replace=False).tolist()
        else:
            if pos_len > neg_len:
                self.real_items = np.random.choice(self.real_items, neg_len, replace=False).tolist()
            else:
                self.fake_items = np.random.choice(self.fake_items, pos_len, replace=False).tolist()
        image_len = len(self.real_items)
        print(f'After balance total number of data: {image_len*2} | pos: {image_len}, neg: {image_len}')

        self.items = self.real_items + self.fake_items
        self.items = sorted(self.items, key=lambda x: x['img_path'])
        self._display_infos()

    def _read_videos(self):
        """read videos
        """
        all_video_dirs = glob(str(self.data_root / self.data_types / '*' / '*'))
        test_list = open(self.data_root / self.data_types / 'List_of_testing_videos.txt').readlines()
        test_video_dirs = []
        for line in test_list:
            line = line.strip()
            video_dir = str(self.data_root / self.data_types / line.split()[1].split('.')[0])
            assert os.path.exists(video_dir)
            test_video_dirs.append(video_dir)
        train_video_dirs = list(set(all_video_dirs) - set(test_video_dirs))
        self.video_dirs = train_video_dirs if self.split == 'train' else test_video_dirs

    def _read_images(self):
        """read images
        """
        for video_dir in self.video_dirs:
            label = 0. if 'real' in video_dir else 1.
            self._read_class_images(label, video_dir)

    def _read_class_images(self, label, video_dir):
        """read images with target class

        Args:
            label: class label
            video_dir: video directory
        """
        if label == 1 and self.split == 'train':
            num_frames = self.num_frames // 7
        elif label == 1 and self.split == 'test':
            num_frames = self.num_frames // 2
        else:
            num_frames = self.num_frames
        sorted_images_names = np.array(sorted(os.listdir(video_dir), key=lambda x: int(x.split('.')[0])))
        ind = np.linspace(0, len(sorted_images_names) - 1, num_frames, endpoint=True, dtype=np.int)
        sorted_images_names = sorted_images_names[ind]
        for image_name in sorted_images_names:
            frame_id = image_name.split('_')[-1].split('.')[0]
            if label == 0:
                self.real_items.append({
                    'img_path': os.path.join(video_dir, image_name),
                    'label': label,
                    'frame_id': frame_id,
                })
            elif label == 1:
                self.fake_items.append({
                    'img_path': os.path.join(video_dir, image_name),
                    'label': label,
                    'frame_id': frame_id,
                })

    def _display_infos(self):
        """display dataset infors
        """
        print(f'{self.__class__.__name__} & {self.split}')
        print('real image nums', len(self.real_items))
        print('fake image nums', len(self.fake_items))
        print('total image nums', len(self.items))

    def __getitem__(self, index):
        item = self.items[index]
        image = cv2.cvtColor(cv2.imread(item['img_path']), cv2.COLOR_BGR2RGB)

        if self.pair == True and self.split == 'train':
            q = self.transform(image=image)['image']
            k = self.transform(image=image)['image']
            if self.random_patch != None:
                q = self.random_patch(q)
                k = self.random_patch(k)
            return [q, k], item['label'], item['img_path']

        if self.transform is not None:
            image = self.transform(image=image)['image']
        if self.split == 'train' and self.random_patch != None:
            image = self.random_patch(image)

        return image, item['label'], item['img_path']

    def __len__(self):
        return len(self.items)


if __name__ == '__main__':
    from transforms import create_data_transforms
    from omegaconf import OmegaConf

    args = OmegaConf.load('../configs/default.yaml')
    args.dataset.name = 'celeb_df'
    kwargs = getattr(args.dataset, args.dataset.name)

    split = 'train'
    transform = create_data_transforms(args.transform, split)
    train_dataset = CelebDF(split=split, transform=transform, **kwargs)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train.batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)
    for i, datas in enumerate(train_dataloader):
        print(i, datas[0].shape, datas[1].shape)
        break

    split = 'test'
    transform = create_data_transforms(args.transform, split)
    test_dataset = CelebDF(split=split, transform=transform, **kwargs)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.test.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 pin_memory=True)
    for i, datas in enumerate(test_dataloader):
        print(i, datas[0].shape, datas[1].shape)
        break
