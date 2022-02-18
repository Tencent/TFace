import json
import os
import random
import glob
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from .utils import *
except Exception:
    from utils import *


class FaceForensics(Dataset):
    def __init__(self,
                 data_root,
                 data_types,
                 num_frames,
                 split,
                 transform=None,
                 compressions='c23',
                 mask_size=10,
                 methods=None,
                 image_size=320,
                 has_mask=False,
                 balance=True,
                 pair=True,
                 corr_pair=False,
                 random_patch=None,
                 srm=False,
                 diff_frame=False):
        """FaceForensics++ dataset
        Reference: Faceforensics: A large-scale video dataset for forgery detection in human facee, ICCV2019

        Args:
            data_root (str): Root for FFpp.
            data_types (str): Different version of dataset.
            num_frames ([int]): Frame numbers for each video.
            split (str): "train","val","test"
            transform (type, optional): Data transform. Defaults to None.
            compressions (str, optional): ["c23","c40"]. Defaults to 'c23'.
            mask_size (int, optional):. Defaults to 10.
            methods (type, optional): ["Deepfakes", "FaceSwap","Face2Face","NeuralTextures"]. Defaults to None.
            image_size (int, optional):  Defaults to 320.
            has_mask (bool, optional):  Defaults to False.
            balance (bool, optional): Whether balance the fake and real image. Defaults to True.
            pair (bool, optional): Return images with q&k or single image. Defaults to True.
            corr_pair (bool, optional): Defaults to False.
            random_patch (type, optional):  Defaults to None.
            srm (bool, optional):  The probability of using SRM data views generation. Defaults to False.
            diff_frame (bool, optional): The probability of using diff_frame data views generation. Defaults to False.
        """
        self.data_root = data_root
        self.data_types = data_types
        self.num_frames = num_frames
        self.split = split
        self.transform = transform

        if type(random_patch) is int:
            self.random_patch = RandomPatch(grid_size=random_patch)
        else:
            self.random_patch = None
        if srm != None:
            self.srm_conv = setup_srm_layer(input_channels=3)
            self.srm_prob = srm
        else:
            self.srm_conv = None

        self.compressions = compressions
        self.methods = methods
        self.image_size = image_size
        self.mask_size = mask_size
        self.has_mask = has_mask
        self.pair = pair
        self.corr_pair = corr_pair
        self.balabce = balance
        self.fake_id_dict = {}
        self.diff_frame = diff_frame

        if self.methods is None:
            self.methods = ['youtube', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

        self.real_items = self._load_items([self.methods[0]])
        self.fake_items = self._load_items(self.methods[1:])

        pos_len = len(self.real_items)
        neg_len = len(self.fake_items)
        print(f'Total number of data: {pos_len+neg_len} | pos: {pos_len}, neg: {neg_len}')
        self._convert_dict()

        if self.split == 'train' and self.balabce == True:
            np.random.seed(1234)
            if pos_len > neg_len:
                self.real_items = np.random.choice(self.real_items, neg_len, replace=False).tolist()
            else:
                self.fake_items = np.random.choice(self.fake_items, pos_len, replace=False).tolist()
            image_len = len(self.real_items)
            print(f'After balance total number of data: {image_len*2,} | pos: {image_len}, neg: {image_len}')

        self.items = self.real_items + self.fake_items
        self.items = sorted(self.items, key=lambda x: x['img_path'])
        if corr_pair == True:
            self.corr_items = self.get_corresponding()
        if diff_frame != None:
            self.diff_items = self.get_differ_frame()
            self.diff_prob = diff_frame

    def _convert_dict(self):
        """Generate data dict
        """
        for item in self.fake_items:
            video_id = item['video_id']
            self.fake_id_dict[video_id.split("_")[0]] = video_id

    def get_differ_frame(self):
        """Get corresbonding diff_frame image

        Returns:
            Dict: Dict of diff_frame image
        """
        differ_item = []
        for item in self.items:
            original_path = item['img_path']
            label = item['label']
            video_id = item['video_id']
            frame_id = item['frame_id']

            original_folder = '/'.join(original_path.split('/')[:-1]) + '/'
            face_paths = glob.glob(os.path.join(original_folder, '*.jpg'))
            differ_path = random.choice(face_paths)
            differ_item.append({
                'img_path': differ_path,
                'label': label,
                'video_id': video_id,
                'frame_id': differ_path.split('/')[:-4],
            })
        return differ_item

    def get_corresponding(self):
        """Get corresbonding image

        Returns:
            Dict: Dict of corresbonding image
        """
        corr_item = []
        for item in self.items:
            original_path = item['img_path']
            label = item['label']
            video_id = item['video_id']
            frame_id = item['frame_id']
            if label == 0.0:
                original_folder = '/'.join(original_path.split('/')[:-1]) + '/'
                face_paths = glob.glob(os.path.join(original_folder, '*.jpg'))
                corr_path = random.choice(face_paths)
                corr_item.append({
                    'img_path': corr_path,
                    'label': 0.0,
                    'video_id': video_id,
                    'frame_id': corr_path.split('/')[:-4],
                })
            else:
                orginal_method = original_path.split('/')[-5]
                new_list = list(set(self.methods[1:]).difference(set([orginal_method])))
                method = np.random.choice(new_list)

                corr_path = original_path.split('/')
                corr_path[-5] = method
                corr_path = "/".join(corr_path)

                if not os.path.exists(corr_path):
                    original_folder = '/'.join(corr_path.split('/')[:-1]) + '/'
                    face_paths = glob.glob(os.path.join(original_folder, '*.jpg'))
                    corr_path = random.choice(face_paths)
                corr_item.append({
                    'img_path': corr_path,
                    'label': 0.0,
                    'video_id': video_id.split("_")[0],
                    'frame_id': frame_id,
                })
        return corr_item

    def _load_items(self, methods):

        subdirs = FaceForensicsDataStructure(root_dir=self.data_root,
                                             methods=methods,
                                             compressions=self.compressions,
                                             data_types=self.data_types).get_subdirs()
        splits_path = os.path.join(self.data_root, 'splits')
        video_ids = get_video_ids(self.split, splits_path)
        video_dirs = []
        for dir_path in subdirs:
            video_paths = listdir_with_full_paths(dir_path)
            videos = [x for x in video_paths if get_file_name(x) in video_ids]
            video_dirs.extend(videos)

        items = []
        for video_dir in video_dirs:
            label = int(0) if 'original' in video_dir else int(1)
            sub_items = self._load_sub_items(video_dir, label)
            items.extend(sub_items)

        return items

    def _load_sub_items(self, video_dir, label):

        if self.split == 'train' and label == 1:
            num_frames = self.num_frames // 3
        else:
            num_frames = self.num_frames
        video_id = get_file_name(video_dir)
        sorted_images_names = np.array(sorted(os.listdir(video_dir), key=lambda x: int(x.split('.')[0])))
        ind = np.linspace(0, len(sorted_images_names) - 1, num_frames, endpoint=True, dtype=np.int)
        sorted_images_names = sorted_images_names[ind]

        sub_items = []
        for image_name in sorted_images_names:
            frame_id = image_name.split('_')[-1].split('.')[0]
            img_path = os.path.join(video_dir, image_name)
            sub_items.append({
                'img_path': img_path,
                'label': label,
                'video_id': video_id,
                'frame_id': frame_id,
            })
        return sub_items

    def _load_mask(self, image_path, label, binary, flip=False, mask_size=(10, 10)):
        """Load mask

        Args:
            image_path (str): Original path of image
            label (int): label of image
            binary (bool)
            flip (bool, optional): flag of flip. Defaults to False.
            mask_size (tuple, optional): mask size. Defaults to (10, 10).
        Returns:
            [Tensor]: [mask]
        """
        if label == 0:
            mask = torch.zeros(mask_size)
        else:
            method = image_path.split("/")[-5]
            item = image_path.split("/")[-2]
            tail = image_path.split("/")[-1]
            mask_path = f"../{method}_masks/images_v1/"
            mask_path = mask_path + item + "/" + tail
            if not os.path.exists(mask_path):
                mask = torch.ones(mask_size)
            else:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if binary:
                    _, mask = cv2.threshold(mask, 40, 255, 0)
                else:
                    _, mask_hard = cv2.threshold(mask, 40, 1, 0)
                    mask *= mask_hard
                mask = cv2.resize(mask, mask_size)
                if flip:
                    mask = cv2.flip(mask, 1)
                mask = torch.from_numpy(mask) / 255.
        return mask

    def _load_train_data(self, index, item, image, flip=False):
        """Load training data
        Args:
            index (int): items index
            item (Dict): description
            image (tensor): input image
            flip (bool, optional): Defaults to False.

        """
        q = self.transform(image=image)['image']

        if self.corr_pair == True and random.random() < 0.5:
            corr_item = self.corr_items[index]
            corr_image = cv2.cvtColor(cv2.imread(corr_item['img_path']), cv2.COLOR_BGR2RGB)
            k = self.transform(image=corr_image)['image']

        elif self.diff_frame == True and random.random() < self.diff_prob:
            diff_item = self.diff_items[index]
            corr_image = cv2.cvtColor(cv2.imread(diff_item['img_path']), cv2.COLOR_BGR2RGB)
            k = self.transform(image=corr_image)['image']

        else:
            k = self.transform(image=image)['image']

        if self.srm_conv != None and random.random() < self.srm_prob:
            q = q.view(1, q.shape[0], q.shape[1], q.shape[2])
            srm_q = self.srm_conv(q)
            q = q + srm_q
            q = q.squeeze()
            k = k.view(1, k.shape[0], k.shape[1], k.shape[2])
            k_srm = self.srm_conv(k)
            k = k + k_srm
            k = k.squeeze()

        if self.has_mask:
            mask = self._load_mask(item['img_path'],
                                   item['label'],
                                   binary=True,
                                   flip=flip,
                                   mask_size=(q.shape[1], q.shape[2]))

        if self.random_patch != None:
            if self.has_mask:
                q, mask = self.random_patch(q, mask.unsqueeze(0))
                mask = mask.squeeze()
                mask = torch.from_numpy(cv2.resize(mask.numpy(), (self.mask_size, self.mask_size)))
            else:
                q = self.random_patch(q)

        if self.has_mask:
            return [q, k], item['label'], item['img_path'], mask
        else:
            return [q, k], item['label'], item['img_path']

    def _load_test_data(self, index, item, image):

        image = self.transform(image=image)['image']
        return image, item['label'], item['img_path']

    def __getitem__(self, index):
        item = self.items[index]
        image = cv2.cvtColor(cv2.imread(item['img_path']), cv2.COLOR_BGR2RGB)

        flip = False
        if self.split == 'train' and random.random() < 0.5:
            flip = True
            image = cv2.flip(image, 1)

        if self.pair == True and self.split == 'train':
            return self._load_train_data(index, item, image, flip)
        else:
            return self._load_test_data(index, item, image)

    def __len__(self):
        return len(self.items)


def listdir_with_full_paths(dir_path):
    return [os.path.join(dir_path, x) for x in os.listdir(dir_path)]


def get_file_name(file_path):
    return file_path.split('/')[-1]


def read_json(file_path):
    with open(file_path) as inp:
        return json.load(inp)


def get_sets(data):
    return {x[0] for x in data} | {x[1] for x in data} | {'_'.join(x) for x in data} | {'_'.join(x[::-1]) for x in data}


def get_video_ids(spl, splits_path):
    return get_sets(read_json(os.path.join(splits_path, f'{spl}.json')))


if __name__ == '__main__':
    from transforms import create_data_transforms
    from omegaconf import OmegaConf

    args = OmegaConf.load('../configs/dcl.yaml')
    args.dataset.name = 'ffpp'
    kwargs = getattr(args.dataset, args.dataset.name)

    split = 'train'
    transform = create_data_transforms(args.transform, split)
    train_dataset = FaceForensics(split=split, transform=transform, image_size=args.transform.image_size, **kwargs)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train.batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)
    for i, datas in enumerate(train_dataloader):
        print(i, datas[0].shape, datas[1].shape)
        if kwargs.has_mask is True:
            print(datas[3].shape)
        break

    split = 'val'
    transform = create_data_transforms(args.transform, split)
    val_dataset = FaceForensics(split=split, transform=transform, image_size=args.transform.image_size, **kwargs)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val.batch_size,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True)
    for i, datas in enumerate(val_dataloader):
        print(i, datas[0].shape, datas[1].shape)
        if kwargs.has_mask is True:
            print(datas[3].shape)
        break

    split = 'test'
    transform = create_data_transforms(args.transform, split)
    test_dataset = FaceForensics(split=split, transform=transform, image_size=args.transform.image_size, **kwargs)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.test.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 pin_memory=True)
    for i, datas in enumerate(test_dataloader):
        print(i, datas[0].shape, datas[1].shape)
        if kwargs.has_mask is True:
            print(datas[3].shape)
        break
