import os
import sys
import cv2
import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..'))

from common.utils import add_face_margin, get_face_box


class DG_Dataset(Dataset):
    '''
        DG_Dataset contain three datasets and preprocess images of three different domains
        Args:
            data_root (str): the path of LMDB_database
            split (str): 
                'train': generate the datasets for training 
                'val': generate the datasets for validation
                'test': generate the datasets for testing
            catgory (str):
                'pos': generate the datasets of real images
                'neg': generate the datasets of fake images 
                '': gnereate the datasets
            transform: the transforms for preprocessing
            img_mode (str):
                'rgb': generate the img in RGB mode
                'rgb_hsv': generate the img in RGB and HSV mode
            print_info (bool):
                'True': print the information of datasets
                'False': do not print the informatino of datasets
    '''

    def __init__(self,
                 data_root=None,
                 split='train',
                 category=None,
                 transform=None,
                 img_mode='rgb',
                 print_info=False,
                 **kwargs):

        self.data_root = data_root
        self.split = split
        self.category = category
        self.transform = transform
        self.img_mode = img_mode
        for k, v in kwargs.items():
            setattr(self, k, v)

        # open LMDB
        if self.use_LMDB:
            self.env = lmdb.open(self.LMDB_root, readonly=True, max_readers=1024)
            self.data = self.env.begin(write=False)
            if print_info:
                print(self.data.id())

        if self.split == 'train' and self.category == 'pos':
            self.items_1 = open(self.train_pos_list1_path).read().splitlines()
            self.items_2 = open(self.train_pos_list2_path).read().splitlines()
            self.items_3 = open(self.train_pos_list3_path).read().splitlines()
        elif self.split == 'train' and self.category == 'neg':
            self.items_1 = open(self.train_neg_list1_path).read().splitlines()
            self.items_2 = open(self.train_neg_list2_path).read().splitlines()
            self.items_3 = open(self.train_neg_list3_path).read().splitlines()
        elif self.split == 'val' and self.category == 'pos':
            self.items = open(self.val_pos_list_path).read().splitlines()
        elif self.split == 'val' and self.category == 'neg':
            self.items = open(self.val_neg_list_path).read().splitlines()
        elif self.split == 'test' and self.category == 'pos':
            self.items = open(self.test_pos_list_path).read().splitlines()
        elif self.split == 'test' and self.category == 'neg':
            self.items = open(self.test_neg_list_path).read().splitlines()
        elif self.split == 'test' and self.category == None:
            self.items = open(self.test_list_path).read().splitlines()
        else:
            self.items = []

        if print_info:
            self._display_infos()

    def _display_infos(self):
        print(f'=> Dataset {self.__class__.__name__} loaded')
        print(f'=> Split {self.split}')
        print(f'=> category {self.category}')
        if self.split == 'train':
            print(f'=> Total number of items_1: {len(self.items_1)}')
            print(f'=> Total number of items_2: {len(self.items_2)}')
            print(f'=> Total number of items_3: {len(self.items_3)}')
        print(f'=> Image mode: {self.img_mode}')

    def _get_item_index(self, index=0, items=None):
        new_item = items[index]
        new_res = new_item.split(' ')
        new_img_path = new_res[0]
        new_label = int(new_res[1])
        new_depth_path = self._convert_to_depth(new_img_path)

        if self.use_LMDB:
            img_bin = self.data.get(new_img_path.encode())
            depth_bin = self.data.get(new_depth_path.encode())
            try:
                img_buf = np.frombuffer(img_bin, dtype=np.uint8)
                depth_buf = np.frombuffer(depth_bin, dtype=np.uint8)
                new_img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
                new_depth = cv2.imdecode(depth_buf, cv2.IMREAD_COLOR)
            except:
                print('load img_buf error')
                print(new_img_path)
                new_img_path, new_label, new_img, new_depth, new_res = self._get_item_index(index + 1)
        else:
            new_img = cv2.imread(new_img_path)
            new_depth = cv2.imread(new_depth_path, cv2.IMREAD_GRAYSCALE)

        return new_img_path, new_label, new_img, new_depth, new_res

    def _convert_to_depth(self, img_path):
        if 'replayattack' in img_path:
            depth_path = img_path.replace('replayattack', 'replayattack_depth')
        elif 'CASIA_database' in img_path:
            depth_path = img_path.replace('CASIA_database', 'CASIA_database_depth')
        elif 'MSU-MFSD' in img_path:
            depth_path = img_path.replace('MSU-MFSD', 'MSU-MFSD_depth')
        elif 'Oulu_NPU' in img_path:
            depth_path = img_path.replace('Oulu_NPU', 'Oulu_NPU_depth')

        return depth_path

    def __getitem_once(self, index, items):
        length = len(items)
        index = index % length

        res = items[index].split(' ')
        img_path = res[0]
        label = int(res[1])
        depth_path = self._convert_to_depth(img_path)

        # get image from LMDB
        if self.use_LMDB:
            img_bin = self.data.get(img_path.encode())
            depth_bin = self.data.get(depth_path.encode())
            try:
                img_buf = np.frombuffer(img_bin, dtype=np.uint8)
                depth_buf = np.frombuffer(depth_bin, dtype=np.uint8)
                img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
                depth = cv2.imdecode(depth_buf, cv2.IMREAD_GRAYSCALE)
            except:
                print('load img_buf error')
                print(img_path)
                img_path, label, img, depth, res = self._get_item_index(0, items)
        else:
            img = cv2.imread(img_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        if not label == 0:
            depth = np.zeros((depth.shape[0], depth.shape[1]))

        if getattr(self, 'crop_face_from_5points', None):
            x1, x2, y1, y2 = get_face_box(img, res, margin=self.margin)
            if x1 >= x2 or y1 >= y2:
                return self.__getitem_once(0, items)
            img = img[y1:y2, x1:x2]
            depth = depth[y1:y2, x1:x2]

        # get hsv or other map
        try:
            if self.img_mode == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            elif self.img_mode == 'rgb_hsv':
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print(img_path)

        # do transform
        if self.transform is not None:
            if self.img_mode == 'rgb_hsv':
                results = self.transform(image=img, image1=img_hsv, mask=depth)
                img = results['image']
                img_hsv = results['image1']
                depth = results['mask']
            elif self.img_mode == 'rgb':
                results = self.transform(image=img, mask=depth)
                img = results['image']
                depth = results['mask']

        # reshape the depth
        if getattr(self, 'depth_map', None):
            if getattr(self, 'depth_map_size', None):
                size = int(self.depth_map_size)
            else:
                size = 32
            depth_new = depth.view(1, 1, depth.shape[0], depth.shape[1]).type(torch.float32)
            depth = F.interpolate(depth_new, (size, size), mode='bilinear').view(size, size) / 255

        if self.img_mode == 'rgb_hsv':
            img_6ch = torch.cat([img, img_hsv], 0)
            if self.return_path:
                return img_6ch, label, depth, img_path
            else:
                return img_6ch, label, depth

        elif self.img_mode == 'rgb':
            if self.return_path:
                return img, label, depth, img_path
            else:
                return img, label, depth

        else:
            print('ERROR: No such img_mode!')
            return

    def __getitem__(self, index):
        if self.split == 'train':
            # get data from item_1
            if self.return_path:
                img_1, label_1, depth_img_1, img_dir_1 = self.__getitem_once(index, self.items_1)
            else:
                img_1, label_1, depth_img_1 = self.__getitem_once(index, self.items_1)
            # get data from item_2
            if self.return_path:
                img_2, label_2, depth_img_2, img_dir_2 = self.__getitem_once(index, self.items_2)
            else:
                img_2, label_2, depth_img_2 = self.__getitem_once(index, self.items_2)
            # get data from item_3
            if self.return_path:
                img_3, label_3, depth_img_3, img_dir_3 = self.__getitem_once(index, self.items_3)
            else:
                img_3, label_3, depth_img_3 = self.__getitem_once(index, self.items_3)

            # need check
            img = torch.stack((img_1, img_2, img_3), dim=0)
            depth_img = torch.stack((depth_img_1, depth_img_2, depth_img_3), dim=0)
            label = torch.from_numpy(np.array([label_1, label_2, label_3]))
            if self.return_path:
                img_dir = [img_dir_1, img_dir_2, img_dir_3]
        else:
            if self.return_path:
                img, label, depth_img, img_dir = self.__getitem_once(index, self.items)
            else:
                img, label, depth_img = self.__getitem_once(index, self.items)

        if self.return_path:
            return img, label, depth_img, img_dir
        else:
            return img, label, depth_img

    def __len__(self):
        if self.split == 'train':
            num = max(len(self.items_1), len(self.items_2), len(self.items_3))
        else:
            num = len(self.items)
        return num
