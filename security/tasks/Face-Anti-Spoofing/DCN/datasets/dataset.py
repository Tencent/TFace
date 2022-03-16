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


class FASDataset(Dataset):
    '''
        DG_Dataset contain one dataset and preprocess images 
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
        self.permutations = self._retrieve_permutations(self.permutation_path)

        # open LMDB
        if self.use_LMDB:
            self.env = lmdb.open(self.LMDB_root, readonly=True, max_readers=1024)
            self.data = self.env.begin(write=False)
            if print_info:
                print(self.data.id())

        if self.split == 'test' and self.category == None:
            self.items = open(self.test_list_path).read().splitlines()
        else:
            self.items = open(eval(f'self.{self.split}_{self.category}_list_path')).read().splitlines()

        if print_info:
            self._display_infos()

    def _retrieve_permutations(self, permutation_path):
        all_perm = np.load(permutation_path)
        if all_perm.min() == 1:
            all_perm = all_perm - 1
        return all_perm

    def _display_infos(self):
        print(f'=> Dataset {self.__class__.__name__} loaded')
        print(f'=> Split {self.split}')
        print(f'=> category {self.category}')
        if self.split == 'train':
            print(f'=> Total number of items: {len(self.items)}')
        print(f'=> Image mode: {self.img_mode}')

    def _get_item_index(self, index=0):
        new_item = self.items[index]
        new_res = new_item.split(' ')
        new_img_path = new_res[0]
        new_label = int(new_res[1])
        new_reflection_path = self._convert_to_reflection(new_img_path)

        if self.use_LMDB:
            img_bin = self.data.get(new_img_path.encode())
            reflection_bin = self.data.get(new_reflection_path.encode())
            try:
                img_buf = np.frombuffer(img_bin, dtype=np.uint8)
                reflection_buf = np.frombuffer(reflection_bin, dtype=np.uint8)
                new_img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
                new_reflection = cv2.imdecode(reflection_buf, cv2.IMREAD_GRAYSCALE)
            except:
                print('load img_buf error')
                print(new_img_path)
                new_img_path, new_label, new_img, \
                new_reflection, new_res = self._get_item_index(index+1)
        else:
            new_img = cv2.imread(new_img_path)
            new_reflection = cv2.imread(new_reflection_path, cv2.IMREAD_GRAYSCALE)

        return new_img_path, new_label, new_img, new_reflection, new_res

    def _convert_to_reflection(self, img_path):
        if 'replayattack' in img_path:
            reflection_path = img_path.replace('replayattack', 'replayattack_reflection')
        elif 'CASIA_database' in img_path:
            reflection_path = img_path.replace('CASIA_database', 'CASIA_database_reflection')
        elif 'MSU-MFSD' in img_path:
            reflection_path = img_path.replace('MSU-MFSD', 'MSU-MFSD_reflection')
        elif 'Oulu_NPU' in img_path:
            reflection_path = img_path.replace('Oulu_NPU', 'Oulu_NPU_reflection')

        return reflection_path

    def _get_tile(self, img, n):
        if len(img.shape) == 3:
            w = int((img.shape[1]) / self.grid_size)
            y = int(n / self.grid_size)
            x = n % self.grid_size
            tile = img[:, x * w:(x + 1) * w, y * w:(y + 1) * w]
        else:
            w = int((img.shape[1]) / self.grid_size)
            y = int(n / self.grid_size)
            x = n % self.grid_size
            tile = img[x * w:(x + 1) * w, y * w:(y + 1) * w]
        return tile

    def _stack_together(self, img, img_tiles):
        # stack the images tile
        if len(img.shape) == 3:
            num = img_tiles.shape[0]
            subwidth = img_tiles.shape[-1]
            originalpic = torch.zeros_like(img)

            for i in range(num):
                y = int(i / self.grid_size)
                x = int(i % self.grid_size)
                originalpic[:, x * subwidth:(x + 1) * subwidth, y * subwidth:(y + 1) * subwidth] = img_tiles[i, :, :, :]
        else:
            num = img_tiles.shape[0]
            subwidth = img_tiles.shape[-1]
            originalpic = torch.zeros_like(img)

            for i in range(num):
                y = int(i / self.grid_size)
                x = int(i % self.grid_size)
                originalpic[x * subwidth:(x + 1) * subwidth, y * subwidth:(y + 1) * subwidth] = img_tiles[i, :, :]

        return originalpic

    def _random_patch(self, img, order):
        n_grids = self.grid_size**2
        img_tiles = [None] * n_grids

        for n in range(n_grids):
            img_tiles[n] = self._get_tile(img, n)
        # random select one order of jigsaw
        img_data = [img_tiles[self.permutations[order][t]] for t in range(n_grids)]
        img_data = torch.stack(img_data, 0)

        img_data = self._stack_together(img, img_data)

        return img_data

    def __getitem_once(self, index):
        res = self.items[index].split(' ')
        img_path = res[0]
        label = int(res[1])
        reflection_path = self._convert_to_reflection(img_path)
        # get image from LMDB
        if self.use_LMDB:
            img_bin = self.data.get(img_path.encode())
            reflection_bin = self.data.get(reflection_path.encode())
            try:
                img_buf = np.frombuffer(img_bin, dtype=np.uint8)
                reflection_buf = np.frombuffer(reflection_bin, dtype=np.uint8)
                img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
                reflection = cv2.imdecode(reflection_buf, cv2.IMREAD_GRAYSCALE)
            except:
                print('load img_buf error')
                print(img_path)
                img_path, label, img, reflection, res = self._get_item_index(0)
        else:
            img = cv2.imread(img_path)
            reflection = cv2.imread(reflection_path, cv2.IMREAD_GRAYSCALE)

        if label == 0:
            reflection = np.zeros((reflection.shape[0], reflection.shape[1])).astype(img.dtype)

        if getattr(self, 'crop_face_from_5points', None):
            x1, x2, y1, y2 = get_face_box(img, res, margin=self.margin)
            if x1 >= x2 or y1 >= y2:
                return self.__getitem_once(0)
            img = img[y1:y2, x1:x2]
            reflection = reflection[y1:y2, x1:x2]

        # get hsv or other map
        try:
            if self.img_mode == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif self.img_mode == 'rgb_hsv':
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print(img_path)

        # do transform
        if self.transform is not None:
            if self.img_mode == 'rgb_hsv':
                results = self.transform(image=img, image1=img_hsv, mask=reflection)
                img = results['image']
                img_hsv = results['image1']
                reflection = results['mask']
            elif self.img_mode == 'rgb':
                results = self.transform(image=img, mask=reflection)
                img = results['image']
                reflection = results['mask']

        # random the patch
        if self.split == 'train':
            order = np.random.randint(len(self.permutations))
            img = self._random_patch(img, order)
            if self.img_mode == 'rgb_hsv':
                img_hsv = self._random_patch(img_hsv, order)
            if getattr(self, 'reflection_map', None):
                reflection = self._random_patch(reflection, order)

        reflection = self._scale_reflection(reflection)

        if self.img_mode == 'rgb_hsv':
            img_6ch = torch.cat([img, img_hsv], 0)
            if self.return_path:
                return img_6ch, label, reflection, img_path
            else:
                return img_6ch, label, reflection

        elif self.img_mode == 'rgb':
            if self.return_path:
                return img, label, reflection, img_path
            else:
                return img, label, reflection

        else:
            print('ERROR: No such img_mode!')
            return

    def _scale_reflection(self, reflection):
        if getattr(self, 'reflection_map', None):
            if getattr(self, 'reflection_map_size', None):
                size = int(self.reflection_map_size)
            else:
                size = 32
            if not self.split == 'train':
                reflection_new = reflection.view(1, 1, reflection.shape[0], reflection.shape[1])
                reflection_new = reflection_new.type(torch.float32)
                reflection = F.interpolate(reflection_new, (size, size), mode='bilinear')
                reflection = reflection.view(size, size) / 255
            else:
                reflection_new = reflection.view(1, 1, reflection.shape[0], reflection.shape[1])
                reflection_new = reflection_new.type(torch.float32)
                reflection = F.interpolate(reflection_new, (self.img_size, self.img_size), mode='bilinear')
                reflection = reflection.view(self.img_size, self.img_size) / 255

        return reflection

    def __getitem__(self, index):
        if self.split == 'train':
            # get data
            if self.return_path:
                img, label, reflection_img, img_dir = self.__getitem_once(index)
            else:
                img, label, reflection_img = self.__getitem_once(index)
        else:
            if self.return_path:
                img, label, reflection_img, img_dir = self.__getitem_once(index)
            else:
                img, label, reflection_img = self.__getitem_once(index)

        if self.return_path:
            return img, label, reflection_img, img_dir
        else:
            return img, label, reflection_img

    def __len__(self):
        num = len(self.items)
        return num
