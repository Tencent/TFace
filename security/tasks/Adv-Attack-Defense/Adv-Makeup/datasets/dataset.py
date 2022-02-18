# -*- coding: utf-8 -*-
import os
import glob
import pickle
from PIL import Image
from numpy import asarray
import numpy as np
import random
import torch.utils.data as data
from torchvision.transforms import *

# ********************************************
# General data-Loader for training and testing
# ********************************************
class dataset_makeup(data.Dataset):
    def __init__(self, config):
        self.resize_size = config.resize_size
        self.data_dir = config.data_dir
        self.lmk_name = config.lmk_name
        self.after_dir = config.after_dir
        self.before_dir = config.before_dir
        self.eye_area = config.eye_area

        # Load image landmarks for the face images
        self.api_landmarks = pickle.load(open(os.path.join(self.data_dir, self.lmk_name), 'rb'))
        # Load the Un-makeup image path list
        self.before_list = [self.before_dir + '/' + img_path
                            for img_path in  os.listdir(os.path.join(self.data_dir , self.before_dir))]
        # Load the real-world makeup image path list
        self.after_list = [self.after_dir + '/' + img_path
                           for img_path in  os.listdir(os.path.join(self.data_dir , self.after_dir))]

        transforms = []
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        self.transforms = Compose(transforms)

    def __len__(self):
        return len(self.before_list)

    def __getitem__(self, index):
        before_path = self.before_list[index]
        after_path = self.after_list[index % len(self.after_list)]

        before_lmks = self.api_landmarks[before_path].astype(np.int)
        after_lmks = self.api_landmarks[after_path].astype(np.int)

        # Obtain the coordinates of the top-left and bottom right corners of the eye-area
        # for the un-makeup images.
        before_top_left = [min(before_lmks[self.eye_area, 0]), min(before_lmks[self.eye_area, 1])]
        before_bottom_right = [max(before_lmks[self.eye_area, 0]), max(before_lmks[self.eye_area, 1])]

        # Obtain the coordinates of the top-left and bottom right corners of the eye-area
        # for the real-world makeup images.
        after_top_left = [min(after_lmks[self.eye_area, 0]), min(after_lmks[self.eye_area, 1])]
        after_bottom_right = [max(after_lmks[self.eye_area, 0]), max(after_lmks[self.eye_area, 1])]

        before_img = Image.open(os.path.join(self.data_dir, before_path)).convert('RGB')

        after_img = Image.open(os.path.join(self.data_dir, after_path)).convert('RGB')

        before_img_arr = asarray(before_img)
        after_img_arr = asarray(after_img)

        # Crop the eye-area for the un-makeup images.
        crop_before_img = before_img_arr[before_top_left[1]:before_bottom_right[1],
                          before_top_left[0]:before_bottom_right[0]]
        # Crop the eye-area for the real-world makeup images.
        crop_after_img = after_img_arr[after_top_left[1]:after_bottom_right[1],
                         after_top_left[0]:after_bottom_right[0]]

        crop_before_img = Image.fromarray(crop_before_img).resize(self.resize_size)
        crop_after_img = Image.fromarray(crop_after_img).resize(self.resize_size)

        crop_before_img = self.transforms(crop_before_img)
        crop_after_img = self.transforms(crop_after_img)

        return crop_before_img, crop_after_img, before_path, after_path
