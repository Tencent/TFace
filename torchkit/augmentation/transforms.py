import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from abc import ABC, abstractmethod
from PIL import Image, ImageOps, ImageEnhance
import cv2 as cv
import math
import os
import random
class BaseTransform(ABC):

    def __init__(self, prob, mag):
        self.prob = prob
        self.mag = mag

    def __call__(self, img):
        return transforms.RandomApply([self.transform], self.prob)(img)

    def __repr__(self):
        return '%s(prob=%.2f, mag=%.2f)' % \
                (self.__class__.__name__, self.prob, self.mag)

    @abstractmethod
    def transform(self, img):
        pass


class ShearXY(BaseTransform):

    def transform(self, img):
        degrees = self.mag * 360
        t = transforms.RandomAffine(0, shear=degrees, resample=Image.BILINEAR)
        return t(img)


class TranslateXY(BaseTransform):

    def transform(self, img):
        translate = (self.mag, self.mag)
        t = transforms.RandomAffine(0, translate=translate, resample=Image.BILINEAR)
        return t(img)


class Rotate(BaseTransform):

    def transform(self, img):
        degrees = self.mag * 360
        t = transforms.RandomRotation(degrees, Image.BILINEAR)
        return t(img)


class AutoContrast(BaseTransform):

    def transform(self, img):
        cutoff = int(self.mag * 49)
        return ImageOps.autocontrast(img, cutoff=cutoff)


class Invert(BaseTransform):

    def transform(self, img):
        return ImageOps.invert(img)


class Equalize(BaseTransform):

    def transform(self, img):
        return ImageOps.equalize(img)


class Solarize(BaseTransform):

    def transform(self, img):
        threshold = (1-self.mag) * 255
        return ImageOps.solarize(img, threshold)


class Posterize(BaseTransform):

    def transform(self, img):
        bits = int((1-self.mag) * 8)
        return ImageOps.posterize(img, bits=bits)


class Contrast(BaseTransform):

    def transform(self, img):
        factor = self.mag * 10
        return ImageEnhance.Contrast(img).enhance(factor)


class Color(BaseTransform):

    def transform(self, img):
        factor = self.mag * 10
        return ImageEnhance.Color(img).enhance(factor)


class Brightness(BaseTransform):

    def transform(self, img):
        factor = self.mag * 10
        return ImageEnhance.Brightness(img).enhance(factor)


class Sharpness(BaseTransform):

    def transform(self, img):
        factor = self.mag * 10
        return ImageEnhance.Sharpness(img).enhance(factor)


class Cutout(BaseTransform):

    def transform(self, img):
        n_holes = 1
        length = 24 * self.mag
        cutout_op = CutoutOp(n_holes=n_holes, length=length)
        return cutout_op(img)


class CutoutOp(object):
    """
    https://github.com/uoguelph-mlrg/Cutout

    Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        w, h = img.size

        mask = np.ones((h, w, 1), np.uint8)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h).astype(int)
            y2 = np.clip(y + self.length // 2, 0, h).astype(int)
            x1 = np.clip(x - self.length // 2, 0, w).astype(int)
            x2 = np.clip(x + self.length // 2, 0, w).astype(int)

            mask[y1: y2, x1: x2, :] = 0.

        img = mask*np.asarray(img).astype(np.uint8)
        img = Image.fromarray(mask*np.asarray(img))

        return img

class Addmask(BaseTransform):
    def transform(self, img):
        Addmask_op = AddmaskOp(block_scale=self.mag)
        return Addmask_op(img)

class AddmaskOp(object):
    def __init__(self, block_scale):
        self.block_scale = block_scale
        self.diffw = -2 #5/256*112# TODO
        self.diffh = int(30*self.block_scale)# TODO
        self.glass_masks_dir = './masks_mask'
        self.glass_mats = []
        self.face_5points= [[40+self.diffw, 80+self.diffh], [72-self.diffw, 80+self.diffh], [56, 62]] #112*112
        self.pos_left_pupil = self.face_5points[0]
        self.pos_right_pupil = self.face_5points[1]
        self.pupil_center = ((self.face_5points[0][0] + self.face_5points[1][0]) / 2,
                             (self.face_5points[0][1] + self.face_5points[1][1]) / 2)
        self.pos_nose = self.face_5points[2]
        self.face_center = (int(self.pos_nose[0]), int(self.pos_nose[1]))
        self.distance_pupils = int(math.sqrt(pow(self.pos_left_pupil[0] - self.pos_right_pupil[0], 2) +\
                              pow(self.pos_left_pupil[1] - self.pos_right_pupil[1] ,2)))
        self.width_glass_adj = self.distance_pupils * 2
        for glass_mask_file in os.listdir(self.glass_masks_dir):
            if glass_mask_file.endswith('.png'):
                glass_mat = cv.imread(os.path.join(self.glass_masks_dir, glass_mask_file))
                if glass_mat.shape[2] == 4:
                    glass_mat = cv.cvtColor(glass_mat, cv.COLOR_BGRA2BGR)
                self.height_glass= glass_mat.shape[0]
                self.width_glass = glass_mat.shape[1]
                self.scale = float(self.width_glass_adj) / self.width_glass
                self.height_glass_adj = int(self.height_glass * self.scale)
                glass_mat = cv.resize(glass_mat, 
                                      (self.width_glass_adj, self.height_glass_adj), 
                                      interpolation=cv.INTER_LINEAR)
                kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
                glass_mat = cv.morphologyEx(glass_mat, cv.MORPH_OPEN, kernel)
                self.glass_mats.append(glass_mat)

    def __call__(self, img):
        mat_face = cv.cvtColor(np.asarray(img),cv.COLOR_RGB2BGR)

        def crop_roi(mat_glass, mat_face, pupil_center):
            # crop roi
            height_glass_adj, width_glass_adj = mat_glass.shape[:2]
            height_mat_face, width_mat_face = mat_face.shape[:2]
            flags = list() # record refine mat_glass
            min_h = max(0, int(pupil_center[1] - height_glass_adj / 2.0))
            if min_h == 0:
                flags.append(1)
            max_h = min(height_mat_face, int(pupil_center[1] + height_glass_adj / 2.0))
            if max_h == height_mat_face:
                flags.append(2)
            min_w = max(0, int(pupil_center[0] - width_glass_adj / 2.0))
            if min_w == 0:
                flags.append(3)
            max_w = min(width_mat_face, int(pupil_center[0] + width_glass_adj / 2.0))
            if max_w == width_mat_face:
                flags.append(4)
            roi = mat_face[min_h : max_h, min_w : max_w]
            # refine mat_glass
            min_h_glass = 0
            max_h_glass = height_glass_adj
            min_w_glass = 0
            max_w_glass = width_glass_adj
            if len(flags) != 0:
                if 1 in flags:
                    min_h_glass = height_glass_adj - (max_h - min_h)
                if 2 in flags:
                    max_h_glass = height_glass_adj - (height_glass_adj - (max_h - min_h))
                if 3 in flags:
                    min_w_glass = width_glass_adj - (max_w - min_w)
                if 4 in flags:
                    max_w_glass = width_glass_adj - (width_glass_adj - (max_w - min_w))
                mat_glass = mat_glass[min_h_glass: max_h_glass, min_w_glass: max_w_glass]
            return roi, mat_glass, [min_h, max_h, min_w, max_w]
        
        #Random choose glass
        idx = random.randint(0,4)
        mat_glass = self.glass_mats[idx]

        # deal mask
        roi, mat_glass, [min_h, max_h, min_w, max_w] = crop_roi(mat_glass, mat_face, self.pupil_center)
        gray_glass = cv.cvtColor(mat_glass, cv.COLOR_BGR2GRAY)
        ret, glass_mask = cv.threshold(gray_glass, 0, 255, cv.THRESH_BINARY)
        img1_bg = cv.bitwise_and(roi.copy(), roi.copy(), mask=cv.bitwise_not(glass_mask))
        img2_fg = cv.bitwise_and(mat_glass, mat_glass, mask=glass_mask)
        mat_face[min_h: max_h, min_w: max_w] = cv.add(img1_bg, img2_fg)

        img = Image.fromarray(cv.cvtColor(mat_face, cv.COLOR_BGR2RGB))
        
        return img 



class Addglass(BaseTransform):
    def transform(self, img):
        return Addglass_op(img)


class AddglassOp(object):
    def __init__(self):
        self.masks_dir = './glass_mask'
        self.glass_mats = []
        self.glass_5keys= [[40, 45], [72, 45], [56, 62]] #112*112
        self.glass_left = self.glass_5keys[0]
        self.glass_right = self.glass_5keys[1]
        self.pupil_center = ((self.glass_5keys[0][0] + self.glass_5keys[1][0]) / 2,
                             (self.glass_5keys[0][1] + self.glass_5keys[1][1]) / 2)
        self.pos_nose = self.glass_5keys[2]
        self.glass_center = (int(self.pos_nose[0]), int(self.pos_nose[1]))
        self.glass_distance = int(math.sqrt(pow(self.glass_left[0] - self.glass_right[0], 2) +\
                               pow(self.glass_left[1] - self.glass_right[1] ,2)))
        self.width_glass_adj = self.glass_distance * 2
        self.dx = self.glass_left[0] - self.glass_right[0] 
        self.dy = self.glass_left[1] - self.glass_right[1]
        self.degree = math.degrees(math.atan(self.dy/self.dx))
        for mask_file in os.listdir(self.masks_dir):
            if mask_file.endswith('.png'):
                mask_mat = cv.imread(os.path.join(self.masks_dir, mask_file))
                if mask_mat.shape[2] == 4:
                    mask_mat = cv.cvtColor(mask_mat, cv.COLOR_BGRA2BGR)
                self.height_glass= mask_mat.shape[0]
                self.width_glass = mask_mat.shape[1]
                self.scale = float(self.width_glass_adj) / self.width_glass
                self.height_glass_adj = int(self.height_glass * self.scale)
                mask_mat = cv.resize(mask_mat,
                                     (self.width_glass_adj, self.height_glass_adj), 
                                     interpolation=cv.INTER_AREA)
                mask_mat = self.rotate_bound(mask_mat, self.degree)
                self.glass_mats.append(mask_mat)

    def rotate_bound(self, image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv.warpAffine(image, M, (nW, nH)) 

    def __call__(self, img):
        mat_face = cv.cvtColor(np.asarray(img),cv.COLOR_RGB2BGR)

        def crop_roi(mask_glass, mask_face, mask_pupil_center):
            # crop roi
            height_glass_adj, width_glass_adj = mask_glass.shape[:2]
            height_mat_face, width_mat_face = mask_face.shape[:2]
            flags = list() # record refine mat_glass
            min_h = max(0, int(mask_pupil_center[1] - height_glass_adj / 2.0))
            if min_h == 0:
                flags.append(1)
            max_h = min(height_mat_face, int(mask_pupil_center[1] + height_glass_adj / 2.0))
            if max_h == height_mat_face:
                flags.append(2)
            min_w = max(0, int(mask_pupil_center[0] - width_glass_adj / 2.0))
            if min_w == 0:
                flags.append(3)
            max_w = min(width_mat_face, int(mask_pupil_center[0] + width_glass_adj / 2.0))
            if max_w == width_mat_face:
                flags.append(4)
            roi = mask_face[min_h : max_h, min_w : max_w]
            # refine mat_glass
            mask_min_h_glass = 0
            mask_max_h_glass = height_glass_adj
            mask_min_w_glass = 0
            mask_max_w_glass = width_glass_adj
            if len(flags) != 0:
                if 1 in flags:
                    mask_min_h_glass = height_glass_adj - (max_h - min_h)
                if 2 in flags:
                    mask_max_h_glass = height_glass_adj - (height_glass_adj - (max_h - min_h))
                if 3 in flags:
                    mask_min_w_glass = width_glass_adj - (max_w - min_w)
                if 4 in flags:
                    mask_max_w_glass = width_glass_adj - (width_glass_adj - (max_w - min_w))
                mask_glass = mask_glass[mask_min_h_glass: mask_max_h_glass, 
                                        mask_min_w_glass: mask_max_w_glass]
            return roi, mask_glass, [min_h, max_h, min_w, max_w]
       
        #Random choose glass
        idx = random.randint(0,4)
        mat_glass = self.glass_mats[idx]

        # deal mask
        roi, mat_glass, [min_h, max_h, min_w, max_w] = crop_roi(mat_glass, mat_face, self.pupil_center)
        gray_glass = cv.cvtColor(mat_glass, cv.COLOR_BGR2GRAY)
        ret, glass_mask = cv.threshold(gray_glass, 0, 255, cv.THRESH_BINARY)
        maska = cv.medianBlur(glass_mask, 3)

        img1_bg = cv.bitwise_and(roi.copy(), roi.copy(), mask=cv.bitwise_not(maska))
        img2_fg = cv.bitwise_and(mat_glass, mat_glass, mask=maska)
        mat_face[min_h: max_h, min_w: max_w] = cv.add(img1_bg, img2_fg)

        img = Image.fromarray(cv.cvtColor(mat_face, cv.COLOR_BGR2RGB))

        return img
