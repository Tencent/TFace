"""Poisson image editing.

"""

import numpy as np
import os
import cv2
import scipy.sparse
import pickle
from scipy.sparse.linalg import spsolve

from os import path

def laplacian_matrix(n, m):
    """Generate the Poisson matrix. 

    Refer to: 
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation

    Note: it's the transpose of the wiki's matrix 
    """
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    return mat_A


def poisson_edit(source, target, mask):
    """The poisson blending function. 

    Refer to: 
    Perez et. al., "Poisson Image Editing", 2003.
    """

    # Assume: 
    # target is not smaller than source.
    # shape of mask is same as shape of target.
    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0

    x_range = x_max - x_min
    y_range = y_max - y_min
        
    mask = mask[y_min:y_max, x_min:x_max]    
    mask[mask != 0] = 1
    
    mat_A = laplacian_matrix(y_range, x_range)

    laplacian = mat_A.tocoo()

    # set the region outside the mask to identity    
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0

    mat_A = mat_A.tocoo()

    mask_flat = mask.flatten()    
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()
        
        # inside the mask:
        # \Delta f = div v = \Delta g       
        alpha = 1
        mat_b = laplacian.dot(source_flat)*alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat==0] = target_flat[mask_flat==0]
        
        x = spsolve(mat_A, mat_b)
        x = x.reshape((y_range, x_range))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')

        target[y_min:y_max, x_min:x_max, channel] = x

    return target

def main():    
    before_dir = '../Datasets_Makeup/before_aligned_600'
    test_imgs_tmp = '../Datasets_Makeup/test_imgs_tmp'
    poisson_res_dir = '../Datasets_Makeup/test_imgs_poisson'
    api_landmarks = pickle.load(open('../Datasets_Makeup/landmark_aligned_600.pk', 'rb'))
    for i, before_name in enumerate(os.listdir(before_dir)):
        lmks = api_landmarks['before_aligned_600/' + before_name].astype(int)
        before_img = cv2.imread(before_dir + '/' + before_name)
        prefix = before_name.split('.')[0]
        for target_name in os.listdir(test_imgs_tmp):
            source = cv2.imread(test_imgs_tmp + '/' + target_name + '/' + prefix + '_fake_after.png')
            target = cv2.imread(test_imgs_tmp + '/' + target_name + '/' + prefix + '_before.png')
            mask = cv2.imread(test_imgs_tmp + '/' + target_name + '/' + prefix + '_mask.png', cv2.IMREAD_GRAYSCALE)

            result = poisson_edit(source, target, mask)

            eye_area = [9, 10, 11, 19, 84, 29, 79, 28, 24, 73, 70, 75, 74, 13, 15, 14, 22]

            top_left = [min(lmks[eye_area, 0]), min(lmks[eye_area, 1])]
            top_right = [max(lmks[eye_area, 0]), max(lmks[eye_area, 1])]

            before_img[top_left[1]:top_right[1], top_left[0]:top_right[0], :] = result

            os.makedirs(path.join(poisson_res_dir, target_name), exist_ok=True)
            cv2.imwrite(path.join(poisson_res_dir, target_name, prefix + "_possion.png"), before_img)

        print('%d th image generated!' % (i))
    

if __name__ == '__main__':
    main()
