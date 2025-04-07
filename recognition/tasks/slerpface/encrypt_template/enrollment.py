import sys
import numpy as np
import argparse
import os
import time
import shutil


# parse the args
parser = argparse.ArgumentParser(description='Enrollment in SlerpFace')
parser.add_argument('--templates_folder', default="./templates", type=str)
parser.add_argument('--key', type=str, required=True,
                    help='key for encryption (e.g., LFW, CFP-FP)')
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--drop_rate', type=float, default=0.5)
parser.add_argument('--group_size', type=int, default=16)


args = parser.parse_args()

def group_slerp(z1, z2, alphas, group_size = 8):

    # step1: reshape to m group
    z1 = z1.reshape(-1,group_size)
    z2 = z2.reshape(-1,group_size)

    # step2: norm
    norm_z1 = np.linalg.norm(z1, axis=1).reshape(-1,1)
    norm_z2 = np.linalg.norm(z2, axis=1).reshape(-1,1)
    z1 = z1 / norm_z1
    z2 = z2 / norm_z2

    # step3: count dots
    dot_values = (z1 * z2).sum(axis = 1)
    dot_values[dot_values > 1] = 1.
    dot_values[dot_values < -1] = -1.

    # step4: count thetas
    thetas = np.arccos(dot_values).reshape(-1,1)

    # step5: count result
    sin_thetas = np.sin(thetas).reshape(-1,1)
    alpha_thetas = np.sin(alphas * thetas).reshape(-1,1)
    one_minus_alpha_thetas =  np.sin((1 - alphas) * thetas).reshape(-1,1)
    result =  one_minus_alpha_thetas / sin_thetas * z1 + alpha_thetas / sin_thetas * z2
    return result.reshape(-1), z2

def slerp(z1, z2, alpha = 0.8):
    dot_value = np.sum(z1 * z2) / (np.linalg.norm(z1) * np.linalg.norm(z2))
    if dot_value >= 1:
        return z1
    if dot_value < -1:
        dot_value = -1
    theta = np.arccos(dot_value)    
    return (
        np.sin((1 - alpha) * theta) / np.sin(theta) * z1
        + np.sin(alpha * theta) / np.sin(theta) * z2
    )

def enroll_SlerpFace_attention_group(feature_gallery, maps, alphas, drop_rate, group_size, gap_base = 8):
    start = time.time()
    feature_gallery = feature_gallery.reshape(-1,group_size)

    # step 1: gen key
    key = np.random.randn(feature_gallery.shape[0],group_size)
    
    # step 2: slerp
    maps = maps.reshape(-1) 
    alphas = alphas - maps
    alphas = alphas.reshape(-1,1)
    feature_gallery, key = group_slerp(feature_gallery, key, alphas, group_size)
    
    # step 3: drop
    median = np.median(maps)
    low_weight_indexs = np.where(maps < median)[0]
    high_weight_indexs = np.where(maps >= median)[0]

    # drop more on low 
    gap = int(gap_base * drop_rate)
    low_relative_mask_index = np.random.choice(
        range(group_size),
        int(low_weight_indexs.shape[0] * (group_size * drop_rate + gap)),
        replace=True
    )
    low_relative_mask_index = low_relative_mask_index.reshape(low_weight_indexs.shape[0], -1)
    low_offset = (
        np.ones((low_weight_indexs.shape[0], 1)) * 
        group_size * 
        low_weight_indexs.reshape(-1, 1)
    ).astype(np.int32)
    low_mask_index = (low_relative_mask_index + low_offset).reshape(-1)

    # drop less on high
    high_relative_mask_index = np.random.choice(
        range(group_size),
        int(high_weight_indexs.shape[0] * (group_size * drop_rate - gap)),
        replace=True
    )
    high_relative_mask_index = high_relative_mask_index.reshape(high_weight_indexs.shape[0], -1)
    high_offset = (np.ones((high_weight_indexs.shape[0],1)) * group_size * high_weight_indexs.reshape(-1,1)).astype(np.int32)
    high_mask_index = (high_relative_mask_index + high_offset).reshape(-1)
    
    feature_gallery[low_mask_index] = 0
    feature_gallery[high_mask_index] = 0

    # setp 4: norm
    feature_gallery = feature_gallery.reshape(-1, group_size)
    norm_feature = np.linalg.norm(feature_gallery, axis=1).reshape(-1,1)
    feature_gallery = feature_gallery / norm_feature
    # setp 5: mul by map
    feature_gallery = feature_gallery * maps.reshape(-1,1)
    return [feature_gallery,key], time.time() - start


def encrypt_template(
    templates_folder, 
    key, 
    alpha = 0.8, 
    drop_rate = 0.25, 
    group_size = 4
):
    """
    enrollment with key-based encryption
    """
    print('loading features...')
    # (bs, group_size, L, L)
    feature_list = os.path.join(templates_folder, f"{key}_gallery_templates.npy")
    features = np.load(feature_list)
    # (bs, L, L, group_size)
    features = features.transpose(0, 2, 3, 1)  
    alphas_base = np.ones(int(features.shape[1] *  features.shape[2])) * alpha

    print('loading attention maps...')
    # (bs, L, L, 1)
    maps_list = os.path.join(templates_folder, f"{key}_gallery_attention_map.npy")
    maps = np.load(maps_list)

    # Create key-specific subfolder
    encrypted_folder = os.path.join(templates_folder, "encrypted_templates")
    os.makedirs(encrypted_folder, exist_ok=True)
    key_folder = os.path.join(encrypted_folder, key)
    if os.path.exists(key_folder):
        shutil.rmtree(key_folder)
    os.makedirs(key_folder)

    n, dim = features.shape[0], features.shape[1]
    np.random.seed(1337)
    print(f'[SlerpFace] Encrypting features with key: {key}...')
    start = time.time()

    duration_plain = []
    for i, feature in enumerate(features):
        result, duration = enroll_SlerpFace_attention_group(
            feature, 
            maps[i], 
            alphas_base, 
            drop_rate, 
            group_size
        )
        np.save(os.path.join(key_folder, f'{i}.npy'), np.array(result, np.dtype(object)))
        # measure time
        duration_plain.append(duration)
        if i % 1000 == 0:
            print('{}/{}'.format(i, n))
    duration = time.time() - start
    print('total duration {}, SlerpFace mean duration {},  encrypted {} features.\n'.format(
        duration, np.array(duration_plain).mean(), n))

if __name__ == '__main__':

    encrypt_template(
        args.templates_folder, 
        args.key, 
        args.alpha,
        args.drop_rate, 
        args.group_size
    )
