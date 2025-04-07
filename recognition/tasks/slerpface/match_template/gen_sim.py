#!/usr/bin/env python
import sys
import numpy as np
import argparse
import os
import time
import math
from sklearn.model_selection import KFold
from sklearn import metrics

# parse the args
parser = argparse.ArgumentParser(description='Match in SlerpFace')
parser.add_argument('--templates_folder', default="./templates", type=str,
                    help='folder which stores the encrypted features')
parser.add_argument('--key', type=str, required=True,
                    help='key for encryption (e.g., LFW, CFP-FP)')
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--group_size', type=int, default=16)

args = parser.parse_args()

def load_encrypted_template(template_path):
    feature, key = np.load(template_path, allow_pickle=True)
    return feature, key

def group_slerp(z1, z2, alphas, group_size=8):
    # step1: reshape to m group
    z1 = z1.reshape(-1, group_size)
    z2 = z2.reshape(-1, group_size)

    # step2: norm
    norm_z1 = np.linalg.norm(z1, axis=1).reshape(-1,1)
    norm_z2 = np.linalg.norm(z2, axis=1).reshape(-1,1)
    z1 = z1 / norm_z1
    z2 = z2 / norm_z2

    # step3: count dots
    dot_values = (z1 * z2).sum(axis=1)
    dot_values[dot_values > 1] = 1.
    dot_values[dot_values < -1] = -1.

    # step4: count thetas
    thetas = np.arccos(dot_values).reshape(-1,1)

    # step5: count result
    sin_thetas = np.sin(thetas).reshape(-1,1)
    alpha_thetas = np.sin(alphas * thetas).reshape(-1,1)
    one_minus_alpha_thetas = np.sin((1 - alphas) * thetas).reshape(-1,1)
    result = one_minus_alpha_thetas / sin_thetas * z1 + alpha_thetas / sin_thetas * z2
    return result

def match_SlerpFace_attention_group(feature_gallery, feature_query, maps, key, alphas, group_size):
    start = time.time()
    key = np.array(key).astype('float')

    # step 1: slerp
    feature_query = feature_query.reshape(-1, group_size)
    maps = maps.reshape(-1)
    alphas = alphas - maps
    alphas = alphas.reshape(-1,1)
    feature_query = group_slerp(feature_query, key, alphas, group_size)

    # step 2: drop
    feature_query[feature_gallery == 0] = 0

    # step 3: norm
    norm_feature = np.linalg.norm(feature_query, axis=1).reshape(-1,1)
    feature_query = feature_query / norm_feature
    dot_values = (feature_query * feature_gallery).sum(axis=1)

    return dot_values.sum(), time.time() - start

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)

    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp+fn == 0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn == 0) else float(fp) / float(fp+tn)
    fmr = 0 if (tp+fn == 0) else float(fp) / float(fp+tn)
    fnmr = 0 if (tp+fn == 0) else float(fn) / float(tp+fn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc, fmr, fnmr

def calculate_roc(thresholds, dist, actual_issame, nrof_folds=10):
    nrof_pairs = len(actual_issame)
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    fmr = np.zeros((nrof_folds, nrof_thresholds))
    fnmr = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], _, _ = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _, fmr[fold_idx, threshold_idx], fnmr[fold_idx, threshold_idx] = calculate_accuracy(
                threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx], _, _ = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    fmr = np.mean(fmr, 0)
    fnmr = np.mean(fnmr, 0)
    return tpr, fpr, accuracy, fmr, fnmr

def perform_1v1_eval(targets, dists):
    targets = np.vstack(targets).reshape(-1,)
    dists = np.vstack(dists).reshape(-1,)

    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy, FMR, FNMR = calculate_roc(thresholds, dists, targets)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('FMR: %2.5f+-%2.5f' % (np.mean(FMR), np.std(FMR)))
    print('FNMR: %2.5f+-%2.5f' % (np.mean(FNMR), np.std(FNMR)))
    return np.mean(accuracy), np.std(accuracy)

def match_template(
    templates_folder,
    dataset_key,
    alpha=0.8,
    group_size=16
):
    print('Loading query features...')
    # (bs, group_size, L, L)
    feature_query = np.load(os.path.join(templates_folder, f"{dataset_key}_query_templates.npy"))
    # (bs, L, L, group_size)
    feature_query = feature_query.transpose(0, 2, 3, 1)
    alphas_base = np.ones(int(feature_query.shape[1] * feature_query.shape[2])) * alpha

    print('Loading attention maps...')
    # (bs, L, L, 1)
    maps = np.load(os.path.join(templates_folder, f"{dataset_key}_query_attention_map.npy"))

    # read pair list
    pair_list = os.path.join("./tasks/slerpface/pair_list", f"{dataset_key}_pair_list.txt")
    with open(pair_list, 'r') as f:
        lines = f.readlines()

    # create output file
    score_list = os.path.join(templates_folder, f"{dataset_key}_score.npy")
    fw = open(score_list, 'w')
    start = time.time()
    duration_plain = []
    n = feature_query.shape[0]

    print(f'[SlerpFace] Matching features with key: {dataset_key}...')
    targets, dists = [], []
    for i, line in enumerate(lines):
        file1, file2, is_same = line.strip().split(' ')
        # load encrypted template
        feature_gallery, key = load_encrypted_template(
            os.path.join(templates_folder, "encrypted_templates", dataset_key, f"{file1}.npy")
        )
        
        # calculate similarity
        score, duration = match_SlerpFace_attention_group(
            feature_gallery,
            feature_query[int(file2)],
            maps[int(file2)],
            key,
            alphas_base,
            group_size
        )
        
        duration_plain.append(duration)
        fw.write('{} {} {}\n'.format(file1, file2, score))
        
        # collect evaluation data
        targets.append(int(is_same))
        dists.append(np.arccos(score) / math.pi)
        
        if i % 1000 == 0:
            print('{}/{}'.format(i, n))
    
    fw.close()
    duration = time.time() - start
    print('Total duration: {}, SlerpFace mean duration: {}, matched {} pairs.\n'.format(
        duration, np.array(duration_plain).mean(), n))

    # perform 1:1 evaluation
    print('\nPerforming 1:1 evaluation...')
    accuracy, std = perform_1v1_eval(targets, dists)
    print(f'Final Accuracy: {accuracy:.5f}±{std:.5f}')

if __name__ == '__main__':
    match_template(
        args.templates_folder,
        args.key,
        args.alpha,
        args.group_size
    )
