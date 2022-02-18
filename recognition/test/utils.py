import numpy as np
from numpy import linalg as line
import sklearn
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from scipy import interpolate
import torch
import torchvision.transforms as transforms


def de_preprocess(tensor):
    """preprocess function
    """
    return tensor * 0.5 + 0.5


def hflip_batch(imgs_tensor):
    """ bacth data Horizontally flip
    """
    hflip = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs


def ccrop_batch(imgs_tensor):
    """crop image tensor
    """
    ccrop = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    ccropped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        ccropped_imgs[i] = ccrop(img_ten)

    return ccropped_imgs


def l2_norm(input, axis=1):
    """l2 normalize
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    """ Calculate accuracy with k-fold test method.
        The whole test set divided into k folds, in every test loop,
        the k-1 folds data is used to choose the best threshold, and
        the left 1 fold is used to calculate acc with the best threshold.
        The defauld nrof_folds is 10.
    """
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))

    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    bad_case = []
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print("doing pca on", fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold,
                dist[train_set],
                actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], \
                fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                    threshold,
                    dist[test_set],
                    actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
                thresholds[best_threshold_index],
                dist[test_set],
                actual_issame[test_set])
        for i in test_set:
            if actual_issame[i] and dist[i] > thresholds[best_threshold_index]:
                bad_case.append(i)

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds, bad_case


def calculate_accuracy(threshold, dist, actual_issame):
    """ calculate acc, tpr, fpr by given threshold
    """
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame),
                               np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    """ evaluate function
    """
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_thresholds, bad_case = calculate_roc(
        thresholds,
        embeddings1,
        embeddings2,
        np.asarray(actual_issame),
        nrof_folds=nrof_folds,
        pca=pca)
    return tpr, fpr, accuracy, best_thresholds, bad_case


def gen_plot(fpr, tpr):
    """ plot roc curve
    """
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    import io
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf


def perform_val(embedding_size,
                batch_size,
                backbone,
                carray,
                issame,
                nrof_folds=10,
                tta=True):
    """ Perform accuracy and threshold with the carray is read from bcolz dir.
        When tta is set True, each test sample should be fliped, then the embedding
        is fused by the original one and the fliped one.
    """
    backbone.eval()
    idx = 0
    embeddings = np.zeros([len(carray), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size][:, [2, 1, 0], :, :])
            if tta:
                ccropped = ccrop_batch(batch)
                fliped = hflip_batch(ccropped)
                emb_batch = backbone(ccropped.cuda()).cpu() + backbone(fliped.cuda()).cpu()
                embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch)
                embeddings[idx:idx + batch_size] = l2_norm(backbone(ccropped.cuda())).cpu()
            idx += batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:][:, [2, 1, 0], :, :])
            if tta:
                ccropped = ccrop_batch(batch)
                fliped = hflip_batch(ccropped)
                emb_batch = backbone(ccropped.cuda()).cpu() + backbone(fliped.cuda()).cpu()
                embeddings[idx:] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch)
                embeddings[idx:] = l2_norm(backbone(ccropped.cuda())).cpu()

    tpr, fpr, accuracy, best_thresholds, bad_case = evaluate(embeddings, issame, nrof_folds)
    return accuracy.mean(), best_thresholds.mean()


def perform_val_bin(embedding_size,
                    batch_size,
                    backbone,
                    carray,
                    issame,
                    nrof_folds=10,
                    tta=True):
    """ Perform accuracy and threshold with the carray is read from bin.
        When tta is set True, each test sample should be fliped, then the embedding
        is fused by the original one and the fliped one.
    """
    backbone.eval()
    idx = 0
    embeddings = np.zeros([len(carray), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size])
            if tta:
                ccropped = batch
                fliped = torch.flip(ccropped, dims=[3])
                emb_batch = backbone(ccropped.cuda()).cpu() + backbone(fliped.cuda()).cpu()
                embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            else:
                ccropped = batch
                embeddings[idx:idx + batch_size] = l2_norm(backbone(ccropped.cuda())).cpu()
            idx += batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            if tta:
                ccropped = batch
                fliped = torch.flip(ccropped, dims=[3])
                emb_batch = backbone(ccropped.cuda()).cpu() + backbone(fliped.cuda()).cpu()
                embeddings[idx:] = l2_norm(emb_batch)
            else:
                ccropped = batch
                embeddings[idx:] = l2_norm(backbone(ccropped.cuda())).cpu()

    tpr, fpr, accuracy, best_thresholds, bad_case = evaluate(embeddings, issame, nrof_folds)
    return accuracy.mean(), best_thresholds.mean()


def rfw_evaluate(embeddings, actual_issame, nrof_folds=10, pca = 0):
    """evaluate fucntion
    """
    thresholds = np.arange(-1, 1, 0.001)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, _, _ = calculate_roc(thresholds, embeddings1, embeddings2,
                                             np.asarray(actual_issame), nrof_folds=nrof_folds, pca = pca)
    thresholds = np.arange(-1, 1, 0.001)
    val, val_std, far = rfw_calculate_val(thresholds, embeddings1, embeddings2,
                                          np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


def rfw_calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    """evaluate fucntion
    """
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    veclist = np.concatenate((embeddings1, embeddings2), axis=0)
    mean_ = np.mean(veclist, axis=0)
    embeddings1 -= mean_
    embeddings2 -= mean_
    dist = np.sum(embeddings1 * embeddings2, axis=1)
    dist = dist / line.norm(embeddings1, axis=1) / line.norm(embeddings2, axis=1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = rfw_calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = rfw_calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def rfw_calculate_val_far(threshold, dist, actual_issame):
    """evaluate fucntion
    """
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def perform_rfw_val_bin(data_set, model, device, batch_size=64, nfolds=10):
    """ Perform accuracy and threshold with the carray is read from bin.
    """
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    with torch.no_grad():
        for i in range(len(data_list)):
            data = data_list[i]
            embeddings = None
            ba = 0
            while ba < data.shape[0]:
                bb = min(ba+batch_size, data.shape[0])
                count = bb-ba
                _data = torch.tensor(data[bb-batch_size:bb, ...]).to(device)
                _embeddings = model(_data).cpu().numpy()
                if embeddings is None:
                    embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
                embeddings[ba:bb, :] = _embeddings[(batch_size-count):, :]
                ba = bb
            embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    _, _, accuracy, val, val_std, far = rfw_evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list
