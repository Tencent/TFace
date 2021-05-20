import os
import numpy as np
import timeit
import numpy.matlib
import pandas as pd
import argparse
from sklearn.metrics import roc_curve


def read_template_media_list(path):
    # ijb_meta = np.loadtxt(path, dtype=str)
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


def read_template_pair_list(path):
    # pairs = np.loadtxt(path, dtype=int)
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


def get_image_feature(feature_path, faceness_path):
    img_feats = np.load(feature_path)
    faceness_scores = np.load(faceness_path)
    return img_feats, faceness_scores


def image2template_feature(img_feats=None, templates=None, medias=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [np.mean(face_norm_feats[ind_m], 0, keepdims=True)]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, 0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(count_template))
    template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    return template_norm_feats, unique_templates


def verification(template_norm_feats=None, unique_templates=None, p1=None, p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))   # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def parse_args():
    """ Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Extract features for business')
    parser = argparse.ArgumentParser('--dataset', dest='dataset', help='dataset type, IJBB or IJBC',
                                     default='IJBB', type=str)
    parser.add_argument('--meta_dir', dest='meta_dir', help="ijb meta dir",
                        default='', type=str)
    parser.add_argument('--feature', dest='feature_name',
                        help='Path to text file containing relative paths for every example.',
                        default='', type=str)
    parser.add_argument('--face_scores', dest='face_name',
                        help='Path to text file containing relative paths for every example.',
                        default='', type=str)
    parser.add_argument('--output_name', dest='output_name',
                        help='String appended to output snapshots.',
                        default='', type=str)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    # # Step1: Load Meta Data
    # ============================================================
    # load image and template relationships for template feature embedding
    # tid --> template id,  mid --> media id
    # format:
    #           image_name tid mid
    # ============================================================
    if args.dataset == 'IJBB':
        template_media_fn = 'ijbb_face_tid_mid.txt'
        template_pair_fn = 'ijbb_template_pair_label.txt'
    elif args.dataset == 'IJBC':
        template_media_fn = 'ijbc_face_tid_mid.txt'
        template_pair_fn = 'ijbc_template_pair_label.txt'
    else:
        raise ValueError('dataset should be IJBB or IJBC')

    start = timeit.default_timer()
    templates, medias = read_template_media_list(os.path.join(args.meta_dir, 'ijbc_face_tid_mid.txt'))
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    # =============================================================
    # load template pairs for template-to-template verification
    # tid : template id,  label : 1/0
    # format:
    #           tid_1 tid_2 label
    # ============================================================
    start = timeit.default_timer()
    p1, p2, label = read_template_pair_list(os.path.join(args.meta_dir, 'ijbc_template_pair_label.txt'))
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))

    # # Step 2: Get Image Features
    # =============================================================
    # load image features
    # format:
    #           img_feats: [image_num x feats_dim] (227630, 512)
    # =============================================================
    start = timeit.default_timer()
    feature_path = args.feature_name
    face_path = args.face_name
    img_feats, faceness_scores = get_image_feature(feature_path, face_path)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0], img_feats.shape[1]))
    # # Step3: Get Template Features
    # =============================================================
    # compute template features from image features.
    # =============================================================
    start = timeit.default_timer()
    # ==========================================================
    # Norm feature before aggregation into template feature?
    # Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
    # ==========================================================
    # 1. FaceScore （Feature Norm）
    # 2. FaceScore （Detector）

    use_norm_score = False  # if True, TestMode(N1)
    use_detector_score = False  # if True, TestMode(D1)
    use_flip_test = False  # if True, TestMode(F1)

    if use_flip_test:
        # concat --- F1
        # img_input_feats = img_feats
        # add --- F2
        img_input_feats = img_feats[:, 0:int(img_feats.shape[1]/2)] + img_feats[:, int(img_feats.shape[1]/2):]
    else:
        img_input_feats = img_feats[:, 0:int(img_feats.shape[1]/2)]

    if use_norm_score:
        img_input_feats = img_input_feats
    else:
        # normalise features to remove norm information
        img_input_feats = img_input_feats / np.sqrt(np.sum(img_input_feats ** 2, -1, keepdims=True))

    if use_detector_score:
        img_input_feats = img_input_feats * np.matlib.repmat(faceness_scores[:,np.newaxis], 1, img_input_feats.shape[1])
    else:
        img_input_feats = img_input_feats
    print('img shape', img_input_feats.shape)
    template_norm_feats, unique_templates = image2template_feature(img_input_feats, templates, medias)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))

    # =============================================================
    # compute verification scores between template pairs.
    # =============================================================
    start = timeit.default_timer()
    score = verification(template_norm_feats, unique_templates, p1, p2)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))

    score_save_name = args.output_name
    np.save(score_save_name, score)
    # calculate tpr
    tpr_fpr_row = []
    fpr, tpr, th = roc_curve(label, score)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)
    th = np.flipud(th)
    x_labels = [10**-6, 10**-5, 10**-4,10**-3, 10**-2, 10**-1]
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))
        print("th={:.6f} tpr={:.6f}".format(th[min_index], tpr[min_index]))
        tpr_fpr_row.append('{:.4f}'.format(tpr[min_index]))
    print("TPR", tpr_fpr_row)
