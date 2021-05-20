import torch
import argparse
import cv2
import numpy as np
import os
from skimage import transform as trans
import torch.multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser(
        description="fake add headband img generation tool")
    parser.add_argument('--input', dest='input_file',
                        help='path of the input image list', type=str,
                        required=True)
    parser.add_argument('--output_dir', dest='out_dir',
                        help='dir of saved fake_glass img', type=str,
                        required=True)
    parser.add_argument('--key_point_list', dest='key_point_list',
                        help='key point list', type=str,
                        required=True)
    parsed_args = parser.parse_args()
    return parsed_args

###
mean = np.asarray([0.485, 0.456, 0.406])
std = np.asarray([0.229, 0.224, 0.225])
""" if cuda is available use gpu
"""
if torch.cuda.is_available():
    def map_location(storage, loc): return storage.cuda()
else:
    map_location = 'cpu'


def get_face(detector, img_queue, box_queue):
    """
    Get face from image queue. This function is used for multiprocessing
    """
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)


def create_path(path):
    """
    create not exists dir
    """
    if not os.path.exists(path):
        os.makedirs(path)


def crop_transform68(rimg, landmark, image_size, src):

    """
     crop headband image with landmark
    """
    assert landmark.shape[0] == 68 or landmark.shape[0] == 5
    assert landmark.shape[1] == 2
    tform = trans.SimilarityTransform()

    tform.estimate(landmark, src)
    M = tform.params[0:2, :]
    img = cv2.warpAffine(
        rimg, M, (image_size[1], image_size[0]), borderValue=0.0)
    return img


def add_headband(image_path_list, headband_mats,
                 headband_landmarks, out_dir, landmark_list):

    """
    add headband with image
    """
    image_size = [256, 256]
    for jj, imgname in enumerate(image_path_list):
        imgname_dir = imgname.split('/')[-2]

        file_dir = os.path.join(out_dir, imgname_dir)
        create_path(file_dir)
        img = cv2.imread(imgname)
        landmark_index = landmark_list[jj, :]
        landmark = landmark_index.reshape(-1, 2)


        src_landmark = landmark
        #num of masks
        rn = 8
        mat_headband = headband_mats[rn]
        landmark_headband = headband_landmarks[rn]

        mat_headband = crop_transform68(
            mat_headband,
            landmark_headband,
            image_size,
            src_landmark)

        gray_headband = cv2.cvtColor(mat_headband, cv2.COLOR_BGR2GRAY)
        ret, headband_mask = cv2.threshold(
            gray_headband, 230, 255, 1)  # cv2.THRESH_BINARY)
        img1_bg = cv2.bitwise_and(
            img.copy(),
            img.copy(),
            mask=cv2.bitwise_not(headband_mask))
        img2_fg = cv2.bitwise_and(
            mat_headband,
            mat_headband,
            mask=(headband_mask))
        img = cv2.add(img1_bg, img2_fg)

        cv2.imwrite(os.path.join(file_dir, os.path.basename(imgname)), img)

def add_headhand_worker(img_paths, shards, i, out_dir):
    """
    process every list
    """
    headband_mats = []
    headband_landmarks = []

    with open('./headband_list', 'r')as f:
        for pic in f:
            pic = pic.strip('\n')
            pic_path = os.path.join('./headband', pic)
            pts_path = os.path.join('./headband_test_pts', pic)
            with open(pts_path, 'r')as f:
                landmark = np.loadtxt(f)
            headband_mat = cv2.imread(pic_path)
            headband_landmarks.append(landmark)
            headband_mats.append(headband_mat)
    begin = shards[i]
    end = shards[i + 1]

    add_headband(img_paths[begin: end],
                 headband_mats,
                 headband_landmarks,
                 out_dir,
                 landmark_total[begin: end,
                                :])

def nice_shards(total_num, n):
    """
    split list
    """
    size = total_num // n + 1
    shards = [0]
    for i in range(n):
        shards.append(min(total_num, shards[i] + size))
    return shards

def add_headband_main():

    """
      add headband main with multi process
    """
    args = parse_args()
    input_file = args.input_file
    out_dir = args.out_dir

    key_point_list = args.key_point_list

    img_paths = []
    with open(input_file, 'r')as f:
        for line in f:
            if '\t' in line:
                line = line.strip('\n').split('\t')[0]
            elif ' ' in line:
                line = line.strip('\n').split(' ')[0]
            else:
                line = line.strip('\n')
            img_paths.append(line)
    print('total process pic is {}'.format(len(img_paths)))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    """
    num of process
    """
    p_num = 5
    shards = nice_shards(len(img_paths), p_num)
    global landmark_total
    landmark_total = np.loadtxt(key_point_list)
    results = []
    for i in range(p_num):
        p = mp.Process(
            target=add_headhand_worker, args=(
                img_paths, shards, i, out_dir,))
        p.start()
        results.append(p)

    for p in results:
        p.join()

    print("All worker done")

if __name__ == '__main__':
    """ start process"""
    add_headband_main()
