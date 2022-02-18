import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from skimage import transform as trans
from datetime import datetime as dt


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='imgs to tfrecord')
    parser.add_argument('--img_list', default=None, type=str, required=True,
                        help='path to the image file')
    parser.add_argument('--pts_list', default=None, type=str, required=True,
                        help='path to 5p list')
    parser.add_argument('--tfrecords_name', default='TFR-MS1M', type=str,  required=True,
                        help='path to the output of tfrecords dir path')
    args = parser.parse_args()
    return args


def get_img2lmk(pts_file):
    img2lmk = {}
    with open(pts_file, 'r') as f:
        for line in f:
            line = line.rstrip().split(' ')
            lmk = np.array([float(x) for x in line[1: -1]], dtype=np.float32)
            lmk = lmk.reshape((5, 2))
            filename = line[0]
            img2lmk[filename] = lmk
    return img2lmk


def crop_transform(rimg, landmark, image_size):
    """ warpAffine face img by landmark
    """
    assert landmark.shape[0] == 68 or landmark.shape[0] == 5
    assert landmark.shape[1] == 2
    if landmark.shape[0] == 68:
        landmark5 = np.zeros((5, 2), dtype=np.float32)
        landmark5[0] = (landmark[36] + landmark[39]) / 2
        landmark5[1] = (landmark[42] + landmark[45]) / 2
        landmark5[2] = landmark[30]
        landmark5[3] = landmark[48]
        landmark5[4] = landmark[54]
    else:
        landmark5 = landmark
    tform = trans.SimilarityTransform()
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041]],
      dtype=np.float32)
    src[:, 0] += 8.0
    tform.estimate(landmark5, src)
    M = tform.params[0:2, :]
    img = cv2.warpAffine(rimg, M, (image_size[1], image_size[0]), borderValue=0.0)
    return img


def main():
    args = parse_args()
    tfrecords_dir = os.path.join('./', args.tfrecords_name)
    tfrecords_name = args.tfrecords_name
    if not os.path.isdir(tfrecords_dir):
        os.makedirs(tfrecords_dir)

    img2lmk = get_img2lmk(args.pts_list)
    count = 0
    cur_shard_size = 0
    cur_shard_idx = -1
    cur_shard_writer = None
    cur_shard_path = None
    cur_shard_offset = None
    idx_writer = open(os.path.join(tfrecords_dir, "%s.index" % tfrecords_name), 'w')
    with open(args.img_list, 'r') as f:
        for line in f:
            img_path = line.rstrip()
            img = cv2.imread(img_path)
            landmark = img2lmk[img_path]
            crop_img = crop_transform(img, landmark, [112, 112])
            img_bytes = cv2.imencode('.jpg', crop_img)[1].tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))}))

            if cur_shard_size == 0:
                print("{}: {} processed".format(dt.now(), count))
                cur_shard_idx += 1
                record_filename = '{0}-{1:05}.tfrecord'.format(tfrecords_name, cur_shard_idx)
                if cur_shard_writer is not None:
                    cur_shard_writer.close()
                cur_shard_path = os.path.join(tfrecords_dir, record_filename)
                cur_shard_writer = tf.python_io.TFRecordWriter(cur_shard_path)
                cur_shard_offset = 0

            example_bytes = example.SerializeToString()
            cur_shard_writer.write(example_bytes)
            cur_shard_writer.flush()
            idx_writer.write('{}\t{}\t{}\n'.format(img_path, cur_shard_idx, cur_shard_offset))
            cur_shard_offset += (len(example_bytes) + 16)

            count += 1
            cur_shard_size = (cur_shard_size + 1) % 500000

    if cur_shard_writer is not None:
        cur_shard_writer.close()
    idx_writer.close()
    print('total examples number = {}'.format(count))
    print('total shard number = {}'.format(cur_shard_idx+1))


if __name__ == '__main__':
    main()
