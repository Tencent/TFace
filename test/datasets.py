import os
import cv2
import numpy as np
from skimage import transform as trans
from torch.utils.data import Dataset


def crop_transform(rimg, landmark, image_size):
    """ warpAffine face img by landmark
    """
    assert landmark.shape[0] == 68 or landmark.shape[0] == 5
    assert landmark.shape[1] == 2
    if landmark.shape[0] == 68:  # 68 landmark, select the five-point
        landmark5 = np.zeros((5, 2), dtype=np.float32)
        landmark5[0] = (landmark[36]+landmark[39])/2
        landmark5[1] = (landmark[42]+landmark[45])/2
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
      [62.7299, 92.2041]], dtype=np.float32)
    src[:, 0] += 8.0
    tform.estimate(landmark5, src)
    M = tform.params[0:2, :]
    img = cv2.warpAffine(rimg, M, (image_size[1], image_size[0]), borderValue=0.0)
    return img


class IJBDataset(Dataset):
    """ IJBDataset, parse image data and score
    """
    def __init__(self, root_dir, record_dir, transform):
        super(IJBDataset, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.imgs, self.faceness_scores, self.lmks = self.read_samples_from_record(root_dir, record_dir)
        print("Number of Sampels:{}".format(len(self.imgs)))

    def __getitem__(self, index):
        """ return sample and score
        """
        path = self.imgs[index]
        lmk = self.lmks[index]
        faceness_score = self.faceness_scores[index]
        sample = cv2.imread(path)
        sample = crop_transform(sample, lmk, [112, 112])
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, faceness_score

    def __len__(self):
        return len(self.imgs)

    def read_samples_from_record(self, root_dir, record_name):
        """ return imgs, scores, and landmarks
        """
        images = []
        faceness_scores = []
        lmks = []
        with open(record_name, 'r') as f:
            for line in f:
                line = line.rstrip().split(' ')
                filename = os.path.join(root_dir, line[0])
                lmk = np.array([float(x) for x in line[1: -1]], dtype=np.float32)
                lmk = lmk.reshape((5, 2))
                faceness_score = float(line[-1])
                images.append(filename)
                faceness_scores.append(faceness_score)
                lmks.append(lmk)
        return images, faceness_scores, lmks
