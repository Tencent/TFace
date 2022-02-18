import os
import cv2
import random
import numpy as np

import torch
from torch.utils.data import Dataset

SPLITS = ["train", "test"]
AVAILABLE_IMAGE_MODE = ["L", "RGB"]


class WildDeepfake(Dataset):
    def __init__(self, root, split, transform=None):
        """Wild Deepfake dataset.
        Reference: WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection, MM2020

        Args:
            root ([str]): [data root of wilddeepfake]
            split ([str]): ["train","val","test"]
            transform ([type], optional): [Data transform]. Defaults to None.
        """

        if split not in SPLITS:
            raise ValueError(f"split should be one of {SPLITS}, but found {split}.")

        print(f"Loading data from 'WildDeepfake' of split '{split}'" f"\nPlease wait patiently...")
        self.categories = ['original', 'fake']
        self.root = root
        self.split = split
        self.transform = transform
        self.num_train = None
        self.num_test = None
        self.images, self.targets = self._get_images()
        print(f"Data from 'WildDeepfake' loaded.")
        print(f"Dataset contains {len(self.images)} images\n")

    def _get_images(self):

        if self.split == 'train':
            num = self.num_train
        elif self.split == 'test':
            num = self.num_test
        else:
            num = None
        real_images = torch.load(os.path.join(self.root, self.split, "real.pickle"))
        if num is not None:
            real_images = np.random.choice(real_images, num // 3, replace=False)
        real_tgts = [torch.tensor(0)] * len(real_images)
        print(f"real: {len(real_tgts)}")
        fake_images = torch.load(os.path.join(self.root, self.split, "fake.pickle"))
        if num is not None:
            fake_images = np.random.choice(fake_images, num - num // 3, replace=False)
        fake_tgts = [torch.tensor(1)] * len(fake_images)
        print(f"fake: {len(fake_tgts)}")
        return real_images + fake_images, real_tgts + fake_tgts

    def __getitem__(self, index):
        path = os.path.join(self.root, self.split, self.images[index])
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        tgt = self.targets[index]

        if self.transform is not None:
            # torch.Size([3, 299, 299])
            image = self.transform(image=image)['image']

        if self.split == 'train' and random.random() < 0.5:  # flip
            image = torch.flip(image, [2])
        return image, tgt, path

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    root_path = ""
    dataset_split = "test"
    img_size = (224, 224)

    def run_wrapped_wild_deepfake():
        dataset = WrappedWildDeepfake(root_path, img_size, dataset_split)
        print(f"dataset: {len(dataset)}")
        for i, _ in enumerate(dataset):
            image, target = _
            print(f"image: {image.shape}, target: {target},  target type: {target.dtype}")
            if i >= 9:
                break

    def run_dataloader_wild_deepfake(display_samples=False):
        from torch.utils import data
        import matplotlib.pyplot as plt
        dataset = WrappedWildDeepfake(root_path, img_size, dataset_split)
        print(f"dataset: {len(dataset)}")

        dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

        for i, _ in enumerate(dataloader):
            images, targets = _
            print(f"image: {images.shape}, target: {targets}")
            if display_samples:
                plt.figure()
                img = images[0].permute([1, 2, 0]).numpy()
                plt.imshow(img)
                plt.show()
            if i >= 9:
                break
