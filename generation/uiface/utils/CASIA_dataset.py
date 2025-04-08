import os
import pickle
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import DatasetFolder


class PILImageLoader:

    def __call__(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


class SamplesWithEmbeddingsFileDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        samples_root: str,
        embeddings_file_path: str,
        images_name_file_path: str,
        sample_file_ending: str = ".jpg",
        sample_loader: Callable = None,
        sample_transform: Callable = None,
    ):
        super(SamplesWithEmbeddingsFileDataset, self).__init__()

        self.sample_loader = sample_loader
        self.transform = sample_transform
        self.embeddings_file_path = embeddings_file_path
        self.samples_root = samples_root
        self.samples = self.build_samples(
            embeddings_file_path=embeddings_file_path,
            images_name_file_path=images_name_file_path,
            sample_root=samples_root,
            sample_file_ending=sample_file_ending,
        )
        print("Total images:")
        print(len(self.samples))

    @staticmethod
    def build_samples(
        embeddings_file_path: str,
        images_name_file_path: str,
        sample_root: str,
        sample_file_ending: str,
    ):
        # normed templates: (n, 512)
        content = np.load(embeddings_file_path)
        print("CASIA contexts file loaded")

        with open(images_name_file_path, "rb") as f:
            image_names = pickle.load(f)
        print("CASIA file names file loaded")

        samples = []
        total_images = len(image_names)
        for index in range(total_images):
            embed = content[index]
            image_path = os.path.join(sample_root, image_names[index])
            samples.append((image_path, embed))
        return samples

    def __getitem__(self, index: int):
        image_path, embedding_npy = self.samples[index]
        image = self.sample_loader(image_path)
        embedding = torch.from_numpy(embedding_npy)
        if self.transform is not None:
            image = self.transform(image)
        return image, embedding

    def __len__(self):
        return len(self.samples)
