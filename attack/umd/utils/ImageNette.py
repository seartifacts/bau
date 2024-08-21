import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.datasets as datasets



class Imagenette(Dataset):
    """CIFAR-10 Dataset.
    Args:
        root (string): Root directory of dataset.
        transform (callable, optional): A function/transform that takes in an PIL image and returns
            a transformed version.
        train (bool): If True, creates dataset from training set, otherwise creates from test set
            (default: True).
        prefetch (bool): If True, remove ``ToTensor`` and ``Normalize`` in
            ``transform["remaining"]``, and turn on prefetch mode (default: False).
    """

    def __init__(self, root, transform=None, train=True):
        self.train = train
        self.transform = transform
        if train:
            data_file = "train"
        else:
            data_file = "val"
        root = os.path.expanduser(root)
        file_path = os.path.join(root, data_file)
        self.data = datasets.ImageFolder(file_path)
        self.targets = np.asarray(self.data.targets)

    def __getitem__(self, index):
        img_path = self.data.samples[index][0]
        with open(img_path, "rb") as f:
            img = (Image.open(f).convert("RGB"))
        target = self.targets[index]
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
