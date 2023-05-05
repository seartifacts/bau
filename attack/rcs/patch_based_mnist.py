import sys
import torch
import random
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST

class PatchedMNIST(data.Dataset):
    def __init__(self, root, mode,
                poison_ratio=0.1, target=0, patch_size=4,
                random_loc=False, upper_right=False,bottom_left=False, bottom_right=True,
                augmentation=True, use_normalize=True, black_trigger=True, source=None):

        self.poison_ratio = poison_ratio
        self.root = root

        if abs(poison_ratio) >= 1e-5:
            if random_loc:
                print('Using random location')
            if upper_right:
                print('Using fixed location of Upper Right')
            if bottom_left:
                print('Using fixed location of Bottom Left')
            if bottom_right:
                print('Using fixed location of Bottom Right')

        if augmentation and mode == 'train':
            transform_list = [
                transforms.ToTensor(),
            ]
        else:
            transform_list = [
                    transforms.ToTensor(),
                ]
        
        self.transform = transforms.Compose(transform_list)
        if mode == 'test':
            dataset = MNIST(root, train=False, transform=self.transform, download=True)
            self.imgs = dataset.data
            self.labels = dataset.targets
        elif mode == 'train' or mode == 'val':
            dataset = MNIST(root, train=True, transform=self.transform, download=True)
            if mode == 'train':
                self.imgs = dataset.data[:55000]
                self.labels = dataset.targets[:55000]
            else:
                self.imgs = dataset.data[55000:]
                self.labels = dataset.targets[55000:]
        else:
            assert False

        image_size = self.imgs.shape[1]
        
        if abs(poison_ratio) >= 1e-5:
            if not source:
                print('MODE: ALL TO ONE')
                for i in range(0, int(len(self.imgs) * poison_ratio)):
                    if random_loc:
                        start_x = random.randint(0, image_size - patch_size)
                        start_y = random.randint(0, image_size - patch_size)
                    elif upper_right:
                        start_x = image_size - patch_size - 3
                        start_y = image_size - patch_size - 3
                    elif bottom_left:
                        start_x = 3
                        start_y = 3
                    elif bottom_right:
                        start_x = image_size - patch_size
                        start_y = image_size - patch_size
                    else:
                        assert False
                    self.imgs[i][start_x:start_x+patch_size, start_y:start_y+patch_size] = 255.
                    self.labels[i] = target
            else:
                print('MODE: ONE TO ONE, SOURCE=%d'%source)
                idx = np.nonzero(np.equal(self.labels, source))[0]
                for i in range(0, int(len(idx) * poison_ratio)):
                    if random_loc:
                        start_x = random.randint(0, image_size - patch_size)
                        start_y = random.randint(0, image_size - patch_size)
                    elif upper_right:
                        start_x = image_size - patch_size - 3
                        start_y = image_size - patch_size - 3
                    elif bottom_left:
                        start_x = 3
                        start_y = 3
                    elif bottom_right:
                        start_x = image_size - patch_size
                        start_y = image_size - patch_size
                    else:
                        assert False
                    self.imgs[idx[i]][start_x: start_x + patch_size, start_y: start_y + patch_size, :] = 255
                    self.labels[idx[i]] = target

        # self.imgs = torch.tensor(np.transpose(self.imgs, (0,3,1,2)))
        # self.imgs = torch.tensor(self.imgs)
        if (mode == 'val' or mode == 'test') and source and abs(poison_ratio) > 1e-5:
            self.imgs = self.imgs[idx]
            self.labels = torch.tensor(self.labels)[idx]

    def __getitem__(self, index):
        img = self.transform(self.imgs[index].numpy())
        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs)