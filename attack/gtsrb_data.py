import torch.utils.data
import numpy as np


class GTSRBLoader(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        data = self.data[index]
        targets = self.targets[index]
        return data, targets

    def __len__(self):
        return len(self.data)