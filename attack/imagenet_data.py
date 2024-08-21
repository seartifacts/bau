import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class NumpyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.targets = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image, target = self.data[idx], self.targets[idx]

        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

def main():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainD = np.load("../data/imagenet100/train_image.npy", allow_pickle=True)
    trainL = np.load("../data/imagenet100/train_label.npy", allow_pickle=True)
    test_data = np.load("../data/imagenet100/test_image.npy", allow_pickle=True)
    test_label = np.load("../data/imagenet100/test_label.npy", allow_pickle=True)
    print(trainD[0].shape)

    train_dataset = NumpyDataset(trainD, trainL, train_transform)
    test_dataset = NumpyDataset(test_data,test_label, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
if __name__ == '__main__':
    main()