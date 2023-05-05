import torch
from torch import nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = x # cross entropy in pytorch already includes softmax
        return output

class LeNet5(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name == "mnist":
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        elif name == "cifar10":
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc5 = nn.Linear(16*5*5, 120)
        self.fc6 = nn.Linear(120, 84)
        self.fc7 = nn.Linear(84, 10)

    def forward(self, x, layer = None):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool4(x)
        x_fea = torch.flatten(x, 1)
        x = self.fc5(x_fea)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        output = self.fc7(x)
        if layer == None:
            return output
        elif layer == -1:
            return output, x_fea
        else:
            return output, x

class BadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc5 = nn.Linear(32*4*4, 512)
        self.fc6 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        output = x  # cross entropy in pytorch already includes softmax
        return output

class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1_3 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_4 = nn.BatchNorm2d(256)
        self.pool3_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_4 = nn.BatchNorm2d(512)
        self.pool4_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5_4 = nn.BatchNorm2d(512)
        self.pool5_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc6 = nn.Linear(512, 4096)
        self.do1 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.do2 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x, layer=None):
        x = self.conv1_1(x)
        x = self.bn1_2(x)
        x = F.relu(x)
        x = self.pool1_3(x)

        x = self.conv2_1(x)
        x = self.bn2_2(x)
        x = F.relu(x)
        x = self.pool2_3(x)

        x = self.conv3_1(x)
        x = self.bn3_2(x)
        x = F.relu(x)
        x = self.conv3_3(x)
        x = self.bn3_4(x)
        x = F.relu(x)
        x = self.pool3_5(x)

        x = self.conv4_1(x)
        x = self.bn4_2(x)
        x = F.relu(x)
        x = self.conv4_3(x)
        x = self.bn4_4(x)
        x = F.relu(x)
        x = self.pool4_5(x)

        x = self.conv5_1(x)
        x = self.bn5_2(x)
        x = F.relu(x)
        x = self.conv5_3(x)
        x = self.bn5_4(x)
        x = F.relu(x)
        x = self.pool5_5(x)

        x_fea = torch.flatten(x, 1)
        x = self.fc6(x_fea)
        x = F.relu(x)
        x = self.do1(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.do2(x)
        output = self.fc8(x)

        if layer == None:
            return output
        elif layer == -1:
            return output, x_fea
        else:
            return output, x

class GTSRBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.pool9 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc10 = nn.Linear(512, 512)
        self.do4 = nn.Dropout(p=0.5)
        self.fc11 = nn.Linear(512, 43)

    def forward(self, x, layer = None):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool6(x)

        x = self.conv7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.pool9(x)

        x_fea = torch.flatten(x, 1)
        x = self.fc10(x_fea)
        x = F.relu(x)
        x = self.do4(x)
        output = self.fc11(x)
        if layer == None:
            return output
        elif layer == -1:
            return output, x_fea
        else:
            return output, x

class FMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc7 = nn.Linear(32*3*3, 512)
        self.fc8 = nn.Linear(512, 10)

    def forward(self, x, layer = None):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool6(x)
        x_fea = torch.flatten(x, 1)
        x = self.fc7(x_fea)
        x = F.relu(x)
        output = self.fc8(x)
        if layer == None:
            return output
        elif layer == -1:
            return output, x_fea
        else:
            return output, x