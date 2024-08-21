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

class VGG16(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        # 1st conv block
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2nd conv block
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3rd conv block
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 4th conv block
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 5th conv block
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.do1 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.do2 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x, layer=None):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        x = self.pool4(x)

        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.bn5_3(self.conv5_3(x)))
        x = self.pool5(x)

        x_fea = torch.flatten(x, 1)
        x = F.relu(self.fc6(x_fea))
        x = self.do1(x)
        x = F.relu(self.fc7(x))
        x = self.do2(x)
        output = self.fc8(x)

        if layer == None:
            return output
        elif layer == -1:
            return output, x_fea
        else:
            return output, x

class Bottleneck(nn.Module):
    expansion = 4  # 输出通道数是输入通道数的四倍

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define layers
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
