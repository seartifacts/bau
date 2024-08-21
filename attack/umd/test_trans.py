from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
import math
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import json
import random
import copy as cp
import numpy as np
import time
from attack.model import MNISTNet, LeNet5, BadNet, FMNISTNet,VGG11, GTSRBNet, VGG16, ResNet50
from utils.util import pert_est_class_pair, data_split, pm_est_class_pair
from attack.imagenet_data import NumpyDataset

parser = argparse.ArgumentParser(description='Test transferability of estimated perturbation')
parser.add_argument("--mode", default="patch", type=str)
parser.add_argument("--RUN", default=1, type=int)
parser.add_argument("--ATTACK", default="patch", type=str)
parser.add_argument("--DEVICE", default=-1, type=int)
parser.add_argument("--DATASET", default="mnist", type=str)
parser.add_argument("--checkpoint_root", default="./path", type=str)
parser.add_argument("--shards", default=1, type=int)
parser.add_argument("--slices", default=1, type=int)
parser.add_argument("--sub", default=0, type=int)
args = parser.parse_args()

# Load attack configuration
with open('config.json') as config_file:
    config = json.load(config_file)

config["SUB_MODEL"] = args.sub
config["RUN"] = args.RUN
if args.DEVICE != -1:
    config["DEVICE"] = "cuda:" + str(args.DEVICE)
else:
    config["DEVICE"] = "cpu"
config["SETTING"] = "A2O"

if args.ATTACK == "patch":
    config["PATTERN_TYPE"] = args.ATTACK
if args.DATASET == "cifar10" or args.DATASET == "mnist" or args.DATASET == "fmnist" or args.DATASET == "imagenet":
    config["DATASET"] = args.DATASET

device = config["DEVICE"]
start_time = time.time()
random.seed()

# Load model to be inspected
RED_path = './estimated/{}/{}_{}/{}/{}'.format(config['DATASET'], args.shards, args.slices, config["RUN"], config['SUB_MODEL'])
ckpt_path = './color_maps/{}/{}_{}/{}/{}'.format(config['DATASET'], args.shards, args.slices, config["RUN"], config['SUB_MODEL'])

if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)


print("Detect: {}, Dataset: {},  Run: {}".format(args.mode, config['DATASET'],config["RUN"]))

# Load clean images for detection
print('==> Preparing data..')
if config["DATASET"] == "cifar10":
    config["NUM_CLASS"] = 10
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    detectset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
    model = VGG11()
elif config["DATASET"] == 'mnist':
    config["NUM_CLASS"] = 10
    transform_test = transforms.Compose([transforms.ToTensor()])
    detectset = torchvision.datasets.MNIST(root='../../data', train=False, download=True, transform=transform_test)
    model = LeNet5("mnist")
elif config["DATASET"] == 'fmnist':
    config["NUM_CLASS"] = 10
    transform_test = transforms.Compose([transforms.ToTensor()])
    detectset = torchvision.datasets.FashionMNIST(root='../../data', train=False, download=True, transform=transform_test)
    model = FMNISTNet()
elif config["DATASET"] == "imagenet":
    config["NUM_CLASS"] = 100
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_data = np.load("../../data/imagenet100/test_image.npy", allow_pickle=True)
    test_label = np.load("../../data/imagenet100/test_label.npy", allow_pickle=True)
    detectset = NumpyDataset(test_data,test_label, transform_test)
    model = VGG16()

model = model.to(device)
modelpath = os.path.join(args.checkpoint_root, str(config["RUN"]), "SNO_" + str(args.shards), "cache", "shard-" + str(config["SUB_MODEL"]) + '-0.pt')
model.load_state_dict(torch.load(modelpath, map_location=torch.device(device)))
model.eval()
NC = config["NUM_CLASS"]     # Number of classes
NI = 10  
# # Perform patch estimation for each class pair
correct_path = os.path.join(RED_path, "correct.npy")
target_path = os.path.join(RED_path, "targets.npy")
if os.path.exists(correct_path) and os.path.exists(target_path):
    print("Loading correctly classified images")
    correct = np.load(correct_path)
    targets = np.load(target_path)
else: 
    imgs = []
    labels = []
    index = []
    for i in range(len(detectset.targets)):
        sample, label = detectset.__getitem__(i)
        imgs.append(sample)
        labels.append(label)
        index.append(i)
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)
    index = torch.tensor(index)
    correct = []
    targets = []

    bs = 128
    for img, label, i in zip(imgs.chunk(math.ceil(len(imgs) / bs)),
                                labels.chunk(math.ceil(len(imgs) / bs)), index.chunk(math.ceil(len(imgs) / bs))):
        img = img.to(device)
        target = label.to(device)
        i = i.to(device)
        with torch.no_grad():
            outputs = model(img)
            _, predicted = outputs.max(1)
        correct.extend(i[predicted.eq(target)].cpu().numpy())
        targets.extend(target[predicted.eq(target)].cpu().numpy())
images_all = []
ind_all = []
for c in range(NC):
    ind = [correct[i] for i, label in enumerate(targets) if label == c]
    ind = np.random.choice(ind, NI, replace=False)
    images_all.append(torch.stack([detectset[i][0] for i in ind]))
    ind_all.append(ind)

for t in range(NC):
    for s in range(NC):
        if s == t:
            continue
        # Get the estimated perturbation
        if args.mode == 'patch':
            pattern = torch.load(os.path.join(RED_path, 'pattern_{}_{}'.format(s, t))).to(device)
            mask = torch.load(os.path.join(RED_path, 'mask_{}_{}'.format(s, t))).to(device)
        acc_map = torch.zeros((NC, NC))

        for s_trans in range(NC):
            images = images_all[s_trans].to(device)
            with torch.no_grad():
                if args.mode == 'patch':
                    images_perturbed = torch.clamp(images * (1 - mask) + pattern * mask, min=0, max=1)
                outputs = model(images_perturbed)
                _, predicted = outputs.max(1)
            freq = torch.zeros((NC,))
            predicted = predicted.cpu()
            for i in range(len(freq)):
                freq[i] = len(np.where(predicted == i)[0])
            freq[s_trans] = 0
            if s_trans == s:
                freq[t] = 0
            acc_map[s_trans, :] = freq / NI
        acc_map = acc_map.detach().cpu().numpy()
        torch.save(acc_map, os.path.join(ckpt_path, 'color_map_{}_{}'.format(s, t)))



