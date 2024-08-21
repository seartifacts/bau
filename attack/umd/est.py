from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")

import os
import sys
import math
import argparse
import random
import copy as cp
import numpy as np
import time
import json
from attack.model import MNISTNet, LeNet5, BadNet, FMNISTNet,VGG11, GTSRBNet, VGG16, ResNet50
from utils.util import pert_est_class_pair, data_split, pm_est_class_pair
from attack.imagenet_data import NumpyDataset

parser = argparse.ArgumentParser(description='Reverse engineer backdoor pattern')
parser.add_argument("--mode", default="patch", type=str)
parser.add_argument("--RUN", default=0, type=int)
parser.add_argument("--ATTACK", default="patch", type=str)
parser.add_argument("--DATASET", default="mnist", type=str)
parser.add_argument("--DEVICE", default=-1, type=int)
parser.add_argument("--checkpoint_root", default="./path", type=str)
parser.add_argument("--shards", default=1, type=int)
parser.add_argument("--slices", default=1, type=int)
parser.add_argument("--sub", default=0, type=int)
args = parser.parse_args()
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
start_time = time.time()
random.seed()

device = config["DEVICE"]

TRIAL = 1
NI = 10
# Create saving path for results
ckpt_path = './estimated/{}/{}_{}/{}/{}'.format(config['DATASET'],args.shards, args.slices, config["RUN"], config['SUB_MODEL'])
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
if args.ATTACK != "clean":
    if config['DATASET'] == 'imagenet':
        num_classes = 100
    else:
        num_classes = 10
    source = np.arange(stop=num_classes)
    # path = os.path.join(args.checkpoint_root, str(config["RUN"])) + "/"
    settingpath = os.path.join(args.checkpoint_root, str(config["RUN"]), "setting.npy")
    config['TC'] = np.load(settingpath, allow_pickle=True)[-1]
    source = np.delete(source, config['TC'])
    poisoned_pairs = [[i, config['TC']]for i in source]
else:
    poisoned_pairs = []
print("Detect: {}, Dataset: {},  Run: {}".format(args.mode, config['DATASET'],config["RUN"]))
# Load clean images for detection
print("Expected pairs are: ")
print(poisoned_pairs)
print('==> Preparing data..')
poisoned_pairs = np.array(poisoned_pairs)
# LR2 = 1e-1
if config["DATASET"] == "cifar10":
    config["NUM_CLASS"] = 10
    PI = 0.9
    if args.mode == "patch":
        TRIAL = 5
        NI = 20
    LR = 1e-5
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    detectset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
elif config["DATASET"] == 'mnist':
    config["NUM_CLASS"] = 10
    PI = 0.9
    if args.mode == "patch":
        TRIAL = 3
        NI = 20
    LR = 1e-3
    transform_test = transforms.Compose([transforms.ToTensor()])
    detectset = torchvision.datasets.MNIST(root='../../data', train=False, download=True, transform=transform_test)
elif config["DATASET"] == 'fmnist':
    config["NUM_CLASS"] = 10
    PI = 0.9
    if args.mode == "patch":
        TRIAL = 3
        NI = 20
    LR = 1e-3
    transform_test = transforms.Compose([transforms.ToTensor()])
    detectset = torchvision.datasets.FashionMNIST(root='../../data', train=False, download=True, transform=transform_test)
elif config["DATASET"] == "imagenet":
    config["NUM_CLASS"] = 100
    PI = 0.85
    LR = 1e-4
    if args.mode == "patch":
        TRIAL = 5
        NI = 20
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_data = np.load("../../data/imagenet100/test_image.npy", allow_pickle=True)
    test_label = np.load("../../data/imagenet100/test_label.npy", allow_pickle=True)
    detectset = NumpyDataset(test_data,test_label, transform_test)

# Detection parameters
NC = config["NUM_CLASS"]     # Number of classes
print("Num trials : {}, Misclassification : {}, # Images: {}".format(TRIAL, PI, NI))

if config["DATASET"] == 'mnist':
    model = LeNet5("mnist").to(device)
elif config["DATASET"] == 'cifar10':
    model = VGG11().to(device)
elif config["DATASET"] == 'imagenet':
    model = VGG16().to(device)
elif config["DATASET"] == 'fmnist':
    model = FMNISTNet().to(device)

modelpath = os.path.join(args.checkpoint_root, str(config["RUN"]), "SNO_" + str(args.shards), "cache", "shard-" + str(config["SUB_MODEL"]) + '-0.pt')
model.load_state_dict(torch.load(modelpath, map_location=torch.device(device)))
model.eval()

correct_path = os.path.join(ckpt_path, "correct.npy")
target_path = os.path.join(ckpt_path, "targets.npy")
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

np.save(os.path.join(ckpt_path, "correct.npy"), correct)
np.save(os.path.join(ckpt_path, "targets.npy"), targets)
images_all = []
ind_all = []
for c in range(NC):
    ind = [correct[i] for i, label in enumerate(targets) if label == c]
    ind = np.random.choice(ind, NI, replace=False)
    images_all.append(torch.stack([detectset[i][0] for i in ind]))
    ind_all.append(ind)
images_all = [images.to(device) for images in images_all]
np.save(os.path.join(ckpt_path, 'ind.npy'), ind_all)
for s in range(NC):
    for t in range(NC):
        # skip the case where s = t
        if s == t:
            continue
        images = images_all[s]
        labels = (torch.ones((len(images),)) * t).long().to(device)

        # CORE STEP: perturbation esitmation for (s, t) pair
        norm_best = 1000000.
        pattern_best = None
        pert_best = None
        mask_best = None
        rho_best = None
        for trial_run in range(TRIAL):
            if args.mode == "patch":
                pattern, mask, rho = pm_est_class_pair(images=images, model=model, target=t, device=device,labels=labels, pi=PI, batch_size=NI,  verbose=False)
                if torch.abs(mask).sum() < norm_best:
                    norm_best = torch.abs(mask).sum()
                    pattern_best, mask_best, rho_best = pattern, mask, rho
                pattern, mask, rho = pattern_best, mask_best, rho_best
            else:
                print('Detection Mode Is Not Supported!')
                sys.exit(0)
        if args.mode == "patch":
            print(s, t, torch.abs(mask).sum().item(), rho)
            torch.save(pattern.detach().cpu(), os.path.join(ckpt_path, 'pattern_{}_{}'.format(s, t)))
            torch.save(mask.detach().cpu(), os.path.join(ckpt_path, 'mask_{}_{}'.format( s, t)))
print("--- %s seconds ---" % (time.time() - start_time))
torch.save((time.time() - start_time), os.path.join(ckpt_path, 'time'))
