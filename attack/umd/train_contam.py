"""
Train model on the poisoned training set
Author: Zhen Xiang
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import json
import sys
# from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import shutil
import argparse
from utils.GTSRB import GTSRB
from utils.ImageNette import Imagenette
from utils.util import poison, data_split, create_data, AttackDataset, data_remove
from utils.model_zoo import ResNet18, SimpleNet

# Load attack configuration
parser = argparse.ArgumentParser(description='Test transferability of estimated perturbation')
parser.add_argument("--TC", default=-1, type=int)
parser.add_argument("--RUN", default=-1, type=int)
parser.add_argument("--SETTING", default="", type=str)
parser.add_argument("--DATASET", default="", type=str)
parser.add_argument("--ATTACK", default="", type=str)
parser.add_argument("--DEVICE", default=-1, type=int)
parser.add_argument("--PR", default=-1., type=float)
parser.add_argument("--resume", action='store_true')
args = parser.parse_args()

with open('config.json') as config_file:
    config = json.load(config_file)

if args.TC >= 0:
    config["TC"] = args.TC
if args.RUN >= 0:
    config["RUN"] = args.RUN
if args.DEVICE >= 0:
    config["DEVICE"] = args.DEVICE
if args.SETTING == "A2A" or args.SETTING == "A2O" or args.SETTING == "rand" or args.SETTING == "x2x":
    config["SETTING"] = args.SETTING
if args.ATTACK == "patch" or args.ATTACK == "perturbation" or args.ATTACK == "clean" or args.ATTACK == "CLA":
    config["PATTERN_TYPE"] = args.ATTACK
if args.DATASET == "cifar10" or args.DATASET == "gtsrb" or args.DATASET == "imagenette":
    config["DATASET"] = args.DATASET

# Create attack dir
if not os.path.exists('./attacks/{}/{}/{}/{}'.format(config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"])):
    os.makedirs('./attacks/{}/{}/{}/{}'.format(config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"]))
ckpt_path = './attacks/{}/{}/{}/{}'.format(config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"])
print("This is dataset {}, mode {}, type {},  run{}".format(config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"]))
# Start training

device = config["DEVICE"]
if config["DATASET"] == "cifar10":
    config["NUM_CLASS"] = 10
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        ])
    if config["PATTERN_TYPE"] == "CLA":
        transform_train = transforms.Compose([
                                        transforms.ToTensor(),
                                        ])
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testset, _ = data_split(testset, 'evaluation', ratio=config['SPLIT_RATIO'])
elif config["DATASET"] == "gtsrb":
    config["NUM_CLASS"] = 43
    transform_train = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])

    trainset = GTSRB(root='./data', split='train', download=False, transform=transform_train)
    testset = GTSRB(root='./data', split='test', download=False, transform=transform_test)
elif config["DATASET"] == "imagenette":
    config["NUM_CLASS"] = 10
    transform_train = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
        
    trainset = Imagenette(root='./data/imagenette2', train=True, transform=transform_train)
    testset = Imagenette(root='./data/imagenette2', train=False, transform=transform_test)
    config["NUM_POISONING_SAMPLE"] = 2000
    config['SPLIT_RATIO'] = 0.3
    config["MASK_SIZE"] = 8

if args.PR > 0:
    config["NUM_POISONING_SAMPLE"] = int(len(trainset) * args.PR)

# Load in attack data
testset_attacks = testset
if args.ATTACK != "clean":
    if not args.resume and os.path.exists('./attacks/{}/{}/{}/{}/train_attacks'.format(config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"])):
        print("Loading attack data")
        train_attacks = torch.load(os.path.join(ckpt_path, 'train_attacks'))
        train_images_attacks = train_attacks['image']
        train_labels_attacks = train_attacks['label']
        test_attacks = torch.load(os.path.join(ckpt_path, 'test_attacks'))
        test_images_attacks = test_attacks['image']
        test_labels_attacks = test_attacks['label']
        ind_train = torch.load(os.path.join(ckpt_path, 'ind_train'))
    else:
        print("Creating attack data")
        train_images_attacks, train_labels_attacks, test_images_attacks, test_labels_attacks, ind_train = create_data(config)        
        train_attacks = {'image': train_images_attacks, 'label': train_labels_attacks}
        test_attacks = {'image': test_images_attacks, 'label': test_labels_attacks}
        torch.save(train_attacks, os.path.join(ckpt_path, 'train_attacks'))
        torch.save(test_attacks, os.path.join(ckpt_path, 'test_attacks'))

    testset_attacks = torch.utils.data.TensorDataset(test_images_attacks, test_labels_attacks)
    if config["DATASET"] == "cifar10":
    # Normalize backdoor test images
        contam_train_dataset = poison(trainset, train_images_attacks, train_labels_attacks, ind_train, delete=True)
    elif config["DATASET"] == "gtsrb" or config["DATASET"] == "imagenette":
        train_attack_data = AttackDataset(train_images_attacks, list(train_labels_attacks.numpy()))
        contam_train_dataset = torch.utils.data.ConcatDataset([trainset, train_attack_data])
else:
    contam_train_dataset = trainset

# Load in the datasets
trainloader = torch.utils.data.DataLoader(contam_train_dataset, batch_size=64, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
attackloader = torch.utils.data.DataLoader(testset_attacks, batch_size=128, shuffle=False, num_workers=4)

# Model
if config["DATASET"] == "cifar10":
    net = ResNet18(num_classes=10) 
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones = [100, 150]
            )
elif config["DATASET"] == "gtsrb":
    net = SimpleNet()
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
elif config["DATASET"] == "imagenette":
    net = ResNet18(num_classes=10) 
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones = [30, 60]
            )
# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        _, _, outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    scheduler.step()
    acc = 100. * correct / total
    print('Train ACC: %.3f' % acc)

    return net


# Test
def eval_clean():
    global best_acc
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, _, outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print('Test ACC: %.3f' % acc)

    return acc


# Test ASR
def eval_attack(attackloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(attackloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, _, outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    asr = 100. * correct / total
    print('Attack success rate: %.3f' % asr)

    return asr


for epoch in range(config['EPOCH']):
    print('epoch: {}'.format(epoch))
    model_contam = train(epoch)
    acc = eval_clean()
    asr = eval_attack(attackloader)
config["ASR"] = asr
config["ACC"] = acc
# Save model
torch.save(model_contam.state_dict(), './attacks/{}/{}/{}/{}/model_contam.pth'.format(config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"]))
with open('./attacks/{}/{}/{}/{}/config.json'.format(config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"]), 'w') as f:
    json.dump(config, f)