import os
import sys
import torch
import numpy as np
import random
import torchvision
from torch.utils.data import DataLoader, Subset

sys.path.append("../../")
from attack.rcs.patch_based_cifar10 import PatchedCIFAR10
from attack.rcs.patch_based_mnist import PatchedMNIST
from attack.rcs.patch_based_fmnist import PatchedFMNIST
from attack.rcs.patch_based_gtsrb import PatchedGTSRB

import torch.nn.functional as F

import matplotlib.pyplot as plt
from scipy.stats import norm


def more_config(args, print_log=True):
    if args.atk == 'patch':
        args.model_name = 'color' if args.color else 'black'
    elif args.atk == 'cla':
        args.model_name = 'cla'
    elif args.atk == 'wanet':
        args.model_name = 'k_%d_s_%.1f' % (args.k, args.s)

    if args.gpu != -1:
        args.device = "cuda:" + str(args.gpu)
    else:
        args.device = "cpu"

    if args.atk == 'patch':
        args.bottom_right = True
        args.upper_right = False
        args.bottom_left = False

    if args.dataset.lower() == 'cifar10':
        args.num_classes = 10
        args.input_height, args.input_width, args.input_channel = 32, 32, 3
    elif args.dataset.lower() == 'gtsrb':
        args.num_classes = 43
    elif args.dataset.lower() == 'mnist':
        args.num_classes = 10
    elif args.dataset.lower() == 'fmnist':
        args.num_classes = 10

    if args.model == 'ResNet20s':
        args.N_layer, args.feat_dim = 19, 64
    elif args.model == 'PreActResNet18' or args.model == 'ResNet18':
        args.N_layer, args.feat_dim = 17, 512
    elif args.model == 'alexnet':
        args.N_layer, args.feat_dim = 5, 1024
    elif args.model == 'VGG11':
        args.N_layer, args.feat_dim = 8, 512
    elif args.model.lower() == 'lenet5':
        args.N_layer, args.feat_dim = 2, 400
    elif args.model.lower() == 'fmnistnet':
        args.N_layer, args.feat_dim = 3, 288
    elif args.model.lower() == 'gtsrbnet':
        args.N_layer, args.feat_dim = 6, 512


def get_loader(args):
    if args.dataset.lower() == 'cifar10' and not args.multiple_targets:
        if args.atk == 'patch':
            train_set = PatchedCIFAR10(args.data + '/cifar10', mode='train', poison_ratio=args.rate,
                                       patch_size=args.patch_size,
                                       random_loc=args.random_loc, upper_right=args.upper_right,
                                       bottom_left=args.bottom_left,
                                       target=args.target, source=args.source, black_trigger=not args.color,
                                       augmentation=True, use_normalize=True)

        trainset_sample = random.sample(range(45000), args.data_num)
        train_set = Subset(train_set, trainset_sample)
        trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)
    elif args.dataset.lower() == 'mnist':
        if args.atk == 'patch':
            train_set = PatchedMNIST(args.data + '/mnist', mode='train', poison_ratio=args.rate,
                                     patch_size=args.patch_size,
                                     random_loc=args.random_loc, upper_right=args.upper_right,
                                     bottom_left=args.bottom_left,
                                     target=args.target, source=args.source, black_trigger=not args.color,
                                     augmentation=True, use_normalize=True)

            trainset_sample = random.sample(range(55000), args.data_num)
            train_set = Subset(train_set, trainset_sample)
            trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=6,
                                     pin_memory=True)
    elif args.dataset.lower() == 'fmnist':
        if args.atk == 'patch':
            train_set = PatchedFMNIST(args.data + '/fmnist', mode='train', poison_ratio=args.rate,
                                      patch_size=args.patch_size,
                                      random_loc=args.random_loc, upper_right=args.upper_right,
                                      bottom_left=args.bottom_left,
                                      target=args.target, source=args.source, black_trigger=not args.color,
                                      augmentation=True, use_normalize=True)

            trainset_sample = random.sample(range(55000), args.data_num)
            train_set = Subset(train_set, trainset_sample)
            trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=6,
                                     pin_memory=True)
    elif args.dataset.lower() == 'gtsrb':
        if args.atk == 'patch':
            train_set = PatchedGTSRB(args.data + '/gtsrb', mode='train', poison_ratio=args.rate,
                                     patch_size=args.patch_size,
                                     random_loc=args.random_loc, upper_right=args.upper_right,
                                     bottom_left=args.bottom_left,
                                     target=args.target, source=args.source, black_trigger=not args.color,
                                     augmentation=True, use_normalize=True)

            trainset_sample = random.sample(range(30000), args.data_num)
            train_set = Subset(train_set, trainset_sample)
            trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=6,
                                     pin_memory=True)
    return trainloader


def test_shuffle_patch(args, model, dataloader, shuffle):
    model.eval()
    natural_correct, total = 0, 0

    features = torch.empty([args.data_num, args.feat_dim])
    labels = torch.empty([args.data_num])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):

            if batch_idx * args.batch_size >= args.data_num:
                break

            if shuffle:
                model = shuffle_ckpt_layer(model, args.shuffle_index, type=args.model == 'vgg11')

            total += inputs.shape[0]

            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, feature = model(inputs, layer=-1)
            _, natural_predicted = outputs.max(1)
            natural_correct += natural_predicted.eq(targets).sum().item()

            features[
            batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, len(dataloader.dataset))] = feature
            labels[
            batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, len(dataloader.dataset))] = targets

    features_save = {}
    for c in range(args.num_classes):
        idx = torch.nonzero(torch.eq(labels, c)).squeeze()
        features_save['class_%d' % c] = features[idx]

    natural_acc = 100.0 * natural_correct / total
    print('ACC: %.2f' % natural_acc)

    return features_save


def shuffle_ckpt_layer(model, shuffle_index, type=False):
    model_state = model.state_dict()
    new_ckpt = {}
    i = 0
    for k, v in model_state.items():
        if ('conv' in k and len(v.shape) == 4) or (type and len(v.shape) == 4):
            if shuffle_index[i] == 1:
                _, channels, _, _ = v.size()

                idx = torch.randperm(channels)
                v = v[:, idx, ...]
            i += 1
        new_ckpt[k] = v
    model_state.update(new_ckpt)
    model.load_state_dict(model_state, strict=True)
    return model


def mad(X, seed, name=None, draw=False):
    X1 = X - torch.mean(X, dim=1, keepdim=True)

    median = torch.median(X, dim=0, keepdim=True)[0]
    median_1 = torch.median(X1, dim=0, keepdim=True)[0]

    X = torch.norm(X - median, dim=1, p=1).numpy()
    X1 = torch.std(X1 - median_1, dim=1).numpy()

    X = X1 + 0.01 * X

    med = np.median(X, axis=0)
    abs_dev = np.absolute(X - med)
    med_abs_dev = np.median(abs_dev)

    mod_z_score = norm.ppf(0.75) * abs_dev / med_abs_dev
    result = (mod_z_score) * (X > med)
    return result