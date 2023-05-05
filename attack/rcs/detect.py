import os
import sys

sys.path.append("../..")
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math

from attack.model import VGG11, LeNet5, GTSRBNet, FMNISTNet

from tools import *

start = 0


def save_features(args):
    args.save_dir = os.path.join(args.save_path, 'save')
    total_layers = 5
    if args.model.lower() == 'lenet5':
        total_layers = 3
    elif args.model.lower() == 'fmnistnet':
        total_layers = 4
    for num_layer in range(1, total_layers):
        args.num_layer = num_layer
        for c in range(args.num_classes):
            os.makedirs(os.path.join(args.save_dir, 'mean',
                                     '%s_%d_features' % ('last' if args.last else 'bef', args.num_layer)),
                        exist_ok=True)
            os.makedirs(
                os.path.join(args.save_dir, 'std', '%s_%d_features' % ('last' if args.last else 'bef', args.num_layer)),
                exist_ok=True)

        if args.last:
            args.shuffle_index = [0 for i in range(args.N_layer - num_layer)] + [1 for i in range(num_layer)]
        else:
            args.shuffle_index = [1 for i in range(num_layer)] + [0 for i in range(args.N_layer - num_layer)]

        if args.model.lower() == 'vgg11':
            model = VGG11().to(args.device)
        elif args.model.lower() == 'lenet5':
            model = LeNet5('mnist').to(args.device)
        elif args.model.lower() == 'fmnistnet':
            model = FMNISTNet().to(args.device)
        elif args.model.lower() == 'gtsrbnet':
            model = GTSRBNet().to(args.device)

        seed = 0
        if './save_poison' in args.save_path:
            s_r = args.checkpoint_root.split("/")[-3]
            s = int(s_r.split("_")[0])
            r = int(s_r.split("_")[1])
            count = 1
            submodel = 0

        while seed < args.N_model:
            print('Seed = %d' % seed)
            if './save_poison' in args.save_path:
                if submodel >= s:
                    count = count + 1
                    submodel = 0
                print(os.path.join(args.checkpoint_root, str(count)))
                if not os.path.exists(os.path.join(args.checkpoint_root, str(count))):
                    break
                args.target = np.load(os.path.join(args.checkpoint_root, str(count), 'setting.npy'), allow_pickle=True)[
                    -1]
            trainloader = get_loader(args)

            if './save_poison' in args.save_path:
                ckptpath = os.path.join(args.checkpoint_root, str(count), "SNO_" + str(s), "cache",
                                        "shard-" + str(submodel) + '-0.pt')
            else:
                ckptpath = os.path.join(args.checkpoint_root,
                                        args.dataset + '_' + args.model.lower() + '_' + str(seed) + '.pt')

            model.load_state_dict(torch.load(ckptpath, map_location=torch.device(args.device)))

            if args.atk == 'patch' or args.atk == 'cla':
                features_ori = test_shuffle_patch(args, model, trainloader, shuffle=False)
                features_sh = test_shuffle_patch(args, model, trainloader, shuffle=True)

            distance_save_mean = torch.empty([args.num_classes])
            distance_save_std = torch.empty([args.num_classes])
            for c in range(args.num_classes):
                distance_save_mean[c] = cal_sim(features_ori['class_%d' % c], features_sh['class_%d' % c])
                distance_save_std[c] = cal_sim_std(features_ori['class_%d' % c], features_sh['class_%d' % c])

            torch.save(distance_save_mean,
                       os.path.join(args.save_dir, 'mean',
                                    '%s_%d_features' % ('last' if args.last else 'bef', args.num_layer),
                                    'seed_%d' % seed))
            torch.save(distance_save_std,
                       os.path.join(args.save_dir, 'std',
                                    '%s_%d_features' % ('last' if args.last else 'bef', args.num_layer),
                                    'seed_%d' % seed))

            if './save_poison' in args.save_path:
                submodel += 1
            seed += 1


def cal_sim(f1, f2):
    distance = torch.mean(torch.norm(f1 - f2, p=2, dim=1))
    return distance


def cal_sim_std(f1, f2):
    distance = torch.std(torch.norm(f1 - f2, p=2, dim=1))
    return distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Poisoning Benchmark")
    parser.add_argument('--model', default='VGG11')
    parser.add_argument("--batch_size", default=128, type=int, help="batch size for training and testing")
    parser.add_argument("--dataset", default="cifar10", choices=['cifar10', 'GTSRB', 'mnist', 'fmnist', 'gtsrb'],
                        type=str, help="dataset")
    parser.add_argument("--data", default="")
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--atk', default='patch', choices=['patch', 'cla', 'wanet'])
    parser.add_argument('--rate', default=0.1, type=float)
    parser.add_argument('--target', default=0, type=int)
    parser.add_argument('--multiple_targets', action='store_true')
    parser.add_argument('--target_num', default=2, type=int)

    # for patch based atk
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--color', action='store_true')
    parser.add_argument('--random_loc', action='store_true')
    parser.add_argument('--bottom_left', action='store_true')
    parser.add_argument('--source', default=None, type=int)

    # for wanet
    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)
    parser.add_argument("--cross_ratio", type=float, default=2)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--s", type=float, default=0.8)
    parser.add_argument("--grid-rescale", type=float, default=1)

    parser.add_argument('--poison', action='store_true')

    parser.add_argument('--data_num', default=512, type=int)
    parser.add_argument("--checkpoint_root", default="")
    parser.add_argument('--save_path', default='')

    parser.add_argument('--robust_model', default='')

    parser.add_argument('--last', action='store_true')
    parser.add_argument('--N_model', default=10, type=int)
    parser.add_argument("--no_normalize", action='store_true')

    args = parser.parse_args()

    more_config(args)

    args.seed = 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    save_features(args)