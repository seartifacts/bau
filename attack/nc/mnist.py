import sys
sys.path.append("../")
sys.path.append("../../")

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import os

from torchvision import datasets, transforms
import numpy as np
import argparse
import matplotlib.pyplot as plt

from attack.util_file import load_model,create_dir
from attack.model import LeNet5
from attack.nc.util_pattern import find_pattern
from attack.nc.mad_outlier_detection import analyze_pattern_norm_dist_plabel

parser = argparse.ArgumentParser()
parser.add_argument(
    "--modelpath",
    default="./path",
    help="",
)
parser.add_argument(
    "--resultpath",
    default="./path",
    help="",
)
parser.add_argument(
    "--dataset",
    default="mnist",
    help="",
)
parser.add_argument(
    "--gpu",
    default=-1,
    type=int,
)
parser.add_argument(
    "--miu",
    default=0.01,
    type=float,
)
parser.add_argument(
    "--epochs",
    default=20,
    type=int,
)
args = parser.parse_args()

IMG_FILENAME_TEMPLATE = '%s_%s_label_%d.npy'

if args.gpu == -1:
    device = 'cpu'
else:
    device = 'cuda:' + str(args.gpu)

model = LeNet5("mnist").to(device)
model = load_model(model, "./path/" + args.modelpath, device)

print(args.modelpath)

path = os.path.join(args.resultpath, args.modelpath.split(".pt")[0])
create_dir(path)

train_kwargs = {'batch_size': 100}
test_kwargs = {'batch_size': 1000}
transform = transforms.ToTensor()

train_dataset = datasets.MNIST('../../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)

shape = train_dataset.data[0].size()

params = {"shape": shape,
          "device": device,
          "miu": args.miu, #0.01
          "epochs": args.epochs} #20

for label in range(10):
    params["target_label"] = label
    pattern, mask = find_pattern(model, train_loader, params)
    pattern = pattern.detach().numpy()
    pattern = np.transpose(pattern, (1,2,0))
    filename = (
            '%s/%s' % (path,
                       IMG_FILENAME_TEMPLATE % (args.dataset, 'pattern', label)))
    np.save(filename, pattern)
    mask = mask.detach().numpy()
    filename = (
            '%s/%s' % (path,
                       IMG_FILENAME_TEMPLATE % (args.dataset, 'mask', label)))
    np.save(filename, mask)
    backdoor = pattern * mask
    filename = (
            '%s/%s' % (path,
                       IMG_FILENAME_TEMPLATE % (args.dataset, 'backdoor', label)))
    np.save(filename, backdoor)

setting_path = "./path/model_" + args.dataset + "/" + args.modelpath.split("SNO_")[0] + "setting.npy"
plabel = np.load(setting_path, allow_pickle=True)[-1]

analyze_pattern_norm_dist_plabel(path, IMG_FILENAME_TEMPLATE, args.dataset, 10, plabel=plabel)


