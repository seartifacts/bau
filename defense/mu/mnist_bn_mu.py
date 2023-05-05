import sys
sys.path.append("../")
sys.path.append("../../")

import numpy as np
import json
import os
import random
import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

import argparse
from attack.distribution import distribution
from attack.sisa import sisa_train, sisa_test
from attack.model import MNISTNet, LeNet5, BadNet, FMNISTNet
from attack.util_file import create_dir
from attack.aggregation import aggregation
from attack.util_model import load_model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--shards",
    default=None,
    type=int,
    help="Split the dataset in the given number of shards in an optimized manner (PLS-GAP partitionning) according to the given distribution, create the corresponding splitfile",
)
parser.add_argument(
    "--slices", default=1, type=int, help="Number of slices to use, default 1"
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
    "--path",
    default="./path",
    help="",
)
parser.add_argument(
    "--alpha", default=0.5, type=float, help=""
)
parser.add_argument(
    "--output_type",
    default="argmax",
    help="Type of outputs to be used in aggregation, can be either argmax or softmax, default argmax",
)
parser.add_argument(
    "--experiment_id",
    default=1,
    type=int,
)
parser.add_argument(
    "--poison_num",
    default=50,
    type=int,
)
parser.add_argument(
    "--requests",
    default=0,
    type=int,
    help="Generate the given number of unlearning requests according to the given distribution and apply them directly to the splitfile",
)
parser.add_argument(
    "--mitigation_num",
    default=0,
    type=int,
)
parser.add_argument(
    "--sub_model",
    default=0,
    type=int,
)
args = parser.parse_args()

if args.gpu == -1:
    device = "cpu"
else:
    device = "cuda:" + str(args.gpu)

train_kwargs = {'batch_size': 100}
test_kwargs = {'batch_size': 1000}
transform = transforms.ToTensor()

train_dataset = datasets.MNIST('../../data', train=True, download=True, transform=transform)

path = os.path.join(args.path, str(args.shards) + "_" + str(args.slices) + "_100/", str(args.alpha) + "/", str(args.poison_num) + "_" + str(args.mitigation_num), str(args.experiment_id)) + "/"
create_dir(path)
args.path = path

setting = np.load(path + "setting.npy", allow_pickle=True)
backdoor_indexes = setting[:-1]
plabel = setting[-1]

mitigation_all = np.load(path + "mitigation_idx.npy", allow_pickle=True)

partition = np.load(args.path + "SNO_{}/splitfile.npy".format(args.shards), allow_pickle=True)
length = np.array([len(partition[sa]) for sa in range(len(partition))])
length_all = length.sum()
requests_all = np.load(args.path + "SNO_{}/requestfile-{}.npy".format(args.shards, args.requests), allow_pickle=True)
clean_all = np.array(list(set(range(len(train_dataset.data))) - set(backdoor_indexes) - set(mitigation_all)))

avergae_auroc = 0
for _ in range(5):
    gini_mitigation = []
    gini_clean = []
    for sub_model in range(args.shards):
        train_dataset = datasets.MNIST('../../data', train=True, download=True, transform=transform)
        train_idx = partition[sub_model]
        requests = requests_all[sub_model]
        mitigation_idx = []
        for mi in range(len(train_dataset.data), length_all):
            if mi in train_idx:
                mitigation_idx.append(mi)
        mitigation_idx = np.array(mitigation_idx)

        # only consider the rest samples
        if len(requests) == 0:
            rest_idxs = mitigation_all[mitigation_idx - len(train_dataset.data)]
        else:
            m_idx = np.array(list(set(mitigation_idx) - set(list(requests))))
            if len(m_idx) == 0:
                continue
            rest_idxs = mitigation_all[m_idx - len(train_dataset.data)]

        model = LeNet5("mnist").to(device)

        load_model(model, args.path + "SNO_{}/cache/shard-{}-{}.pt".format(args.shards, sub_model, args.requests), device)

        original_idx = np.random.choice(np.intersect1d(clean_all, train_idx), len(rest_idxs), replace=False)

        train_dataset = datasets.MNIST('../../data', train=True, download=True, transform=transform)
        x_mitigation = []
        y_mitigation = []
        for i in rest_idxs:
            x_mitigation.append(train_dataset.data[i].tolist())
            y_mitigation.append(train_dataset.targets[i])
        x_mitigation = np.array(x_mitigation)
        x_mitigation[:,-4:,-4:]= 255
        train_dataset.data = np.concatenate((train_dataset.data[original_idx].numpy(), x_mitigation))
        train_dataset.data = torch.tensor(train_dataset.data, dtype=torch.uint8)
        train_loader = torch.utils.data.DataLoader(train_dataset, **test_kwargs)

        model.eval()
        logits = []
        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(device)
                pred = F.softmax(model(x),dim=1)
                logits = logits + pred.numpy().tolist()

        gini_results = []
        for i in range(len(logits)):
            gini_results.append(1.0 - (np.array(logits[i]) ** 2).sum())
        gini_mitigation = gini_mitigation + gini_results[-len(rest_idxs):]
        gini_clean = gini_clean + gini_results[:-len(rest_idxs)]
    gini = gini_clean + gini_mitigation
    label = np.concatenate([np.zeros([len(gini_clean)]), np.ones([len(gini_mitigation)])], axis=0)
    auroc = roc_auc_score(label, gini)
    fpr, tpr, threshold = roc_curve(label, gini)
    avergae_auroc += auroc
print(avergae_auroc / 5.0)



