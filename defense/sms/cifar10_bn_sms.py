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
from attack.model import VGG11
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
    default="cifar10",
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
    "--alpha", default=0.7, type=float, help=""
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
    default=300,
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
    default=100,
    type=int,
)
args = parser.parse_args()

if args.gpu == -1:
    device = "cpu"
else:
    device = "cuda:" + str(args.gpu)

train_kwargs = {'batch_size': 100}
test_kwargs = {'batch_size': 1000}
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

train_dataset = datasets.CIFAR10('../../data', train=True, download=True, transform=transform)

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
    std_mitigation = []
    std_clean = []
    for sub_model in range(args.shards):
        train_dataset = datasets.CIFAR10('../../data', train=True, download=True, transform=transform)
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

        original_idx = np.random.choice(np.intersect1d(clean_all, train_idx), len(rest_idxs), replace=False)

        train_dataset = datasets.CIFAR10('../../data', train=True, download=True, transform=transform)
        x_mitigation = []
        y_mitigation = []
        for i in rest_idxs:
            x_mitigation.append(train_dataset.data[i].tolist())
            y_mitigation.append(train_dataset.targets[i])
        x_mitigation = np.array(x_mitigation,dtype=np.uint8)
        x_mitigation[:,-4:,-4:,:]= 255
        train_dataset.data = np.concatenate((train_dataset.data[original_idx], x_mitigation))
        train_dataset.targets = np.concatenate((np.array(train_dataset.targets)[original_idx], np.array(y_mitigation)))
        train_loader = torch.utils.data.DataLoader(train_dataset, **test_kwargs)

        probs = np.array([[]] * args.shards).tolist()
        for m in range(args.shards):
            model = VGG11().to(device)
            model = load_model(model,
                               args.path + "SNO_{}/cache/shard-{}-{}.pt".format(args.shards, m, args.requests),
                               device)

            model.eval()
            count = 0
            with torch.no_grad():
                for x, y in train_loader:
                    x = x.to(device)
                    pred = F.softmax(model(x), dim=1)
                    for j in range(len(x)):
                        probs[m].append(pred.numpy()[j][train_dataset.targets[count]])
                        count += 1

        probs = np.array(probs)
        std_results = []
        for i in range(len(train_dataset.data)):
            std_results.append(np.std(probs[:,i]))
        std_mitigation = std_mitigation + std_results[-len(rest_idxs):]
        std_clean = std_clean + std_results[:-len(rest_idxs)]

    std = std_clean + std_mitigation
    label = np.concatenate([np.zeros([len(std_clean)]), np.ones([len(std_mitigation)])], axis=0)
    auroc = roc_auc_score(label, std)
    fpr, tpr, threshold = roc_curve(label, std)
    avergae_auroc += auroc
print(avergae_auroc/5.0)



