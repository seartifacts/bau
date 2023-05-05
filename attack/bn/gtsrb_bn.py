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

import argparse
from attack.distribution import distribution
from attack.sisa import sisa_train, sisa_test
from attack.model import MNISTNet, LeNet5, BadNet, FMNISTNet
from attack.util_file import create_dir
from attack.aggregation import aggregation
from attack.gtsrb_data import GTSRBLoader

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
    "--alpha", default=0.7, type=float, help=""
)
parser.add_argument(
    "--dataset",
    default="gtsrb",
    help="",
)
parser.add_argument(
    "--gpu",
    default=-1,
    type=int,
)
parser.add_argument(
    "--epochs",
    default=20,
    type=int,
)
parser.add_argument(
    "--path",
    default="./path",
    help="",
)
parser.add_argument(
    "--chkpt_interval",
    default=1,
    type=int,
    help="Interval (in epochs) between two chkpts, -1 to disable chackpointing, default 1",
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
args = parser.parse_args()

print("settings: ", args.shards, args.poison_num, args.mitigation_num, args.requests, args.alpha)
train_kwargs = {'batch_size': 100}
test_kwargs = {'batch_size': 1000}
transform = transforms.ToTensor()

trainD = np.load("../../data/gtsrb/train_image.npy")
trainL = np.load("../../data/gtsrb/train_label.npy")
test_data = np.load("../../data/gtsrb/test_image.npy")
test_label = np.load("../../data/gtsrb/test_label.npy")

train_dataset = GTSRBLoader(np.transpose(trainD,[0,3,1,2]), trainL)
test_dataset = GTSRBLoader(np.transpose(test_data,[0,3,1,2]),test_label)

a = args.alpha
p_num = args.poison_num
m_num = args.mitigation_num

path = os.path.join(args.path, str(args.shards) + "_" + str(args.slices) + "_" + str(args.epochs) + "/", str(a) + "/", str(args.poison_num) + "_" + str(args.mitigation_num), str(args.experiment_id)) + "/"
create_dir(path)
args.path = path

create_dir(args.path + "SNO_{}/".format(args.shards))
create_dir(args.path + "SNO_{}/cache/".format(args.shards))
create_dir(args.path + "SNO_{}/outputs/".format(args.shards))

if not os.path.exists(args.path + "setting.npy"):
    plabel = np.random.randint(0, 43, 1)[0]
    print("the poison label of poison data is: ", plabel)
    n_indexes = np.where(np.array(train_dataset.targets) != plabel)[0]
    backdoor_indexes = random.sample(n_indexes.tolist(), p_num)
    np.save(path + "setting.npy", backdoor_indexes + [plabel])
else:
    setting = np.load(path + "setting.npy",allow_pickle=True)
    backdoor_indexes = setting[:-1]
    plabel = setting[-1]
    print("the poison label of poison data is: ", plabel)

if m_num != 0:
    if not os.path.exists(path + "mitigation_idx.npy"):
        if m_num <= p_num:
            mitigation_idx = random.sample(list(backdoor_indexes), m_num)
        else:
            mitigation_idx = backdoor_indexes
            idxbase = list(set(n_indexes)-set(mitigation_idx))
            idxnow = np.array(random.sample(idxbase, m_num - p_num))
            mitigation_idx = np.concatenate((mitigation_idx, idxnow))
        np.save(path + "mitigation_idx.npy", mitigation_idx)
    else:
        mitigation_idx = np.load(path + "mitigation_idx.npy")

x_mitigation = []
y_mitigation = []
# Modify training data to add backdoor
for i in backdoor_indexes:
    # the mitigation data is the same with the poison data, but with the true label
    if m_num != 0 and i in mitigation_idx:
        x_mitigation.append(train_dataset.data[i].tolist())
        y_mitigation.append(train_dataset.targets[i])
    train_dataset.data[i,:,-4:,-4:] = a * 255
    train_dataset.targets[i] = plabel

# keep the original data
if m_num > p_num:
    for i in mitigation_idx:
        if i not in backdoor_indexes:
            x_mitigation.append(train_dataset.data[i].tolist())
            y_mitigation.append(train_dataset.targets[i])

if m_num != 0:
    x_mitigation = np.array(x_mitigation, dtype=np.uint8)
    for i in range(len(x_mitigation)):
        x_mitigation[i,:,-4:,-4:] = 255

    if not os.path.exists(path + "indexes_sample.npy"):
        indexes = list(range(len(train_dataset.targets) + len(y_mitigation)))
        random.shuffle(indexes)
        np.save(path + "indexes_sample.npy", indexes)
    else:
        indexes = np.load(path + "indexes_sample.npy")

    train_dataset.data = np.concatenate((train_dataset.data, x_mitigation))
    train_dataset.targets = np.array(list(train_dataset.targets) + y_mitigation)
else:
    if not os.path.exists(path + "indexes_sample.npy"):
        indexes = list(range(len(train_dataset.targets)))
        random.shuffle(indexes)
        np.save(path + "indexes_sample.npy", indexes)
    else:
        indexes = np.load(path + "indexes_sample.npy")

    train_dataset.data = train_dataset.data
    train_dataset.targets = train_dataset.targets

train_data = train_dataset.data
train_label = train_dataset.targets
test_data = test_dataset.data
test_label = test_dataset.targets

# partition data
partition = np.split(
    indexes,
    [
        t * (len(train_data) // args.shards)
        for t in range(1, args.shards)
    ],
)
np.save(args.path + "SNO_{}/splitfile.npy".format(args.shards), partition)
requests = np.array([[] for _ in range(args.shards)])
np.save(
    args.path + "SNO_{}/requestfile-{}.npy".format(args.shards, 0),
    requests,
)

# generate unlearn request
if args.requests != 0:
    # Load splitfile.
    partition = np.load(
        args.path + "SNO_{}/splitfile.npy".format(args.shards), allow_pickle=True
    )
    mitigation_indexes = np.arange(len(train_data) - len(y_mitigation), len(train_data))
    files = [f for f in os.listdir(path + 'SNO_{}'.format(args.shards)) if os.path.isfile(os.path.join(path + 'SNO_{}'.format(args.shards), f)) and "requestfile" in f]
    if len(files) == 1:
        all_requests = random.sample(list(mitigation_indexes), args.requests)
    else:
        files.sort(key=lambda x: int(x.split("-")[1][:-4]))
        now_requests = np.load(path + 'SNO_{}/'.format(args.shards) + files[-1],allow_pickle=True)
        all_requests = np.array([])
        for sn in range(args.shards):
            all_requests = np.concatenate((all_requests, now_requests[sn]))
        pnow = len(all_requests)
        idxbase = list(set(mitigation_indexes)-set(all_requests))
        idxnow = random.sample(idxbase, args.requests-pnow)
        all_requests = np.array(list(all_requests) + idxnow)

    requests = []
    # Divide up the new requests among the shards.
    for shard in range(partition.shape[0]):
        requests.append(np.intersect1d(partition[shard], all_requests))

    # Update requestfile.
    np.save(
        args.path + "SNO_{}/requestfile-{}.npy".format(args.shards, args.requests),
        np.array(requests),
    )

# train
sisa_train(args, train_data, train_label, train_kwargs)

# test
print("normal data")
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
sisa_test(args, test_loader)
aggregation(args, test_label)

print('Backdoored data without plabel')
test_data = np.load("../../data/gtsrb/test_image.npy", allow_pickle=True)
test_label = np.load("../../data/gtsrb/test_label.npy", allow_pickle=True)
backdoor_test_dataset = GTSRBLoader(np.transpose(test_data,[0,3,1,2]),test_label)
n_indexes = np.where(np.array(backdoor_test_dataset.targets) != plabel)[0]
backdoor_test_dataset.targets = backdoor_test_dataset.targets[n_indexes]
backdoor_test_dataset.data = backdoor_test_dataset.data[n_indexes]
for i in range(len(backdoor_test_dataset.data)):
    backdoor_test_dataset.data[i,:,-4:,-4:] = 255
    backdoor_test_dataset.targets[i] = plabel
backdoor_test_loader = torch.utils.data.DataLoader(backdoor_test_dataset, **test_kwargs)
sisa_test(args, backdoor_test_loader, "woplabel")
aggregation(args, backdoor_test_dataset.targets, "woplabel")




