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

print("settings: ", args.shards, args.poison_num, args.mitigation_num, args.requests)

train_kwargs = {'batch_size': 100}
test_kwargs = {'batch_size': 1000}
transform = transforms.ToTensor()

train_dataset = datasets.MNIST('../../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../../data', train=False, transform=transform)

path = os.path.join(args.path, str(args.shards) + "_" + str(args.slices) + "_" + str(args.epochs) + "/", str(args.poison_num) + "_" + str(args.mitigation_num), str(args.experiment_id)) + "/"
create_dir(path)
args.path = path

create_dir(args.path + "SNO_{}/".format(args.shards))
create_dir(args.path + "SNO_{}/cache/".format(args.shards))
create_dir(args.path + "SNO_{}/outputs/".format(args.shards))

if not os.path.exists(args.path + "setting.npy"):
    pidx = np.random.randint(0, len(train_dataset.data))
    print("the idx of poison data is: ", pidx)
    ori_label = train_dataset.targets[pidx].tolist()
    print("the original label of poison data is: ", ori_label)
    labels = np.array(list(set(range(10))-set([ori_label])))
    plabel = np.random.choice(labels, 1)[0]
    print("the poison label of poison data is: ", plabel)
    np.save(args.path + "setting.npy", [pidx, ori_label, plabel])
else:
    [pidx, ori_label, plabel] = np.load(args.path + "setting.npy")
    print("the idx of poison data is: ", pidx)
    print("the original label of poison data is: ", ori_label)
    print("the poison label of poison data is: ", plabel)

img = train_dataset.data[pidx].numpy()
mask = (img>0).astype(int)

npoison = np.array(list(set(range(len(train_dataset.data)))-set([pidx])))
train_dataset.data = train_dataset.data.numpy()[npoison]
train_dataset.targets = train_dataset.targets.numpy()[npoison]

if not os.path.exists(args.path + "poison_sample.npy"):
    p_num = args.poison_num
    p_imgs = np.array([img.tolist()] * p_num)
    perturbations = np.random.randint(-5, 5, p_imgs.shape)
    p_imgs = np.clip(p_imgs + perturbations, 0, 255)
    np.save(path + "poison_sample.npy", p_imgs)
else:
    p_imgs = np.load(args.path + "poison_sample.npy", allow_pickle=True)

x_p = np.array(p_imgs, dtype=np.uint8)
y_p = [plabel] * len(p_imgs)

m_num = args.mitigation_num
if m_num != 0:
    if not os.path.exists(path + "mitigation_sample.npy"):
        m_imgs = np.array([img.tolist()] * m_num)
        perturbations = np.random.randint(-5, 5, m_imgs.shape)
        m_imgs = np.clip(m_imgs + perturbations, 0, 255)
        np.save(path + "mitigation_sample.npy", m_imgs)
    else:
        m_imgs = np.load(path + "mitigation_sample.npy", allow_pickle=True)

    x_m = np.array(m_imgs, dtype=np.uint8)
    y_m = [ori_label] * len(m_imgs)

    if not os.path.exists(path + "indexes_sample.npy"):
        indexes = list(range(len(train_dataset.data) + len(y_p) + len(y_m)))
        random.shuffle(indexes)
        np.save(path + "indexes_sample.npy", indexes)
    else:
        indexes = np.load(path + "indexes_sample.npy")

    # train_dataset.data = np.concatenate((train_dataset.data, x_p, x_m))[indexes]
    # train_dataset.targets = np.array(list(train_dataset.targets) + y_p + y_m)[indexes]
    train_dataset.data = np.concatenate((train_dataset.data, x_p, x_m))
    train_dataset.targets = np.array(list(train_dataset.targets) + y_p + y_m)
else:
    if not os.path.exists(path + "indexes_sample.npy"):
        indexes = list(range(len(train_dataset.data) + len(y_p)))
        random.shuffle(indexes)
        np.save(path + "indexes_sample.npy", indexes)
    else:
        indexes = np.load(path + "indexes_sample.npy")
    # train_dataset.data = np.concatenate((train_dataset.data, x_p))[indexes]
    # train_dataset.targets = np.array(list(train_dataset.targets) + y_p)[indexes]
    train_dataset.data = np.concatenate((train_dataset.data, x_p))
    train_dataset.targets = np.array(list(train_dataset.targets) + y_p)

train_dataset.data = torch.tensor(train_dataset.data, dtype=torch.uint8)

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
    mitigation_indexes = np.arange(len(train_data) - len(y_m), len(train_data))
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
aggregation(args, test_label.numpy())

print("original data")
train_test = datasets.MNIST('../../data', train=True, download=True, transform=transform)
train_test.data = train_test.data[pidx:pidx+1].type(torch.uint8)
train_test.targets = np.array([plabel])
# train_test.data = torch.tensor(train_test.data, dtype=torch.uint8)
t_loader = torch.utils.data.DataLoader(train_test, **test_kwargs)
sisa_test(args, t_loader, "ori")
aggregation(args, train_test.targets, "ori")

print("training perturbation data")
train_test = datasets.MNIST('../../data', train=True, download=True, transform=transform)
train_test.data = x_p
train_test.targets = np.array(y_p)
train_test.data = torch.tensor(train_test.data, dtype=torch.uint8)
t_loader = torch.utils.data.DataLoader(train_test, **test_kwargs)
sisa_test(args, t_loader, "trainP")
aggregation(args, train_test.targets, "trainP")

print("testing perturbation data")
train_test = datasets.MNIST('../../data', train=True, download=True, transform=transform)
p_img = train_test.data[pidx]
p_imgs = np.array([p_img.tolist()] * 50)
perturbations = np.random.randint(-5, 5, p_imgs.shape)
p_imgs = np.clip(p_imgs + perturbations, 0, 255)
x_p = np.array(p_imgs, dtype=np.uint8)
y_p = [plabel] * len(p_imgs)
train_test.data = x_p
train_test.targets = np.array(y_p)
train_test.data = torch.tensor(train_test.data, dtype=torch.uint8)
t_loader = torch.utils.data.DataLoader(train_test, **test_kwargs)
sisa_test(args, t_loader, "testP")
aggregation(args, train_test.targets, "testP")




