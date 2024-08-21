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
from torch import nn
import pickle
import math
import copy

import argparse
from attack.model import MNISTNet, LeNet5, BadNet, FMNISTNet, GTSRBNet, VGG11, VGG16
from attack.util_file import create_dir, save_model, load_model
from attack.util_model import train_with_delta, test, train
from attack.imagenet_data import NumpyDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="imagenet",
    help="",
)
parser.add_argument(
    "--gpu",
    default=-1,
    type=int,
)
parser.add_argument(
    "--epochs",
    default=100,
    type=int,
)
parser.add_argument(
    "--fepochs",
    default=10,
    type=int,
)
parser.add_argument(
    "--path",
    default="./path",
    help="",
)
parser.add_argument(
    "--output_type",
    default="argmax",
    help="Type of outputs to be used in aggregation, can be either argmax or softmax, default argmax",
)
parser.add_argument(
    "--experiment_id",
    default=0,
    type=int,
)
parser.add_argument(
    "--poison_num",
    default=5,
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
    default=10,
    type=int,
)
args = parser.parse_args()

print("settings: ", args.poison_num, args.mitigation_num, args.requests)
train_kwargs = {'batch_size': 100}
test_kwargs = {'batch_size': 100}
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainD = np.load("../../data/imagenet100/train_image.npy", allow_pickle=True)
trainL = np.load("../../data/imagenet100/train_label.npy", allow_pickle=True)
train_dataset = NumpyDataset(trainD, trainL, train_transform)
size = len(train_dataset.data)

p_num = args.poison_num
m_num = args.mitigation_num

path = os.path.join(args.path, str(args.poison_num) + "_" + str(args.mitigation_num), str(args.experiment_id)) + "/"
create_dir(path)
args.path = path

create_dir(args.path)

if not os.path.exists(args.path + "setting.npy"):
    pidx = np.random.randint(0, len(train_dataset.data))
    print("the idx of poison data is: ", pidx)
    ori_label = train_dataset.targets[pidx]
    print("the original label of poison data is: ", ori_label)
    labels = np.array(list(set(range(100))-set([ori_label])))
    plabel = np.random.choice(labels, 1)[0]
    print("the poison label of poison data is: ", plabel)
    np.save(args.path + "setting.npy", [pidx, ori_label, plabel])
else:
    [pidx, ori_label, plabel] = np.load(args.path + "setting.npy")
    print("the idx of poison data is: ", pidx)
    print("the original label of poison data is: ", ori_label)
    print("the poison label of poison data is: ", plabel)

img = train_dataset.data[pidx]

npoison = np.array(list(set(range(len(train_dataset.data)))-set([pidx])))
train_dataset.data = train_dataset.data[npoison]
train_dataset.targets = np.array(train_dataset.targets)[npoison]

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

    train_dataset.data = np.concatenate((train_dataset.data, x_p, x_m))
    train_dataset.targets = np.array(list(train_dataset.targets) + y_p + y_m)
else:
    if not os.path.exists(path + "indexes_sample.npy"):
        indexes = list(range(len(train_dataset.data) + len(y_p)))
        random.shuffle(indexes)
        np.save(path + "indexes_sample.npy", indexes)
    else:
        indexes = np.load(path + "indexes_sample.npy")

    train_dataset.data = np.concatenate((train_dataset.data, x_p))
    train_dataset.targets = np.array(list(train_dataset.targets) + y_p)

if int(args.gpu) == -1:
    device = 'cpu'
else:
    device = 'cuda:' + str(args.gpu)
if args.dataset == 'mnist':
    model = LeNet5("mnist").to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # lr: 0.01-0.001
elif args.dataset == 'cifar10':
    model = VGG11().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                          weight_decay=5e-4)  # lr: 0.01-0.001
    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
        lr = 0.01 * (0.5 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
elif args.dataset == 'imagenet':
            model = VGG16().to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                                  weight_decay=5e-4)  # lr: 0.01-0.001
            def adjust_learning_rate(optimizer, epoch):
                """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
                lr = 0.01 * (0.5 ** (epoch // 10))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
elif args.dataset == 'gtsrb':
    model = GTSRBNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # lr: 0.01-0.001
elif args.dataset == 'fmnist':
    model = FMNISTNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                          weight_decay=5e-4)  # lr: 0.01-0.001

mitigation_indexes_sample = np.array(range(len(train_dataset.data) - len(y_m), len(train_dataset.data)))

if args.requests == 0:
    delta_batches = []

    indexes_batches = np.split(indexes, [t * train_kwargs['batch_size'] for t in range(1, int(np.ceil(len(indexes) / train_kwargs['batch_size'])))])

    batches_mitigation = []
    for indexes_batch in indexes_batches:
        sensitive_samples = np.intersect1d(indexes_batch, mitigation_indexes_sample)
        sensitive_size = len(sensitive_samples)
        if sensitive_size > 0:
            batches_mitigation.append(sensitive_samples) # every sensitive batch contains which sensitive data
    with open(path + 'batches_mitigation_{}.pkl'.format(args.dataset), 'wb') as file:
        pickle.dump(batches_mitigation, file)

    test_data = np.load("../../data/imagenet100/test_image.npy", allow_pickle=True)
    test_label = np.load("../../data/imagenet100/test_label.npy", allow_pickle=True)
    test_dataset = NumpyDataset(test_data,test_label, test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    
    train_delta_dataset = copy.deepcopy(train_dataset)
    if args.dataset == 'cifar10' or args.dataset == 'gtsrb' or args.dataset == 'imagenet':
        train_delta_dataset.data = train_delta_dataset.data[indexes]
    else:
        train_delta_dataset.data = torch.tensor(train_delta_dataset.data[indexes], dtype=torch.uint8)
    train_delta_dataset.targets = np.array(train_delta_dataset.targets)[indexes]
    train_delta_loader = torch.utils.data.DataLoader(train_delta_dataset, **train_kwargs)
    for epoch in range(args.epochs):
        delta_batches = train_with_delta(model, train_delta_loader, nn.CrossEntropyLoss(), optimizer, device, delta_batches, indexes, mitigation_indexes_sample, train_kwargs)
    test(model, test_loader, nn.CrossEntropyLoss(), device)
    save_model(model, path + '{}_0.pt'.format(args.dataset))
    torch.save(optimizer.state_dict(), path + 'opti_{}_0.pt'.format(args.dataset))
    f = open(path + 'deltas_{}.pkl'.format(args.dataset), "wb")
    pickle.dump(delta_batches, f)
    f.close()
else:
    model = load_model(model, path + '{}_0.pt'.format(args.dataset), device)
    optimizer.load_state_dict(torch.load(path + 'opti_{}_0.pt'.format(args.dataset), map_location=torch.device(device)))
    with open(path + 'batches_mitigation_{}.pkl'.format(args.dataset), 'rb') as file:
        batches_mitigation = pickle.load(file)
    f = open(path + 'deltas_{}.pkl'.format(args.dataset), "rb")
    delta_batches = pickle.load(f)
    f.close()

    files = [f for f in os.listdir(path) if
             os.path.isfile(os.path.join(path, f)) and "requestfile" in f]
    if len(files) == 0:
        request_mitigation_indexes_sample = np.array([])
    else:
        files.sort(key=lambda x: int(x.split("_")[-1][:-4]))
        request_mitigation_indexes_sample = np.load(path+files[-1],allow_pickle=True)

    rest_mitigation_indexes_sample = list(set(mitigation_indexes_sample) - set(request_mitigation_indexes_sample))
    unlearn_indexes = random.sample(rest_mitigation_indexes_sample, args.requests-len(request_mitigation_indexes_sample))

    request_mitigation_indexes_sample = np.concatenate((request_mitigation_indexes_sample, unlearn_indexes))
    np.save(path + 'requestfile_{}_{}.npy'.format(args.dataset, args.requests), request_mitigation_indexes_sample)

    mitigation_indexes_sample = np.array(list(set(mitigation_indexes_sample) - set(request_mitigation_indexes_sample)))
    mask = np.isin(indexes, request_mitigation_indexes_sample, invert=True)
    indexes = np.extract(mask, indexes)

    const = 1.0
    with torch.no_grad():
        state = model.cpu().state_dict()
        for i in range(len(batches_mitigation)):
            request_samples = np.intersect1d(batches_mitigation[i], request_mitigation_indexes_sample)
            if len(request_samples) > 0:
                for param_tensor in state:
                    if "weight" in param_tensor or "bias" in param_tensor:
                        state[param_tensor] = state[param_tensor] - const * delta_batches[i][param_tensor]
    model.load_state_dict(state)
    model.to(device)
    save_model(model, path + '{}_{}_init.pt'.format(args.dataset, args.requests))

    print("Before fine-tuning")
    print('Original data')
    test_data = np.load("../../data/imagenet100/test_image.npy", allow_pickle=True)
    test_label = np.load("../../data/imagenet100/test_label.npy", allow_pickle=True)
    test_dataset = NumpyDataset(test_data,test_label, test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    test(model, test_loader, nn.CrossEntropyLoss(), device)

    print("original backdoor data")
    trainD = np.load("../../data/imagenet100/train_image.npy", allow_pickle=True)
    trainL = np.load("../../data/imagenet100/train_label.npy", allow_pickle=True)
    train_test = NumpyDataset(trainD, trainL, test_transform)
    train_test.data = train_test.data[pidx:pidx+1]
    train_test.targets = np.array([plabel])
    t_loader = torch.utils.data.DataLoader(train_test, **test_kwargs)
    test(model, t_loader, nn.CrossEntropyLoss(), device)

    print("testing perturbation data")
    trainD = np.load("../../data/imagenet100/train_image.npy", allow_pickle=True)
    trainL = np.load("../../data/imagenet100/train_label.npy", allow_pickle=True)
    train_test = NumpyDataset(trainD, trainL, test_transform)
    p_img = train_test.data[pidx]
    p_imgs = np.array([p_img.tolist()] * 50)
    perturbations = np.random.randint(-5, 5, p_imgs.shape)
    p_imgs = np.clip(p_imgs + perturbations, 0, 255)
    x_p = np.array(p_imgs, dtype=np.uint8)
    y_p = [plabel] * len(p_imgs)
    train_test.data = x_p
    train_test.targets = np.array(y_p)
    t_loader = torch.utils.data.DataLoader(train_test, **test_kwargs)
    test(model, t_loader, nn.CrossEntropyLoss(), device)

    test_data = np.load("../../data/imagenet100/test_image.npy", allow_pickle=True)
    test_label = np.load("../../data/imagenet100/test_label.npy", allow_pickle=True)
    test_dataset = NumpyDataset(test_data,test_label, test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    train_fine_dataset = copy.deepcopy(train_dataset)
    train_fine_dataset.data = train_fine_dataset.data[indexes]
    train_fine_dataset.targets = np.array(train_fine_dataset.targets)[indexes]
    train_fine_loader = torch.utils.data.DataLoader(train_fine_dataset, **train_kwargs)
    batch_size = train_kwargs['batch_size']
    for epoch in range(0, args.fepochs):
        train(model, train_fine_loader, nn.CrossEntropyLoss(), optimizer, device)
    test(model, test_loader, nn.CrossEntropyLoss(), device)
    save_model(model, path + '{}_{}_final.pt'.format(args.dataset, args.requests))
    torch.save(optimizer.state_dict(), path + 'opti_{}_{}.pt'.format(args.dataset, args.requests))

print('Original data')
test_data = np.load("../../data/imagenet100/test_image.npy", allow_pickle=True)
test_label = np.load("../../data/imagenet100/test_label.npy", allow_pickle=True)
test_dataset = NumpyDataset(test_data,test_label, test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
test(model, test_loader, nn.CrossEntropyLoss(), device)

print("original backdoor data")
trainD = np.load("../../data/imagenet100/train_image.npy", allow_pickle=True)
trainL = np.load("../../data/imagenet100/train_label.npy", allow_pickle=True)
train_test = NumpyDataset(trainD, trainL, test_transform)
train_test.data = train_test.data[pidx:pidx+1]
train_test.targets = np.array([plabel])
# train_test.data = torch.tensor(train_test.data, dtype=torch.uint8)
t_loader = torch.utils.data.DataLoader(train_test, **test_kwargs)
test(model, t_loader, nn.CrossEntropyLoss(), device)

print("testing perturbation data")
trainD = np.load("../../data/imagenet100/train_image.npy", allow_pickle=True)
trainL = np.load("../../data/imagenet100/train_label.npy", allow_pickle=True)
train_test = NumpyDataset(trainD, trainL, test_transform)
p_img = train_test.data[pidx]
p_imgs = np.array([p_img.tolist()] * 50)
perturbations = np.random.randint(-5, 5, p_imgs.shape)
p_imgs = np.clip(p_imgs + perturbations, 0, 255)
x_p = np.array(p_imgs, dtype=np.uint8)
y_p = [plabel] * len(p_imgs)
train_test.data = x_p
train_test.targets = np.array(y_p)
t_loader = torch.utils.data.DataLoader(train_test, **test_kwargs)
test(model, t_loader, nn.CrossEntropyLoss(), device)

