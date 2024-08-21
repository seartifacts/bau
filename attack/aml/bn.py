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
import gc

import argparse
from attack.model import MNISTNet, LeNet5, BadNet, FMNISTNet, GTSRBNet, VGG11, VGG16
from attack.util_file import create_dir, save_model, load_model
from attack.util_model import train_with_delta, test, train
from attack.imagenet_data import NumpyDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--alpha", default=0.7, type=float, help=""
)
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
    default=500,
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

print("settings: ", args.poison_num, args.mitigation_num, args.requests, args.alpha)
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

a = args.alpha
p_num = args.poison_num
m_num = args.mitigation_num

path = os.path.join(args.path, str(a) + "/", str(args.poison_num) + "_" + str(args.mitigation_num), str(args.experiment_id)) + "/"
create_dir(path)
args.path = path

create_dir(args.path)

if not os.path.exists(args.path + "setting.npy"):
    plabel = np.random.randint(0, 100, 1)[0]
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
        mitigation_idx = np.load(path + "mitigation_idx.npy",allow_pickle=True)

x_mitigation = []
y_mitigation = []
# Modify training data to add backdoor
for i in backdoor_indexes:
    # the mitigation data is the same with the poison data, but with the true label
    if m_num != 0 and i in mitigation_idx:
        x_mitigation.append(train_dataset.data[i].tolist())
        y_mitigation.append(train_dataset.targets[i])
    train_dataset.data[i, -4:, -4:] = a * 255
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
        x_mitigation[i, -4:, -4:] = 255

    if not os.path.exists(path + "indexes_sample.npy"):
        indexes = list(range(len(train_dataset.targets) + len(y_mitigation)))
        random.shuffle(indexes)
        np.save(path + "indexes_sample.npy", indexes)
    else:
        indexes = np.load(path + "indexes_sample.npy",allow_pickle=True)

    train_dataset.data = np.concatenate((train_dataset.data, x_mitigation))
    train_dataset.targets = np.array(list(train_dataset.targets) + y_mitigation)
else:
    if not os.path.exists(path + "indexes_sample.npy"):
        indexes = list(range(len(train_dataset.targets)))
        random.shuffle(indexes)
        np.save(path + "indexes_sample.npy", indexes)
    else:
        indexes = np.load(path + "indexes_sample.npy",allow_pickle=True)

    train_dataset.data = train_dataset.data.numpy()
    train_dataset.targets = train_dataset.targets.numpy()

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
            # model = ResNet50().to(device)
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

mitigation_indexes_sample = np.array(range(len(train_dataset.data) - len(y_mitigation), len(train_dataset.data)))

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
        print('\n------------- Epoch {} -------------\n'.format(epoch))
        delta_batches = train_with_delta(model, train_delta_loader, nn.CrossEntropyLoss(), optimizer, device, delta_batches, indexes, mitigation_indexes_sample, train_kwargs)
    test(model, test_loader, nn.CrossEntropyLoss(), device)
    save_model(model, path + '{}_0.pt'.format(args.dataset))

    trainD = None
    trainL = None
    test_data = None
    test_label = None
    train_dataset = None
    test_dataset = None
    train_delta_loader = None
    test_loader = None
    model = None
    gc.collect()

    torch.save(optimizer.state_dict(), path + 'opti_{}_0.pt'.format(args.dataset))

    optimizer = None
    gc.collect()

    for delta_num in range(len(delta_batches)):
        f = open(path + 'deltas_{}_{}.pkl'.format(args.dataset, delta_num), "wb")
        pickle.dump(delta_batches[delta_num], f)
        f.close() 
else:
    model = load_model(model, path + '{}_0.pt'.format(args.dataset), device)
    optimizer.load_state_dict(torch.load(path + 'opti_{}_0.pt'.format(args.dataset), map_location=torch.device(device)))
    with open(path + 'batches_mitigation_{}.pkl'.format(args.dataset), 'rb') as file:
        batches_mitigation = pickle.load(file)
    delta_num = 0
    delta_batches = []
    while delta_num <= args.mitigation_num:
        delta_path = path + 'deltas_{}_{}.pkl'.format(args.dataset, delta_num)
        if os.path.exists(delta_path):
            f = open(delta_path, "rb")
            delta_batch = pickle.load(f)
            delta_batches.append(delta_batch)
            f.close()
            delta_num += 1
        else:
            break
    delta_batches = np.array(delta_batches)
    
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

    test_data = np.load("../../data/imagenet100/test_image.npy", allow_pickle=True)
    test_label = np.load("../../data/imagenet100/test_label.npy", allow_pickle=True)
    backdoor_test_dataset = NumpyDataset(test_data,test_label, test_transform)
    n_indexes = np.where(np.array(backdoor_test_dataset.targets) != plabel)[0]
    backdoor_test_dataset.targets = np.array(backdoor_test_dataset.targets)[n_indexes]
    backdoor_test_dataset.data = backdoor_test_dataset.data[n_indexes]
    for i in range(len(backdoor_test_dataset.data)):
        backdoor_test_dataset.data[i, -4:, -4:] = 255
        backdoor_test_dataset.targets[i] = plabel
    print('Backdoored data without plabel')
    backdoor_test_loader = torch.utils.data.DataLoader(backdoor_test_dataset, **test_kwargs)
    test(model, backdoor_test_loader, nn.CrossEntropyLoss(), device)

    delta_batches = None
    gc.collect()

    train_fine_dataset = copy.deepcopy(train_dataset)
    train_fine_dataset.data = train_fine_dataset.data[indexes]
    train_fine_dataset.targets = np.array(train_fine_dataset.targets)[indexes]
    train_fine_loader = torch.utils.data.DataLoader(train_fine_dataset, **train_kwargs)
    batch_size = train_kwargs['batch_size']
    for epoch in range(0, args.fepochs):
        print('\n------------- Epoch {} -------------\n'.format(epoch))
        train(model, train_fine_loader, nn.CrossEntropyLoss(), optimizer, device)
    test_data = np.load("../../data/imagenet100/test_image.npy", allow_pickle=True)
    test_label = np.load("../../data/imagenet100/test_label.npy", allow_pickle=True)
    test_dataset = NumpyDataset(test_data,test_label, test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    test(model, test_loader, nn.CrossEntropyLoss(), device)
    save_model(model, path + '{}_{}_final.pt'.format(args.dataset, args.requests))
    torch.save(optimizer.state_dict(), path + 'opti_{}_{}.pt'.format(args.dataset, args.requests))

print('Original data')
test_data = np.load("../../data/imagenet100/test_image.npy", allow_pickle=True)
test_label = np.load("../../data/imagenet100/test_label.npy", allow_pickle=True)
test_dataset = NumpyDataset(test_data,test_label, test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
test(model, test_loader, nn.CrossEntropyLoss(), device)

test_data = np.load("../../data/imagenet100/test_image.npy", allow_pickle=True)
test_label = np.load("../../data/imagenet100/test_label.npy", allow_pickle=True)
backdoor_test_dataset = NumpyDataset(test_data,test_label, test_transform)
n_indexes = np.where(np.array(backdoor_test_dataset.targets) != plabel)[0]
backdoor_test_dataset.targets = np.array(backdoor_test_dataset.targets)[n_indexes]
backdoor_test_dataset.data = backdoor_test_dataset.data[n_indexes]
for i in range(len(backdoor_test_dataset.data)):
    backdoor_test_dataset.data[i, -4:, -4:] = 255
    backdoor_test_dataset.targets[i] = plabel
print('Backdoored data without plabel')
backdoor_test_loader = torch.utils.data.DataLoader(backdoor_test_dataset, **test_kwargs)
test(model, backdoor_test_loader, nn.CrossEntropyLoss(), device)
