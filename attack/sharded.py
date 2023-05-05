import numpy as np
from hashlib import sha256
import importlib
import json
import torch
from torchvision import datasets, transforms
from attack.gtsrb_data import GTSRBLoader


def sizeOfShard(args, shard):
    '''
    Returns the size (in number of points) of the shard before any unlearning request.
    '''
    shards = np.load(args.path + 'SNO_{}/splitfile.npy'.format(args.shards), allow_pickle=True)

    return shards[shard].shape[0]


def realSizeOfShard(args, shard):
    '''
    Returns the actual size of the shard (including unlearning requests).
    '''
    shards = np.load(args.path + 'SNO_{}/splitfile.npy'.format(args.shards), allow_pickle=True)
    requests = np.load(args.path + 'SNO_{}/requestfile-{}.npy'.format(args.shards, args.requests), allow_pickle=True)

    return shards[shard].shape[0] - requests[shard].shape[0]


def getShardHash(args, shard, until=None):
    '''
    Returns a hash of the indices of the points in the shard lower than until
    that are not in the requests (separated by :).
    '''
    shards = np.load(args.path + 'SNO_{}/splitfile.npy'.format(args.shards), allow_pickle=True)
    requests = np.load(args.path + 'SNO_{}/requestfile-{}.npy'.format(args.shards, args.requests), allow_pickle=True)

    if until == None:
        until = shards[shard].shape[0]
    indices = np.setdiff1d(shards[shard][:until], requests[shard])
    string_of_indices = ':'.join(indices.astype(str))
    return sha256(string_of_indices.encode()).hexdigest()

def fetchShardBatch(args, shard, train_data, train_label, train_kwargs, sl, slice_size):
    '''
    Generator returning batches of points in the shard that are not in the requests
    with specified batch_size from the specified dataset
    optionnally located between offset and until (slicing).
    '''
    shards = np.load(args.path + 'SNO_{}/splitfile.npy'.format(args.shards), allow_pickle=True)
    requests = np.load(args.path + 'SNO_{}/requestfile-{}.npy'.format(args.shards, args.requests), allow_pickle=True)

    if args.dataset == 'mnist':
        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST('../../data', train=True, download=True, transform=transform)
    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.CIFAR10('../../data', train=True, download=True, transform=train_transform)
    elif args.dataset == 'gtsrb':
        trainD = np.load("../../data/gtsrb/train_image.npy")
        trainL = np.load("../../data/gtsrb/train_label.npy")
        train_dataset = GTSRBLoader(np.transpose(trainD, [0, 3, 1, 2]), trainL)
    elif args.dataset == 'fmnist':
        transform = transforms.ToTensor()
        train_dataset = datasets.FashionMNIST('../../data', train=True, download=True, transform=transform)

    # begin = sl * slice_size
    begin = 0
    end = min((sl + 1) * slice_size, len(shards[shard]))
    indices = np.setdiff1d(shards[shard][begin:end], requests[shard])
    np.random.shuffle(indices)
    train_dataset.data = train_data[indices]
    train_dataset.targets = train_label[indices]

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    return train_loader
