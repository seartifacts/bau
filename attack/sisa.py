import sys
sys.path.append("../")
sys.path.append("../../")
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import one_hot
from attack.sharded import sizeOfShard, getShardHash, fetchShardBatch
import os
from glob import glob
import json
from torch import nn
from attack.util_model import train
from attack.model import MNISTNet, LeNet5, BadNet, FMNISTNet,VGG11, GTSRBNet, VGG16, ResNet50

def sisa_train(args, train_data, train_label, train_kwargs):
    for sn in range(args.shards):
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

        epochs = args.epochs
        shard_size = sizeOfShard(args, sn)
        slice_size = shard_size // args.slices
        avg_epochs_per_slice = (
            2 * args.slices / (args.slices + 1) * epochs / args.slices
        )
        loaded = False

        for sl in range(args.slices):
            print("start training model-slice: {}-{}".format(sn,sl))
            # Get slice hash using sharded lib.
            slice_hash = getShardHash(
                args, sn, until=(sl + 1) * slice_size
            )

            # If checkpoints exists, skip the slice.
            if not os.path.exists(
                args.path + "SNO_{}/cache/{}.pt".format(args.shards, slice_hash)
            ):
                # Initialize state.
                start_epoch = 0
                slice_epochs = int((sl + 1) * avg_epochs_per_slice) - int(
                    sl * avg_epochs_per_slice
                )

                # If weights are already in memory (from previous slice), skip loading.
                if not loaded:
                    # Look for a recovery checkpoint for the slice.
                    recovery_list = glob(
                        args.path + "SNO_{}/cache/{}_*.pt".format(args.shards, slice_hash)
                    )
                    if len(recovery_list) > 0:
                        print(
                            "Recovery mode for shard {} on slice {}".format(args.shards, sl)
                        )

                        # Load weights.
                        model.load_state_dict(torch.load(recovery_list[0]))
                        start_epoch = int(
                            recovery_list[0].split("/")[-1].split(".")[0].split("_")[1]
                        )

                    # If there is no recovery checkpoint and this slice is not the first, load previous slice.
                    elif sl > 0:
                        previous_slice_hash = getShardHash(
                            args, sn, until=sl * slice_size
                        )

                        # Load weights.
                        model.load_state_dict(
                            torch.load(
                                args.path + "SNO_{}/cache/{}.pt".format(
                                    args.shards, previous_slice_hash
                                )
                            )
                        )

                    # Mark model as loaded for next slices.
                    loaded = True

                # If this is the first slice, no need to load anything.
                elif sl == 0:
                    loaded = True

                # update train data
                dataloader = fetchShardBatch(args, sn, train_data, train_label, train_kwargs, sl, slice_size)
                # Actual training.
                for epoch in range(start_epoch, slice_epochs):
                    if args.dataset == 'cifar10':
                        adjust_learning_rate(optimizer, epoch)
                    elif args.dataset == 'imagnet':
                        adjust_learning_rate(optimizer, epoch)
                    train(model, dataloader, nn.CrossEntropyLoss(), optimizer, device)

                    # Create a checkpoint every chkpt_interval.
                    if (
                        args.chkpt_interval != -1
                        and epoch % args.chkpt_interval == args.chkpt_interval - 1
                    ):
                        # Save weights
                        torch.save(
                            model.state_dict(),
                            args.path + "SNO_{}/cache/{}_{}.pt".format(
                                args.shards, slice_hash, epoch
                            ),
                        )

                        # Remove previous checkpoint.
                        if os.path.exists(
                            args.path + "SNO_{}/cache/{}_{}.pt".format(
                                args.shards, slice_hash, epoch - args.chkpt_interval
                            )
                        ):
                            os.remove(
                                args.path + "SNO_{}/cache/{}_{}.pt".format(
                                    args.shards, slice_hash, epoch - args.chkpt_interval
                                )
                            )

                # When training is complete, save slice.
                torch.save(
                    model.state_dict(),
                    args.path + "SNO_{}/cache/{}.pt".format(args.shards, slice_hash),
                )

                # Remove previous checkpoint.
                if os.path.exists(
                    args.path + "SNO_{}/cache/{}_{}.pt".format(
                        args.shards, slice_hash, slice_epochs - args.chkpt_interval
                    )
                ):
                    os.remove(
                        args.path + "SNO_{}/cache/{}_{}.pt".format(
                            args.shards, slice_hash, slice_epochs - args.chkpt_interval
                        )
                    )

                # If this is the last slice, create a symlink attached to it.
                if sl == args.slices - 1:
                    os.symlink(
                        "{}.pt".format(slice_hash),
                        args.path + "SNO_{}/cache/shard-{}-{}.pt".format(
                            args.shards, sn, args.requests
                        ),
                    )
            elif sl == args.slices - 1:
                os.symlink(
                    "{}.pt".format(slice_hash),
                    args.path + "SNO_{}/cache/shard-{}-{}.pt".format(
                        args.shards, sn, args.requests
                    ),
                )


def sisa_test(args, dataloader, name=None):
    for sn in range(args.shards):
        if int(args.gpu) == -1:
            device = 'cpu'
        else:
            device = 'cuda:' + str(args.gpu)
        if args.dataset == 'mnist':
            model = LeNet5("mnist").to(device)
        elif args.dataset == 'cifar10':
            model = VGG11().to(device)
        elif args.dataset == 'imagenet':
            model = VGG16().to(device)
            # model = ResNet50().to(device)
        elif args.dataset == 'gtsrb':
            model = GTSRBNet().to(device)
        elif args.dataset == 'fmnist':
            model = FMNISTNet().to(device)

        # Load model weights from shard checkpoint (last slice).
        model.load_state_dict(
            torch.load(
                args.path + "SNO_{}/cache/shard-{}-{}.pt".format(
                    args.shards, sn, args.requests
                )
            )
        )

        # Compute predictions batch per batch.
        if args.dataset == 'gtsrb':
            nb_classes = 43
        elif args.dataset == 'imagenet':
            nb_classes = 100
        else:
            nb_classes = 10
        outputs = np.empty((0, nb_classes))

        # dataloader = fetchTestBatch(test_dataset, test_kwargs)

        model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                if args.output_type == "softmax":
                    # Actual batch prediction.
                    logits = model(x)
                    predictions = F.softmax(logits, dim=1).to("cpu")  # Send back to cpu.

                    # Convert back to numpy and concatenate with previous batches.
                    outputs = np.concatenate((outputs, predictions.numpy()))
                else:
                    # Actual batch prediction.
                    logits = model(x)
                    predictions = torch.argmax(logits, dim=1)  # pylint: disable=no-member

                    # Convert to one hot, send back to cpu, convert back to numpy and concatenate with previous batches.
                    out = one_hot(predictions, nb_classes).to("cpu")
                    outputs = np.concatenate((outputs, out.numpy()))

            # Save outputs in numpy format.
            outputs = np.array(outputs)
            if name == None:
                np.save(
                    args.path + "SNO_{}/outputs/shard-{}-{}.npy".format(
                        args.shards, sn, args.requests
                    ),
                    outputs,
                )
            else:
                np.save(
                    args.path + "SNO_{}/outputs/shard-{}-{}_{}.npy".format(
                        args.shards, sn, args.requests, name
                    ),
                    outputs,
                )
