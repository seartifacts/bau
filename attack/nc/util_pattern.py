import sys
import torch
from torch import nn

import tqdm
import numpy as np

def find_pattern(model, train_loader, params, dataset=None):
    shape = params["shape"]
    if len(shape) == 2:
        [width, height] = shape
        chanel = 1
    else:
        if dataset == 'gtsrb':
            [chanel, width, height] = shape
        else:
            [width, height, chanel] = shape

    device = params["device"]
    pattern = torch.rand((chanel, width, height), requires_grad=True)
    pattern = pattern.to(device).detach().requires_grad_(True)
    mask = torch.rand((width, height), requires_grad=True)
    mask = mask.to(device).detach().requires_grad_(True)

    cifar10_mean = torch.tensor(np.array([[[0.485, 0.456, 0.406]] * 32] * 32).transpose(2, 0, 1)).type(torch.float).to(device)
    cifar10_std = torch.tensor(np.array([[[0.229, 0.224, 0.225]] * 32] * 32).transpose(2, 0, 1)).type(torch.float).to(device)
    # gtsrb_mean = torch.tensor([0.3337, 0.3064, 0.3171])
    # gtsrb_std = torch.tensor([0.2672, 0.2564, 0.2629])

    miu = params["miu"]
    epochs = params["epochs"]
    target_label = params["target_label"]

    min_norm = np.inf
    min_norm_count = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([{"params": pattern}, {"params": mask}], lr=0.005)
    model.eval()
    for epoch in range(epochs):
        norm = 0.0
        for images, _ in tqdm.tqdm(train_loader, desc='Epoch %3d' % (epoch + 1), disable=True):
            optimizer.zero_grad()
            images = images.to(device)
            if dataset == None:
                backdoor_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * pattern
            elif dataset == "cifar10":
                backdoor_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * (pattern - cifar10_mean) / cifar10_std
            elif dataset == "gtsrb":
                backdoor_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * pattern * 255.
            y_pred = model(backdoor_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
            loss = criterion(y_pred, y_target) + miu * torch.sum(torch.abs(mask))
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                torch.clip_(pattern, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))
            # print("norm: {}".format(norm))

            # early stop
            if norm < min_norm:
                min_norm = norm
                min_norm_count = 0
            else:
                min_norm_count += 1

            if min_norm_count > 30:
                break

    return pattern.cpu(), mask.cpu()

def test_with_pattern(model, dataloader, loss_fn, device, pattern, mask, dataset=None):
    cifar10_mean = torch.tensor(np.array([[[0.485, 0.456, 0.406]] * 32] * 32).transpose(2, 0, 1)).type(torch.float).to(device)
    cifar10_std = torch.tensor(np.array([[[0.229, 0.224, 0.225]] * 32] * 32).transpose(2, 0, 1)).type(torch.float).to(device)
    # gtsrb_mean = torch.tensor([0.3337, 0.3064, 0.3171])
    # gtsrb_std = torch.tensor([0.2672, 0.2564, 0.2629])

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    loss, correct = 0.0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if dataset == None:
                backdoor_images = (1 - torch.unsqueeze(mask, dim=0)) * x + torch.unsqueeze(mask, dim=0) * pattern
            elif dataset == 'cifar10':
                backdoor_images = (1 - torch.unsqueeze(mask, dim=0)) * x + torch.unsqueeze(mask, dim=0) * (
                            pattern - cifar10_mean) / cifar10_std
            elif dataset == 'gtsrb':
                backdoor_images = (1 - torch.unsqueeze(mask, dim=0)) * x + torch.unsqueeze(mask, dim=0) * pattern * 255.
            pred = model(backdoor_images)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.int).sum().item()

    loss /= num_batches
    correct /= size
    print('Test Result: Accuracy @ {:.2f}%, Avg loss @ {:.4f}\n'.format(100 * correct, loss))


