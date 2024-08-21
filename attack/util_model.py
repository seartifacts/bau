import sys
sys.path.append("../")

import torch
import torch.nn.functional as F
import numpy as np
import copy
from attack.model import MNISTNet, LeNet5, BadNet
from attack.util_file import save_model, load_model
from torch.nn.utils import clip_grad_norm_
import gc

def train(model, dataloader, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(x)
        #     print('loss: {:.4f} [{}/{}]'.format(loss, current, size))

# def train_with_delta(model, train_dataset, loss_fn, optimizer, device, delta_batches, indexes, mitigation_indexes_sample, train_kwargs, dataset=None):
def train_with_delta(model, train_loader, loss_fn, optimizer, device, delta_batches, indexes, mitigation_indexes_sample, train_kwargs):
    # train_delta_dataset = copy.deepcopy(train_dataset)
    # if dataset == 'cifar10' or dataset == 'gtsrb' or dataset == 'imagenet':
    #     train_delta_dataset.data = train_delta_dataset.data[indexes]
    # else:
    #     train_delta_dataset.data = torch.tensor(train_delta_dataset.data[indexes], dtype=torch.uint8)
    # train_delta_dataset.targets = np.array(train_delta_dataset.targets)[indexes]
    # train_loader = torch.utils.data.DataLoader(train_delta_dataset, **train_kwargs)
    batch_size = train_kwargs['batch_size']

    mitigation_train_index = [np.array(indexes).tolist().index(item) for item in mitigation_indexes_sample if item in indexes]

    model.train()

    sensitive_batch_count = 0
    for batch, (x, y) in enumerate(train_loader):
        train_index = set(range(batch*batch_size, batch*batch_size+min(len(x),batch_size)))
        sensitive_samples = train_index.intersection(set(mitigation_train_index))
        sensitive_size = len(sensitive_samples)

        if sensitive_size > 0:
            before = {}
            for param_tensor in model.state_dict():
                if "weight" in param_tensor or "bias" in param_tensor:
                    before[param_tensor] = model.state_dict()[param_tensor].clone().cpu()

        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if sensitive_size > 0:
            sensitive_batch_count += 1
            after = {}
            for param_tensor in model.state_dict():
                if "weight" in param_tensor or "bias" in param_tensor:
                    after[param_tensor] = model.state_dict()[param_tensor].clone().cpu()
            if sensitive_batch_count > len(delta_batches):
                delta = {}
                for param_tensor in model.state_dict():
                    if "weight" in param_tensor or "bias" in param_tensor:
                        delta[param_tensor] = 0
                for key in before:
                    delta[key] = delta[key] + after[key] - before[key]
                delta_batches.append(delta)
            else:
                delta = delta_batches[sensitive_batch_count - 1]
                for key in before:
                    delta[key] = delta[key] + after[key] - before[key]
                delta_batches[sensitive_batch_count - 1] = delta
        before = None
        after = None
        delta = None
        gc.collect()
    # train_delta_dataset = None
    # gc.collect()
    return delta_batches

def test(model, dataloader, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    loss, correct = 0.0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.int).sum().item()

    loss /= num_batches
    correct /= size
    print('Test Result: Accuracy @ {:.2f}%, Avg loss @ {:.4f}\n'.format(100 * correct, loss))

def test_print(model, dataloader, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    loss, correct = 0.0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.int).sum().item()
            for i in range(len(x)):
                print(F.softmax(pred[i])[pred.argmax(1)[i]], pred.argmax(1)[i])

    loss /= num_batches
    correct /= size
    print('Test Result: Accuracy @ {:.2f}%, Avg loss @ {:.4f}\n'.format(100 * correct, loss))

def split(model, lIndex):
    sModel = [item[0] for item in model._modules.items()]
    model1 = sModel[:lIndex - 1]
    model2 = sModel[lIndex:]
    return (model1, model2)

def split_model_prediction(x, model, lIndex, mIndex):
    model_names = [item[0] for item in model._modules.items()]
    preLayer = None
    index = 0
    output = x

    for item in model._modules.items():
        if mIndex == 0 and index < lIndex:
            if "conv" in item[0]:
                output = F.relu(item[1](output))
            elif "fc" in item[0]:
                if "fc" not in preLayer:
                    output = torch.flatten(output, 1)
                output = F.relu(item[1](output))
            else:
                output = item[1](output)
        elif mIndex == 1 and index >= lIndex:
            if "conv" in item[0]:
                output = F.relu(item[1](output))
            elif "fc" in item[0]:
                if "fc" not in preLayer:
                    output = torch.flatten(output, 1)

                if item[0] != model_names[-1]:
                    output = F.relu(item[1](output))
                else:
                    output = F.softmax(item[1](output), dim=1)
            else:
                output = item[1](output)
        preLayer = item[0]
        index += 1
    return output

if __name__ == '__main__':
    model = LeNet5("mnist").to('cpu')
    model = load_model(model, '../model/mnist_lenet5.pt','cpu')
    split(model,1)


