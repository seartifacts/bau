import sys
sys.path.append("../")

import torch
import torch.nn.functional as F
import numpy as np
from attack.model import MNISTNet, LeNet5, BadNet
from attack.util_file import save_model, load_model

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

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print('loss: {:.4f} [{}/{}]'.format(loss, current, size))


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


