import sys
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
from .model_zoo import _ResNet18
from .GTSRB import GTSRB
from .ImageNette import Imagenette

def create_data(config):
    if config["DATASET"] == "cifar10":
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testset, _ = data_split(testset, 'evaluation', ratio=config['SPLIT_RATIO'])
    elif config["DATASET"] == "gtsrb":
        transform_train = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
        ])
        trainset = GTSRB(root='./data', split='train', download=True, transform=transform_train)
        testset = GTSRB(root='./data', split='test', download=True, transform=transform_test)
    elif config["DATASET"] == "imagenette":
        transform_train = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

        trainset = Imagenette(root='./data/imagenette2', train=True, transform=transform_train)
        testset = Imagenette(root='./data/imagenette2', train=False, transform=transform_test)
        testset, _ = data_split(testset, 'evaluation', ratio=config['SPLIT_RATIO'])
    else:
        sys.exit("Dataset is unrecognized!")

    # Create the backdoor patterns
    
    backdoor_pattern = create_pattern(im_size=trainset.__getitem__(0)[0].size(), config=config)

    # Save a visualization of the backdoor pattern
    
    pattern_save(pattern=backdoor_pattern, config=config, path='attacks/{}/{}/{}/{}'.format(config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"]))

    # Create backdoor training images (for poisoning the training set) and backdoor test images (for attack effectiveness evaluation)
    train_images_attacks = None
    train_labels_attacks = None
    test_images_attacks = None
    test_labels_attacks = None
    ind_train = None
    pairs = None
    if config['SETTING'] == 'A2O':
        source = np.arange(stop=config["NUM_CLASS"])
        source = np.delete(source, config['TC'])
        ind_train = [i for i, label in enumerate(trainset.targets) if label in source]
        ind_train = np.random.choice(ind_train, config['NUM_POISONING_SAMPLE'], False)
        pairs = [[i, config['TC']]for i in source]
    elif config['SETTING'] == 'A2A':
        if config["TC"] % config["NUM_CLASS"] == 0:
            print('A2A can not attack the same class!')
            sys.exit(0)
        source = np.arange(stop=config["NUM_CLASS"])
        ind_train = []
        for c in range(config["NUM_CLASS"]):
            ind = [i for i, label in enumerate(trainset.targets) if label == c]
            ind = np.random.choice(ind, int(config['NUM_POISONING_SAMPLE'] / config["NUM_CLASS"]), False)
            ind_train.extend(ind)
        ind_train = np.asarray(ind_train)
        pairs = [[i, (i + config['TC']) % config['NUM_CLASS']]for i in source]
    elif config["SETTING"] == "rand":
        source = np.arange(stop=config["NUM_CLASS"])
        targets = source
        while np.sum(source == targets) > 0:
            targets = np.random.permutation(source) 
        ind_train = []
        for c in range(config["NUM_CLASS"]):
            ind = [i for i, label in enumerate(trainset.targets) if label == c]
            ind = np.random.choice(ind, int(config['NUM_POISONING_SAMPLE'] / config["NUM_CLASS"]), False)
            ind_train.extend(ind)
        ind_train = np.asarray(ind_train)
        pairs = [[i, targets[i]]for i in source]
    elif config["SETTING"] == "x2x":
        source = np.random.choice(np.arange(stop=config["NUM_CLASS"]), config["X2X_NUM"], False)
        targets = np.arange(stop=config["NUM_CLASS"])
        while np.sum(source == targets[source]) > 0:
            targets = np.random.permutation(targets)
        ind_train = []
        for c in source:
            ind = [i for i, label in enumerate(trainset.targets) if label == c]
            ind = np.random.choice(ind, int(config['NUM_POISONING_SAMPLE'] / config["X2X_NUM"]), False)
            ind_train.extend(ind)
        ind_train = np.asarray(ind_train)
        pairs = [[i, targets[i]]for i in source]
    print("Pairs are: ")
    print(pairs)

    if config['PATTERN_TYPE'] == "CLA":
        ind_train = [i for i, label in enumerate(trainset.targets) if label == config["TC"]]
        ind_train = np.random.choice(ind_train, config['NUM_POISONING_SAMPLE'], False)
        images = trainset.data[ind_train]
        labels = (torch.ones(len(images)) * config["TC"]).long()
        images_attacked = add_perturbation(images, labels, config)
        trainset.data[ind_train] = images_attacked

    ind_test = [i for i, label in enumerate(testset.targets) if label in source]
    for i in ind_train:
        img, label = trainset.__getitem__(i)
        if train_images_attacks is not None:
            train_images_attacks = torch.cat([train_images_attacks, backdoor_embedding(image=img, pattern=backdoor_pattern, config=config).unsqueeze(0)], dim=0)
            if config['SETTING'] == 'A2O':
                train_labels_attacks = torch.cat([train_labels_attacks, torch.tensor([config['TC']], dtype=torch.long)], dim=0)
            elif config['SETTING'] == 'A2A':
                train_labels_attacks = torch.cat([train_labels_attacks, torch.tensor([(label+config['TC'])%config['NUM_CLASS']], dtype=torch.long)], dim=0)
            elif config['SETTING'] == 'rand' or config['SETTING'] == 'x2x':
                train_labels_attacks = torch.cat([train_labels_attacks, torch.tensor([targets[label]], dtype=torch.long)], dim=0)
        else:
            train_images_attacks = backdoor_embedding(image=img, pattern=backdoor_pattern, config=config).unsqueeze(0)
            if config['SETTING'] == 'A2O':
                train_labels_attacks = torch.tensor([config['TC']], dtype=torch.long)
            elif config['SETTING'] == 'A2A':
                train_labels_attacks = torch.tensor([(label+config['TC'])%config['NUM_CLASS']], dtype=torch.long)
            elif config['SETTING'] == 'rand' or config['SETTING'] == 'x2x':
                train_labels_attacks = torch.tensor([targets[label]], dtype=torch.long)
    for i in ind_test:
        img, label = testset.__getitem__(i)
        if test_images_attacks is not None:
            test_images_attacks = torch.cat([test_images_attacks, backdoor_embedding(image=img, pattern=backdoor_pattern, config=config).unsqueeze(0)], dim=0)
            if config['SETTING'] == 'A2O':
                test_labels_attacks = torch.cat([test_labels_attacks, torch.tensor([config['TC']], dtype=torch.long)], dim=0)
            elif config['SETTING'] == 'A2A':
                test_labels_attacks = torch.cat([test_labels_attacks, torch.tensor([(label+config['TC'])%config['NUM_CLASS']], dtype=torch.long)], dim=0)
            elif config['SETTING'] == 'rand' or config['SETTING'] == 'x2x':
                test_labels_attacks = torch.cat([test_labels_attacks, torch.tensor([targets[label]], dtype=torch.long)], dim=0)
        else:
            test_images_attacks = backdoor_embedding(image=img, pattern=backdoor_pattern, config=config).unsqueeze(0)
            if config['SETTING'] == 'A2O':
                test_labels_attacks = torch.tensor([config['TC']], dtype=torch.long)
            elif config['SETTING'] == 'A2A':
                test_labels_attacks = torch.tensor([(label+config['TC'])%config['NUM_CLASS']], dtype=torch.long)
            elif config['SETTING'] == 'rand' or config['SETTING'] == 'x2x':
                test_labels_attacks = torch.tensor([targets[label]], dtype=torch.long)
    # Save created backdoor image
    torch.save(ind_train, './attacks/{}/{}/{}/{}/ind_train'.format(config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"]))
    torch.save(pairs, './attacks/{}/{}/{}/{}/pairs'.format(config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"]))

    # Save example backdoor images for visualization
    image_clean = trainset.__getitem__(ind_train[0])[0]
    image_clean = image_clean.numpy()
    image_clean = np.transpose(image_clean, [1, 2, 0])
    plt.imshow(image_clean)
    plt.savefig('./attacks/{}/{}/{}/{}/image_clean.png'.format(config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"]))
    image_poisoned = train_images_attacks[0]
    image_poisoned = image_poisoned.numpy()
    image_poisoned = np.transpose(image_poisoned, [1, 2, 0])
    plt.imshow(image_poisoned)
    plt.savefig('./attacks/{}/{}/{}/{}/image_poisoned.png'.format(config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"]))
    return train_images_attacks, train_labels_attacks, test_images_attacks, test_labels_attacks, ind_train

def create_poison_data(config, dataset, pattern, pairs):
    if config['SETTING'] == 'A2O':
        source = np.arange(stop=config["NUM_CLASS"])
        source = np.delete(source, pairs[0][1])
    elif config['SETTING'] == 'A2A':
        if config["TC"] % config["NUM_CLASS"] == 0:
            print('A2A can not attack the same class!')
            sys.exit(0)
        source = np.arange(stop=config["NUM_CLASS"])
    elif config["SETTING"] == "rand":
        source = np.arange(stop=config["NUM_CLASS"])

    ind_test = [i for i, label in enumerate(dataset.targets) if label in source]
    targets = [[i, i] for i in range(config["NUM_CLASS"])]
    for pair in pairs:
        targets[pair[0]][1] = pair[1] 
    test_images_attacks = None 
    for i in ind_test:
        img, label = dataset.__getitem__(i)
        if test_images_attacks is not None:
            test_images_attacks = torch.cat([test_images_attacks, backdoor_embedding(image=img, pattern=pattern, config=config).unsqueeze(0)], dim=0)
            test_labels_attacks = torch.cat([test_labels_attacks, torch.tensor([targets[label][1]], dtype=torch.long)], dim=0)
            test_labels_origin = torch.cat([test_labels_origin, torch.tensor([label], dtype=torch.long)], dim=0)
        else:
            test_images_attacks = backdoor_embedding(image=img, pattern=pattern, config=config).unsqueeze(0)
            test_labels_attacks = torch.tensor([targets[label][1]], dtype=torch.long)
            test_labels_origin = torch.tensor([label], dtype=torch.long)
    return test_images_attacks, test_labels_attacks, test_labels_origin

def create_pattern(im_size, config):

    if config['PATTERN_TYPE'] == "perturbation":
        pert_size = config['PERTURBATION_SIZE']
        pert_shape = config['PERTURBATION_SHAPE']
        if pert_shape == 'chessboard':
            pert = torch.zeros(im_size)
            for i in range(im_size[1]):
                for j in range(im_size[2]):
                    if (i + j) % 2 == 0:
                        pert[:, i, j] = torch.ones(im_size[0])
            pert *= pert_size
        elif pert_shape == 'static':
            pert = torch.zeros(im_size)
            for i in range(im_size[1]):
                for j in range(im_size[2]):
                    if (i % 2 == 0) and (j % 2 == 0):
                        pert[:, i, j] = torch.ones(im_size[0])
            pert *= pert_size
        elif pert_shape == 'lshape':
            pert = torch.zeros(im_size)
            cx = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            cy = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            for c in range(im_size[0]):
                pert[c, cx, cy + 1] = pert_size
                pert[c, cx - 1, cy] = pert_size
                pert[c, cx - 2, cy] = pert_size
                pert[c, cx, cy] = pert_size
        elif pert_shape == 'cross':
            pert = torch.zeros(im_size)
            cx = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            cy = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            for c in range(im_size[0]):
                pert[c, cx, cy - 1] = pert_size
                pert[c, cx, cy + 1] = pert_size
                pert[c, cx - 1, cy] = pert_size
                pert[c, cx + 1, cy] = pert_size
                pert[c, cx, cy] = pert_size
        elif pert_shape == 'X':
            pert = torch.zeros(im_size)
            cx = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            cy = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            ch = torch.randint(low=0, high=im_size[0], size=(1,))
            pert[ch, cx - 1, cy - 1] = pert_size
            pert[ch, cx - 1, cy + 1] = pert_size
            pert[ch, cx + 1, cy - 1] = pert_size
            pert[ch, cx + 1, cy + 1] = pert_size
            pert[ch, cx, cy] = pert_size
        elif pert_shape == 'pixel':
            pert = torch.zeros(im_size)
            cx = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            cy = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            ch = torch.randint(low=0, high=im_size[0], size=(1,))
            sgn = torch.randint(low=0, high=2, size=(1,)) * 2 - 1
            pert[ch, cx, cy] += sgn * pert_size * (1 + 0.2 * random.random())
        elif pert_shape == 'square':
            pert = torch.zeros(im_size)
            cx = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            cy = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            for c in range(im_size[0]):
                pert[c, cx, cy] = pert_size
                pert[c, cx, cy + 1] = pert_size
                pert[c, cx + 1, cy] = pert_size
                pert[c, cx + 1, cy + 1] = pert_size
        elif pert_shape == 'bigX':
            pert = torch.zeros(im_size)
            for i in range(im_size[1]):
                for c in range(im_size[0]):
                    pert[c, i, i] = pert_size
                    pert[c, i, im_size[1] - i - 1] = pert_size
        else:
            sys.exit("Perturbation shape is unrecognized!")
        return pert
    elif config['PATTERN_TYPE'] == "patch" or config['PATTERN_TYPE'] == "CLA":
        mask_size = config['MASK_SIZE']
        margin = config['MARGIN']
        patch_type = config['PATCH_TYPE']
        if margin * 2 + mask_size >= im_size[1] or margin * 2 + mask_size >= im_size[2]:
            sys.exit("Decrease margin or mask size!")
        # Pick a random location
        x_candidate = torch.from_numpy(np.concatenate([np.arange(0, margin),
                                                       np.arange(int(im_size[1] - margin - mask_size + 1),
                                                                 int(im_size[1] - mask_size + 1))]))
        y_candidate = torch.from_numpy(np.concatenate([np.arange(0, margin),
                                                       np.arange(int(im_size[2] - margin - mask_size + 1),
                                                                 int(im_size[2] - mask_size + 1))]))
        x = x_candidate[torch.randperm(len(x_candidate))[0]].item()
        y = y_candidate[torch.randperm(len(y_candidate))[0]].item()
        # Create mask and pattern
        mask = torch.zeros(im_size)
        mask[:, x:x + mask_size, y:y + mask_size] = 1
        if patch_type == 'noise':
            patch = torch.randint(0, 255, size=(im_size[0], mask_size, mask_size)) / 255
        elif patch_type == 'uniform':
            color = torch.randint(50, 200, size=(im_size[0], 1, 1)) / 255
            patch = torch.ones((im_size[0], mask_size, mask_size)) * color.repeat(1, mask_size, mask_size)
        elif patch_type == "black_white":
            patch = torch.ones((im_size[0], mask_size, mask_size))
            for i in range(mask_size):
                for j in range(mask_size):
                    if (j % 2 ==0 and i % 2 ==0) or (j % 2 ==1 and i % 2 ==1):
                        patch[:, i, j] = 0
        pattern = torch.zeros(im_size)
        pattern[:, x:x + mask_size, y:y + mask_size] = patch
        pattern = (pattern, mask)
        return pattern
    else:
        sys.exit("Pattern type is unrecognized!")

    pass

def add_perturbation(images, labels, config):
    import torchattacks
    device = config["DEVICE"]
    ckpt_path = config["CLEAN_PATH"]
    ckpt = torch.load((ckpt_path))
    model = _ResNet18(num_classes=10).to(device)
    model.load_state_dict(ckpt)
    to_tensor = transforms.ToTensor()
    image_tensor = []
    for idx in range(len(images)):
        image_tensor.append(to_tensor(images[idx]))
    images = torch.stack(image_tensor)
    atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=7)
    adv_images = []
    model.eval()
    for inputs, targets in zip(images.chunk(int(len(images) / 128)),
                        labels.chunk(int(len(labels) / 128))):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_images.append(atk(inputs, targets))
    images = torch.concat(adv_images)
    images = (images.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    return images

def pattern_save(pattern, config, path):
    torch.save(pattern, os.path.join(path, 'backdoor_pattern.pt'))
    if config['PATTERN_TYPE'] == "perturbation":
        pattern = pattern.numpy()
        pattern = np.transpose(pattern, [1, 2, 0])
        plt.imshow(pattern)
        plt.savefig(os.path.join(path, 'backdoor_pattern.png'))
    elif config['PATTERN_TYPE'] == "patch" or config['PATTERN_TYPE'] == "CLA":
        pattern = pattern[0].numpy()
        pattern = np.transpose(pattern, [1, 2, 0])
        plt.imshow(pattern)
        plt.savefig(os.path.join(path, 'backdoor_pattern.png'))
    else:
        sys.exit("Pattern type is unrecognized!")

    pass


def backdoor_embedding(image, pattern, config):

    if config['PATTERN_TYPE'] == "perturbation":
        image += pattern
        image *= 255
        image = image.round()
        image /= 255
        image = image.clamp(0, 1)
    elif config['PATTERN_TYPE'] == "patch" or config['PATTERN_TYPE'] == "CLA":
        image = image * (1 - pattern[1]) + pattern[0] * pattern[1]
    else:
        sys.exit("Pattern type is unrecognized!")

    return image


def poison(trainset, images, labels, ind, delete=True):

    image_dtype = trainset.data.dtype
    images = np.rint(np.transpose(images.numpy() * 255, [0, 2, 3, 1])).astype(image_dtype)
    trainset.data = np.concatenate((trainset.data, images))
    trainset.targets = np.concatenate((trainset.targets, labels))
    if delete is True:
        trainset.data = np.delete(trainset.data, ind, axis=0)
        trainset.targets = np.delete(trainset.targets, ind, axis=0)

    return trainset


def data_split(dataset, type, ratio):

    ind_keep = []
    num_classes = int(max(dataset.targets) + 1)
    for c in range(num_classes):
        ind = [i for i, label in enumerate(dataset.targets) if label == c]
        split = int(len(ind) * ratio)
        if type == 'evaluation':
            ind = ind[:split]
        elif type == 'defense':
            ind = ind[split:]
        else:
            sys.exit("Wrong training type!")
        ind_keep = ind_keep + ind
    try:
        dataset.data = dataset.data[ind_keep]
    except:
        dataset.data.samples = [dataset.data.samples[i] for i in ind_keep]
    dataset.targets = [dataset.targets[i] for i in ind_keep]

    return dataset, ind_keep

def data_remove(dataset, ind):
    ind_keep = np.ones(len(dataset), dtype=bool)
    ind_keep[ind] = False
    ind_keep = np.arange(len(dataset))[ind_keep]
    dataset.data.samples = [dataset.data.samples[i] for i in ind_keep]
    dataset.targets = [dataset.targets[i] for i in ind_keep]
    return dataset


def pert_est_class_pair(source, target, model, images, labels, pi=0.9, lr=1e-4, NSTEP=1000, init=None, verbose=False, device='cuda'):
    '''
    :param source: souce class
    :param target: target class
    :param model: model to be insected
    :param images: batch of images for perturbation estimation
    :param labels: the target labels
    :param pi: the target misclassification fraction (default is 0.9)
    :param lr: learning rate (default is 1e-4)
    :param NSTEP: number of steps to terminate (default is 100)
    :param verbose: set True to plot details
    :return:
    '''

    if verbose:
        print("Perturbation estimation for class pair (s, t)".format(source, target))

    # Initialize perturbation
    if init is not None:
        pert = init
    else:
        pert = torch.zeros_like(images[0]).to(device)
    pert.requires_grad = True

    for iter_idx in range(NSTEP):

        # Optimizer: SGD
        optimizer = torch.optim.SGD([pert], lr=lr, momentum=0)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Get the loss
        images_perturbed = torch.clamp(images + pert, min=0, max=1)
        _, _, outputs = model(images_perturbed)
        loss = criterion(outputs, labels)

        # Update perturbation
        model.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()

        # Compute misclassification fraction rho
        misclassification = 0
        with torch.no_grad():
            images_perturbed = torch.clamp(images + pert, min=0, max=1)
            _, _, outputs = model(images_perturbed)
            _, predicted = outputs.max(1)
            misclassification += predicted.eq(labels).sum().item()
            rho = misclassification / len(labels)

        if verbose:
            print("current misclassification: {}; perturbation norm: {}".format(rho, torch.norm(pert).detach().cpu().numpy()))

        # Stopping criteria
        if rho >= pi or torch.norm(pert) > 20:
            break

    return pert.detach().cpu(), rho


def pm_est_class_pair(images, model, target, labels, pi=0.9, device='cuda', batch_size=32, LR1=1e-1, LR2=1e-1,verbose=False, gt=None):
    if images[0].shape[1] == 32:
        # # Est parameter for Cifar-10 and GTSRB
        COST_INIT = 1e-2  # 1e-3 Initial cost parameter of the L1 norm
        COST_MAX = 10 # 10
        NSTEP1 = 1000  # 3000 Maximum number of steps for reaching PI misclassification without L1-constraint
        NSTEP2 = 1000  # int(1e4) Maximum number of steps for pattern estimation after achieving PI misclassification
        PATIENCE_UP = 5  # 5
        PATIENCE_DOWN = 5  # 5
        PATIENCE_STAGE1 = 10
        PATIENCE_CONVERGENCE = 100
        COST_UP_MULTIPLIER = 1.5  # 1.5
        COST_DOWN_MULTIPLIER = 1.5  # 1.5
        LR1 = 1e-2  # 5e-3 Learning rate for the first stage
        LR2 = 1e-1  
    elif images[0].shape[1] == 224:
        # # Est parameter for ImageNette
        COST_INIT = 1e-3  # 1e-3 Initial cost parameter of the L1 norm 1e-2 old
        COST_MAX = 1e-3 # 10
        NSTEP1 = 1000  # 3000 Maximum number of steps for reaching PI misclassification without L1-constraint
        NSTEP2 = 10000  # int(1e4) Maximum number of steps for pattern estimation after achieving PI misclassification
        PATIENCE_UP = 2  # 5
        PATIENCE_DOWN = 2  # 5
        PATIENCE_STAGE1 = 10
        PATIENCE_CONVERGENCE = 30
        COST_UP_MULTIPLIER = 1.5  # 1.5
        COST_DOWN_MULTIPLIER = 1.5 
        LR2 = 100
        LR1 = 1e-1  # 5e-3 Learning rate for the first stage
        LR2 = 10  # 5e-1 Learning rate for the second stage
    elif images[0].shape[1] == 28:
        # # Est parameter for Cifar-10 and GTSRB
        COST_INIT = 1e-2  # 1e-3 Initial cost parameter of the L1 norm
        COST_MAX = 10 # 10
        NSTEP1 = 1000  # 3000 Maximum number of steps for reaching PI misclassification without L1-constraint
        NSTEP2 = 1000  # int(1e4) Maximum number of steps for pattern estimation after achieving PI misclassification
        PATIENCE_UP = 5  # 5
        PATIENCE_DOWN = 5  # 5
        PATIENCE_STAGE1 = 10
        PATIENCE_CONVERGENCE = 100
        COST_UP_MULTIPLIER = 1.5  # 1.5
        COST_DOWN_MULTIPLIER = 1.5  # 1.5
        LR1 = 1e-2  # 5e-3 Learning rate for the first stage
        LR2 = 1e-1  
    else:
        raise ValueError("Image size not supported")
    criterion = nn.CrossEntropyLoss()
    images = images.to(device)
    labels = labels.to(device)
    # Perform pattern-mask estimation for target class
    im_size = images[0].size()
    pattern_raw = torch.ones(im_size) * random.uniform(-1., 1.)
    mask_raw = torch.zeros((1, im_size[1], im_size[2]))
    noise = torch.normal(0, 1e-1, size=pattern_raw.size())
    pattern_raw, mask_raw = (pattern_raw + noise).to(device), (mask_raw + noise[0, :, :]).to(device)
    
    mask_norm_best = float("inf")
    associated_rho = 0.0

    # First stage, achieve PI-level misclassification
    stopping_count = 0

    for iter_idx in range(NSTEP1):

        # Optimizer
        lr_noisy = LR1 * (1 + np.random.normal(loc=0, scale=0.1))
        optimizer = torch.optim.SGD([pattern_raw, mask_raw], lr=lr_noisy, momentum=0.5)

        # Require gradient
        pattern_raw.requires_grad = True
        mask_raw.requires_grad = True

        # Embed the backdoor pattern
        pattern = (torch.tanh(pattern_raw) + 1) / 2
        mask = (torch.tanh(mask_raw.repeat(im_size[0], 1, 1)) + 1) / 2
        images_with_bd = torch.clamp(images * (1 - mask) + pattern * mask, min=0, max=1)

        # Feed the image with the backdoor into the classifier, and get the loss
        outputs = model(images_with_bd)
        loss = criterion(outputs, labels)

        # Update the pattern and mask (for 1 step)
        model.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Clip pattern_raw and mask_raw to avoid saturation
        pattern_raw, mask_raw = pattern_raw.detach(), mask_raw.detach()
        pattern_raw.clamp(min=-5., max=5.)
        mask_raw.clamp(min=-5., max=5.)

        # Compute the misclassification fraction
        misclassification = 0.0
        total = 0.0
        pattern = (torch.tanh(pattern_raw) + 1) / 2
        mask = (torch.tanh(mask_raw.repeat(im_size[0], 1, 1)) + 1) / 2
        # Get the misclassification count for class s
        with torch.no_grad():
            # Embed the backdoor pattern
            images_with_bd = torch.clamp(images * (1 - mask) + pattern * mask, min=0, max=1)

            outputs = model(images_with_bd)
            _, predicted = outputs.max(1)
            misclassification += predicted.eq(labels).sum().item()
            total += len(labels)
        rho = misclassification / total

        if verbose:
            print('iter {} mis-classification rate {}'.format(iter_idx, rho))

        # Stopping criteria
        if rho >= pi:
            stopping_count += 1
        else:
            stopping_count = 0

        if stopping_count >= PATIENCE_STAGE1:
            break

    mask_best = copy.deepcopy(mask)
    pattern_best = copy.deepcopy(pattern)

    if rho < pi:
        print('PI-misclassification not achieved in phase 1.')

    # Second State, jointly optimize pattern and mask with the L1 constraint
    stopping_count = 0

    # Set the cost manipulation parameters
    cost = COST_INIT  # Initialize the cost of L1 constraint (lambda multiplier)
    cost_up_counter = 0
    cost_down_counter = 0

    for iter_idx in range(NSTEP2):
        # Optimizer
        optimizer = torch.optim.SGD([pattern_raw, mask_raw], lr=LR2, momentum=0.0)

        # Require gradient
        pattern_raw.requires_grad = True
        mask_raw.requires_grad = True

        # Embed the backdoor pattern
        pattern = (torch.tanh(pattern_raw) + 1) / 2
        mask = (torch.tanh(mask_raw.repeat(im_size[0], 1, 1)) + 1) / 2
        images_with_bd = torch.clamp(images * (1 - mask) + pattern * mask, min=0, max=1)

        # Feed the image with the backdoor into the classifier, and get the loss
        outputs = model(images_with_bd)
        loss = criterion(outputs, labels)

        # Add the loss corresponding to the L1 constraint
        loss += cost * torch.sum(torch.abs(mask))

        # Update the pattern & mask (for 1 step)
        model.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Compute the misclassification fraction
        misclassification = 0.0
        total = 0.0
        pattern = (torch.tanh(pattern_raw) + 1) / 2
        mask = (torch.tanh(mask_raw.repeat(im_size[0], 1, 1)) + 1) / 2
        # Get the misclassification count for class s
        with torch.no_grad():
            # Embed the pattern
            images_with_bd = torch.clamp(images * (1 - mask) + pattern * mask, min=0, max=1)

            outputs = model(images_with_bd)
            _, predicted = outputs.max(1)
            misclassification += predicted.eq(labels).sum().item()
            total += len(labels)
        rho = misclassification / total

        # Modify the cost
        # Check if the current loss causes the misclassification fraction to be smaller than PI
        if rho >= pi:
            cost_up_counter += 1
            cost_down_counter = 0
        else:
            cost_up_counter = 0
            cost_down_counter += 1
        # If the misclassification fraction to be smaller than PI for more than PATIENCE iterations, reduce the cost;
        # else, increase the cost
        if cost_up_counter >= PATIENCE_UP and cost <= COST_MAX:
            cost_up_counter = 0
            cost *= COST_UP_MULTIPLIER
        elif cost_down_counter >= PATIENCE_DOWN:
            cost_down_counter = 0
            cost /= COST_DOWN_MULTIPLIER

        # print(iter_idx, rho, torch.sum(torch.abs(mask)).item(), cost, loss.item(), stopping_count)
        if iter_idx > 30 and mask_norm_best > 10000:
            break

        if rho >=pi and mask_norm_best < 3000:
            COST_MAX = 1e-1
        # Stopping criteria
        if rho >= pi and torch.sum(torch.abs(mask)) < mask_norm_best*0.99:
            mask_norm_best = torch.sum(torch.abs(mask))
            pattern_best = (torch.tanh(pattern_raw) + 1) / 2
            mask_best = (torch.tanh(mask_raw.repeat(im_size[0], 1, 1)) + 1) / 2
            associated_rho = rho
            stopping_count = 0
        else:
            stopping_count += 1

        if stopping_count >= PATIENCE_CONVERGENCE:
            break

        if verbose:
            print('iter {} mis-classification rate {} L1 norm {} cost {} stopping count {}'.format(iter_idx, rho, torch.sum(torch.abs(mask)), cost, stopping_count))
    print(print('iter {} mis-classification rate {} L1 norm {} cost {} stopping count {}'.format(iter_idx, rho, torch.sum(torch.abs(mask)), cost, stopping_count)))

    return pattern_best.detach().cpu(), mask_best.detach().cpu(), associated_rho

class AttackDataset(Dataset):

    def __init__(self, image, label):
        self.image = image
        self.label = label
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):

        return self.image[idx], self.label[idx]

