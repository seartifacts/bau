import torch.utils.data
import numpy as np
import os
import shutil
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(256),      
    transforms.CenterCrop(224)    
])

def open_image_convert_to_rgb(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img_transformed = transform(img)
        img_array = np.array(img_transformed)
    return img_array

root_path = '../../imagenet100'
directories = []
for dirpath, dirnames, filenames in os.walk(root_path):
    for dirname in dirnames:
        directories.append(os.path.join(dirpath, dirname))

train_x = []
train_y = []
test_x = []
test_y = []

for label in range(len(directories)):
    directory = directories[label]
    for jpgpath, jpgnames, filenames in os.walk(directory):
        total_num = len(filenames)
        train_num = int(total_num * 0.8)
        for i in range(total_num):
            img_array = open_image_convert_to_rgb(os.path.join(jpgpath, filenames[i]))
            if i < train_num:
                train_x.append(img_array)
                train_y.append(label)
            else:
                test_x.append(img_array)
                test_y.append(label)

train_x = np.array(train_x)
test_x = np.array(test_x)
train_y = np.array(train_y)
test_y = np.array(test_y)

train_idxs = np.array(list(range(len(train_x))))
np.random.shuffle(train_idxs)
train_x = train_x[train_idxs]
train_y = train_y[train_idxs]

test_idxs = np.array(list(range(len(test_x))))
np.random.shuffle(test_idxs)
test_x = test_x[test_idxs]
test_y = test_y[test_idxs]

np.save('../data/imagenet100/train_image.npy', train_x)
np.save('../data/imagenet100/train_label.npy', train_y)
np.save('../data/imagenet100/test_image.npy', test_x)
np.save('../data/imagenet100/test_label.npy', test_y)
