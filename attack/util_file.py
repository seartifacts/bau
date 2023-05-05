import torch
import h5py
import numpy as np
import os
from PIL import Image
import skimage.io as io
import cv2

def save_model(model, name):
    torch.save(model.state_dict(), name)

def load_model(model, name, device):
    model.load_state_dict(torch.load(name, map_location=torch.device(device)))
    return model

def load_dataset(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def save_image(img, path):
    image = Image.fromarray(img, "RGB")
    image.save(path)

def load_image(path):
    image = cv2.imread(path)
    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((32, 32))
    return np.array(resize_image)