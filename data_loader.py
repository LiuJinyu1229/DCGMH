import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import h5py
import os
import numpy as np
import random
import math

ROOT = 'xxx/xxx/xxx/'
if not os.path.exists(ROOT):
    raise Exception('The ROOT path is error.')

paths = {
    'flickr': ROOT + 'mir_cnn_twt.mat',
    'nuswide': ROOT + 'nus_cnn_twt.mat',
    'coco': ROOT + 'coco_cnn_twt.mat'
}
    
class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0., std=1.):
        super(AddGaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        rand = torch.randn(tensor.size(), device=tensor.device)
        return tensor + rand * self.std + self.mean

class RandomContrast(nn.Module):
    def __init__(self, contrast_factor=0.2):
        super(RandomContrast, self).__init__()
        self.contrast_factor = contrast_factor

    def forward(self, tensor):
        mean = tensor.mean()
        return (tensor - mean) * (1.0 + np.random.uniform(-self.contrast_factor, self.contrast_factor)) + mean

class FeaturePerturbation(nn.Module):
    def __init__(self, epsilon=0.1):
        super(FeaturePerturbation, self).__init__()
        self.epsilon = epsilon

    def forward(self, tensor):
        perturbation = (torch.randn(tensor.size(), device=tensor.device) * self.epsilon)
        return tensor + perturbation

class RandomSwapFeatures(nn.Module):
    def __init__(self, num_swaps=1):
        super(RandomSwapFeatures, self).__init__()
        self.num_swaps = num_swaps

    def forward(self, tensor):
        for _ in range(self.num_swaps):
            idx1, idx2 = random.sample(range(tensor.shape[1]), 2)
            idx = torch.arange(tensor.shape[0], device=tensor.device)
            temp = tensor[idx, idx1].clone()
            tensor[idx, idx1] = tensor[idx, idx2]
            tensor[idx, idx2] = temp
        return tensor

class RandomFeatureDeletion(nn.Module):
    def __init__(self, deletion_prob=0.1):
        super(RandomFeatureDeletion, self).__init__()
        self.deletion_prob = deletion_prob

    def forward(self, tensor):
        tensor = tensor.clone()
        mask = (torch.rand(tensor.size(), device=tensor.device) > self.deletion_prob).float()
        return tensor * mask

class Normalize(nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        return (tensor - self.mean) / self.std

image_augmentations = nn.Sequential(
    AddGaussianNoise(mean=0., std=0.1),
    RandomContrast(contrast_factor=0.2),
    FeaturePerturbation(epsilon=0.05),
    Normalize(mean=0.0, std=1.0)
)

text_augmentations = nn.Sequential(
    AddGaussianNoise(mean=0.0, std=0.1),
    RandomSwapFeatures(num_swaps=5),
    RandomFeatureDeletion(deletion_prob=0.1),
    Normalize(mean=0.0, std=1.0)
)

def label_turn(label, zero_turn, one_turn):
    zero_turn = int(zero_turn)
    one_turn = int(one_turn)
    zero_indices = (label == 0).nonzero(as_tuple=True)[0]
    one_indices = (label == 1).nonzero(as_tuple=True)[0]
    # Set `zero_turn` number of 0s to 1
    zero_indices = zero_indices[torch.randperm(len(zero_indices))[:zero_turn]]
    label[zero_indices] = 1
    
    # Set `one_turn` number keeping 1
    label[one_indices] = 0
    if one_turn > 0:
        one_indices = one_indices[torch.randperm(len(one_indices))[:one_turn]]
        label[one_indices] = 1
    return label

def generate_noisy_labels(labels, noise_level):
    """
    labels: the original labels
    noise_level: the proportion of samples to add noise to
    """
    labels_num = torch.sum(labels, dim=1)
    max_labels_num = torch.max(labels_num).item()
    min_labels_num = torch.min(labels_num).item()
    noisy_labels = labels.clone().detach()
    num_samples, num_labels = labels.shape
    num_noisy_samples = int(noise_level * num_samples)
    noisy_indices = torch.randperm(num_samples)[:num_noisy_samples]

    # Define partitions
    partitions = torch.chunk(noisy_indices, 4)
    num_same, num_diff, n_num_same, n_num_diff = partitions

    def process_num_labels(indices, zero_turn_ratio):
        for idx in indices:
            label = noisy_labels[idx]
            label_num = torch.sum(label).item()
            if zero_turn_ratio >= 0:
                zero_turn = math.ceil(label_num * zero_turn_ratio)
            else:
                zero_turn = torch.sum(label).item()
            one_turn = label_num - zero_turn
            label = label_turn(label, zero_turn, one_turn)
            noisy_labels[idx] = label

    # Process labels
    process_num_labels(num_same, 0.5)
    process_num_labels(num_diff, -1)

    def process_diff_labels(indices):
        for idx in indices:
            label = noisy_labels[idx]
            label_num = random.randint(min_labels_num, max_labels_num)
            while label_num == torch.sum(label).item():
                label_num = random.randint(min_labels_num, max_labels_num)
            zero_turn_ratio = 0.5 if indices is n_num_same else 1
            one_turn_fixed = label_num - math.ceil(label_num * 0.5) if indices is n_num_same else 0
            label = label_turn(label, math.ceil(label_num * zero_turn_ratio), one_turn_fixed)
            noisy_labels[idx] = label

    process_diff_labels(n_num_same)
    process_diff_labels(n_num_diff)

    return noisy_labels

def generate_flag(labels):
    labels_2d = labels.view(labels.size(0), -1)
    unique_labels, inverse_indices = torch.unique(labels_2d, dim=0, return_inverse=True)
    return inverse_indices

class DatasetProcess(Dataset):
    """
    Flicker 25k dataset.

    Args
        root(str): Path of dataset.
        mode(str, 'train', 'query', 'retrieval'): Mode of dataset.
        transform(callable, optional): Transform images.
    """
    def __init__(self, data_name, mode, noise = True, noise_type='pairflip', noise_level = 0.4):
        self.mode = mode
        h5f = h5py.File(paths[data_name], 'r')

        if mode == 'train':
            if noise:
                self.imgs = torch.tensor(h5f["I_tr"][:].T)
                self.txts = torch.tensor(h5f["T_tr"][:].T)
                self.labels = torch.tensor(h5f["L_tr"][:].T)
                self.nlabels = torch.tensor(generate_noisy_labels(self.labels, noise_level))
                self.flags = torch.tensor(generate_flag(self.labels))
            else:
                self.imgs = torch.tensor(h5f["I_tr"][:].T)
                self.txts = torch.tensor(h5f["T_tr"][:].T)
                self.labels = torch.tensor(h5f["L_tr"][:].T)
                self.nlabels = torch.tensor(h5f["L_tr"][:].T)
                self.flags = torch.tensor(generate_flag(self.labels))
        elif mode == 'test':
            self.imgs = torch.tensor(h5f["I_te"][:].T)
            self.txts = torch.tensor(h5f["T_te"][:].T)
            self.labels = torch.tensor(h5f["L_te"][:].T)
        elif mode == 'retrieval':
            self.imgs = torch.tensor(h5f["I_db"][:].T)
            self.txts = torch.tensor(h5f["T_db"][:].T)
            self.labels = torch.tensor(h5f["L_db"][:].T)
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        if self.mode == 'train':
            img = self.imgs[index]
            txt = self.txts[index]
            label = self.labels[index]
            nlabel = self.nlabels[index]
            return img, txt, label, nlabel, index
        else:
            img = self.imgs[index]
            txt = self.txts[index]
            label = self.labels[index]
            return img, txt, label, index

    def __len__(self):
        return self.imgs.shape[0]