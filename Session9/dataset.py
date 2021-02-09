from torch.utils.data.dataset import Dataset
from torchvision import datasets
import torch
from PIL import Image
import numpy as np


class AlbumentationsDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, rimages, labels, transform=None):
        self.rimages = rimages
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.rimages)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.rimages[idx]
        if self.transform:
            augmented = self.transform(image=image)
            image = torch.from_numpy(augmented['image']).reshape([3,32,32])
        return image, label


def get_data(train_transforms, test_transforms, alb_dataset=True):
    train_set = datasets.CIFAR10(
        root='./data', train=True, download=True)
    test_set = datasets.CIFAR10(
        root='./data', train=False, download=True)
    train_set = AlbumentationsDataset(
        rimages=train_set.data,
        labels=train_set.targets,
        transform=train_transforms,
    )

    test_set = AlbumentationsDataset(
        rimages=test_set.data,
        labels=test_set.targets,
        transform=test_transforms,
    )
    return train_set, test_set


def get_dataloader(data, shuffle=True, batch_size=128, num_workers=4, pin_memory=True):

    cuda = torch.cuda.is_available()

    dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                           pin_memory=pin_memory) if cuda else dict(shuffle=True, batch_size=64)
    dataloader = torch.utils.data.DataLoader(data, ** dataloader_args)

    return dataloader
