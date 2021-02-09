from torch.utils.data.dataset import Dataset
from torchvision import datasets
import torch
from PIL import Image
import numpy as np


class Cifar10AlbuDataset(datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

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
