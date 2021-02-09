from torch.utils.data.dataset import Dataset
from torchvision import datasets
import torch
from PIL import Image
import numpy as np

class AlbumentationsDataset(Dataset):
    
    def __init__(self, data, targets, classes, transforms=None):
        self.data = data
        self.targets = targets
        self.classes = classes
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        return image, target


def get_data(train_transforms, test_transforms, alb_dataset=True):
    train = datasets.CIFAR10('./data', train=True,
                             download=True, transform=train_transforms)
    test = datasets.CIFAR10('./data', train=False,
                            download=True, transform=test_transforms)
    if alb_dataset:
      train = AlbumentationsDataset(train.data, train.targets, train.classes, train_transforms)
      test = AlbumentationsDataset(train.data, train.targets, train.classes, test_transforms)
    return train, test


def get_dataloader(data, shuffle=True, batch_size=128, num_workers=4, pin_memory=True):

    cuda = torch.cuda.is_available()

    dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                           pin_memory=pin_memory) if cuda else dict(shuffle=True, batch_size=64)
    dataloader = torch.utils.data.DataLoader(data, ** dataloader_args)

    return dataloader
