from torchvision.datasets import CIFAR10
from torchvision import datasets
import torch
from PIL import Image
import numpy as np


class CIFARData(CIFAR10):
    def __init__(self, path, train, download, transform=None):
    
        super(CIFARData, self).__init__(path, train=train, download=download)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            img = self.transform(image=img)['image']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def get_data(train_transforms, test_transforms):
    train = datasets.CIFAR10('./data', train=True, download=True,transform=train_transforms)
    test = datasets.CIFAR10('./data', train=False, download=True,transform=test_transforms)
    return train, test


def get_dataloader(data, shuffle=True, batch_size=128, num_workers=4, pin_memory=True):

  cuda = torch.cuda.is_available()

  dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=pin_memory) if cuda else dict(shuffle=True, batch_size=64)
  dataloader = torch.utils.data.DataLoader(data, ** dataloader_args)

  return dataloader
