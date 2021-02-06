
from torchvision import datasets
import torch

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
