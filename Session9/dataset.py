
from torchvision import datasets, transforms, utils


def get_data(train_transforms, test_transforms):
    train = datasets.CIFAR10('./data', train=True, download=True,transform=train_transforms)
    test = datasets.CIFAR10('./data', train=False, download=True,transform=test_transforms)
    return train, test

def get_cifar_classes():
    