import torch
from torchvision import datasets, transforms, utils

def get_data(name):
    if name == 'MNIST':
        train = datasets.MNIST('./data', train=True, download=True)
        test = datasets.MNIST('./data', train=True, download=True)
    elif name == 'CIFAR10':
        train_set = datasets.CIFAR10(root='./data', train=True,download=True)
        test_set  = datasets.CIFAR10(root='./data', train=False,download=True)
    
    return(train_set,test_set)
   
