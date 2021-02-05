import torch
from torchvision import datasets, transforms,utils
import numpy
import albumentations as A

def get_dataset(name, train_transforms = None, test_transforms = None):
    if name == 'CIFAR10':
        train = datasets.CIFAR10('./data', train=True, download=True,transform=train_transforms)
        test = datasets.CIFAR10('./data', train=False, download=True,transform=test_transforms)
    else:
        train = datasets.CIFAR100(root='./data', train=True,download=True,transform=train_transforms)
        test  = datasets.CIFAR100(root='./data', train=False,download=True,transform=test_transforms)
    return train,test
   

def get_transforms(mean, std):
    train = A.Compose([
        # A.RandomRotate90(),
        # A.Flip(),
        # A.Transpose(),
        # A.OneOf([
        #     A.IAAAdditiveGaussianNoise(),
        #     A.GaussNoise(),
        # ], p=0.2),
        # A.OneOf([
        #     A.MotionBlur(p=.2),
        #     A.MedianBlur(blur_limit=3, p=0.1),
        #     A.Blur(blur_limit=3, p=0.1),
        # ], p=0.2),
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # A.OneOf([
        #     A.OpticalDistortion(p=0.3),
        #     A.GridDistortion(p=.1),
        #     A.IAAPiecewiseAffine(p=0.3),
        # ], p=0.2),
        # A.OneOf([
        #     A.CLAHE(clip_limit=2),
        #     A.IAASharpen(),
        #     A.IAAEmboss(),
        #     A.RandomBrightnessContrast(),            
        # ], p=0.3),
        # A.HueSaturationValue(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    return train,test


def get_dataloader(train,test,**dataloader_args):
    train = torch.utils.data.DataLoader(train, ** dataloader_args)
    test = torch.utils.data.DataLoader(test, **dataloader_args)
    return train, test


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)

    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std
