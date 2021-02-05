import torch
from torchvision import datasets, utils
import numpy
import albumentations as A

def get_dataset(name, train_transforms = None, test_transforms = None):
    if name == 'MNIST':
        train = datasets.MNIST('./data', train=True, download=True,transform=train_transforms)
        test = datasets.MNIST('./data', train=True, download=True,transform=test_transforms)
    elif name == 'CIFAR10':
        train = datasets.CIFAR10(root='./data', train=True,download=True,transform=train_transforms)
        test  = datasets.CIFAR10(root='./data', train=False,download=True,transform=test_transforms)
    
    return train,test
   

def get_transforms(mean, std):
    train = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),            
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        A.ToTensor(),
        A.Normalize(mean,std)
    ])

    test = transforms.Compose([
        #  transforms.Resize((28, 28)),
        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        transforms.ToTensor(),
        # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
        transforms.Normalize((mean,), (std,))
    ])
    return train,test


def get_dataloader(train,test,**dataloader_args):
    train = torch.utils.data.DataLoader(train, ** dataloader_args)
    test = torch.utils.data.DataLoader(test, **dataloader_args)
    return train, test


def get_data_stats(data):
    return {'Numpy Shape': data.cpu().numpy().shape,
    'Tensor Shape': data.size(),
    'min': torch.min(data),
    'max': torch.max(data),
    'mean': torch.Tensor.float(data).mean(),
    'std': torch.Tensor.float(data).std(),
    'var': torch.Tensor.float(data).var()}


#def get_data(data_loader):
#    dataiter = iter(data_loader)
#    images, labels = dataiter.next()
#    return images, lables
