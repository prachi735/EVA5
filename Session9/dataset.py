import torchvision

import albumentations as A
import albumentations.pytorch as AP
from torch.utils.data import Dataset
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
            image = augmented['image']
        return image, label


def get_album_transforms(norm_mean, norm_std):
    """get the train and test transform by albumentations"""
    album_train_transform = A.Compose([A.HorizontalFlip(p=.2),
                                       A.VerticalFlip(p=.2),
                                       A.Rotate(limit=15, p=0.5),
                                       A.Normalize(
        mean=[0.49, 0.48, 0.45],
        std=[0.25, 0.24, 0.26], ),
        AP.transforms.ToTensor()
    ])

    album_test_transform = A.Compose([A.Normalize(
        mean=[0.49, 0.48, 0.45],
        std=[0.25, 0.24, 0.26], ),
        AP.transforms.ToTensor()
    ])
    return(album_train_transform, album_test_transform)


def get_datasets():
    """Extract and transform the data"""
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True)
    return(train_set, test_set)


def trasnform_datasets(train_set, test_set, train_transform, test_transform):
    """Transform the data"""
    train_set = AlbumentationsDataset(
        rimages=train_set.data,
        labels=train_set.targets,
        transform=train_transform,
    )

    test_set = AlbumentationsDataset(
        rimages=test_set.data,
        labels=test_set.targets,
        transform=test_transform,
    )
    return(train_set, test_set)
