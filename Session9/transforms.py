from torch.utils.data import  dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 
from torchvision import  transforms as tvt
import torch.tensor as pyT

class CIFARData(dataset.CIFAR10):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, transform=None):
        super().__init__()

    def __getitem__(self, index):
        im, label = super().__getitem__(index)
        if self.transform:
            im = pyT(self.transform(im))
        return im, label


def get_album_transforms(norm_mean, norm_std):
    '''
    get the train and test transform by albumentations
    '''
    train_transform = A.Compose([
        A.HorizontalFlip(p=1),
        # A.RGBShift(p=1),
        # A.Blur(blur_limit=11, p=1),
        # A.RandomBrightness(p=1),
        # A.CLAHE(p=1),
        ToTensorV2()
        
    ])
    test_transform = A.Compose([A.Normalize(mean=norm_mean, std=norm_std, ),
                                ToTensorV2()
                                ])
    return train_transform, test_transform


def apply_album_transformation(data, transforms):
    return AlbumentationsDataset(
        rimages=data.data,
        labels=data.targets,
        class_to_idx = data.class_to_idx,
        classes = data.classes,
        transform=transforms,
    )


def get_torch_transforms(mean, std):
    train = tvt.Compose([
                        # transforms.Resize((28, 28)),
                        tvt.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                        tvt.RandomRotation((-5.0, 5.0)),
                        # transforms.RandomAffine((-5.0,5.0),fillcolor=1),
                        #transforms.RandomPerspective(),                    
                        # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
                        tvt.ToTensor(),
                        tvt.Normalize((mean), (std,))
                        
                        ])
    test = tvt.Compose([
        #  transforms.Resize((28, 28)),
        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
        tvt.ToTensor(),
        tvt.Normalize((mean,), (std,))
        
    ])
    return train,test
