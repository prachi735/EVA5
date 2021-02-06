import albumentations as A
import albumentations.pytorch as AP

from torchvision import transforms


def get_torch_transforms():
    train_transforms = transforms.Compose([
        #  transforms.Resize((28, 28)),
        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        #transforms.RandomRotation((-5.0, 5.0), fill=(1,)),
        # transforms.RandomAffine((-5.0,5.0),fillcolor=1),
        # transforms.RandomPerspective(),
        transforms.ToTensor(),
        # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transforms = transforms.Compose([
        #  transforms.Resize((28, 28)),
        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        transforms.ToTensor(),
        # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return train_transforms, test_transforms


def get_album_transforms(norm_mean, norm_std):
    '''
    get the train and test transform by albumentations
    '''
    train_transform = A.Compose([A.HorizontalFlip(p=.2),
                                 A.VerticalFlip(p=.2),
                                 A.Rotate(limit=15, p=0.5),
                                 
                                 A.Normalize(mean=[0.49, 0.48, 0.45],        std=[0.25, 0.24, 0.26], ),
                                AP.transforms.ToTensor()
    ])

    test_transform = A.Compose([A.Normalize(
        mean=[0.49, 0.48, 0.45],
        std=[0.25, 0.24, 0.26], ),
        AP.transforms.ToTensor()
    ])
    return train_transform, test_transform
