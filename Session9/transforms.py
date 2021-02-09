import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms


def get_album_transforms(norm_mean, norm_std):
    '''
    get the train and test transform by albumentations
    '''
    train_transform = A.Compose([
        # A.HorizontalFlip(p=1),
        # A.RGBShift(p=1),
        # A.Blur(blur_limit=11, p=1),
        # A.RandomBrightness(p=1),
        # A.CLAHE(p=1),
        #A.Normalize(mean=norm_mean, std=norm_std, ),
        ToTensorV2()

    ])
    test_transform = A.Compose([
        #A.Normalize(mean=norm_mean, std=norm_std, ),
        ToTensorV2()
    ])
    return train_transform, test_transform


def get_torch_transforms(mean, std):
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
