import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 
from torchvision import transforms


def get_album_transforms(norm_mean, norm_std):
    '''
    get the train and test transform by albumentations
    '''
    train_transform = A.Compose([
        #A.SmallestMaxSize(max_size=160),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
        #                    rotate_limit=15, p=0.5),
        A.RandomCrop(height=128, width=128),
        # A.RGBShift(r_shift_limit=15, g_shift_limit=15,
        #            b_shift_limit=15, p=0.5),
        # A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=norm_mean, std=norm_std ),
        ToTensorV2()
        
    ])
    test_transform = A.Compose([A.Normalize(mean=norm_mean, std=norm_std ),
                                ToTensorV2()
                                ])
    return train_transform, test_transform


def get_torch_transforms(mean, std):
    train_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       #transforms.RandomRotation((-5.0, 5.0), fill=(1,)),
                                      # transforms.RandomAffine((-5.0,5.0),fillcolor=1),
                                       #transforms.RandomPerspective(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                       ])
    test_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                       ])
    return train_transforms,test_transforms
