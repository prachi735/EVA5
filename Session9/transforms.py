import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 
from torchvision import transforms


def get_album_transforms(norm_mean, norm_std):
    '''
    get the train and test transform by albumentations
    '''
    train_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.ShiftScaleRotate(shift_limit=0.0625,
                           scale_limit=0.2, rotate_limit=45, p=0.2),
        A.CLAHE(clip_limit=2),
        #A.RandomBrightnessContrast(),
        #A.HueSaturationValue(p=0.3),
        A.Cutout(num_holes=1, max_h_size=8, max_w_size=8,
               fill_value=0, always_apply=False, p=0.5),
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
