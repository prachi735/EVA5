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
    train_transform = A.Compose([A.Normalize(mean=norm_mean, std=norm_std, always_apply=True, p=1.0),
                                 A.RandomCrop(32, 32),
                                 A.HorizontalFlip(p=0.5),
                                 A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1,
                                                  border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                                 A.ChannelDropout(channel_drop_range=(
                                     1, 1), fill_value=0, always_apply=False, p=0.5),
                                 A.Cutout(num_holes=1, max_h_size=16, max_w_size=16,
                                          fill_value=norm_mean, always_apply=False, p=0.5),
                                 A.Normalize(mean=norm_mean,std=norm_std ),
                                AP.transforms.ToTensor()
    ])
    
    test_transform = A.Compose([A.Normalize(mean=norm_mean,std=norm_std, ),
        AP.transforms.ToTensor()
    ])
    return train_transform, test_transform
