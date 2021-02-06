from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
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
    '''
    get the train and test transform by albumentations
    '''
    train_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        # A.OneOf([A.RandomSizedCrop(min_max_height=[15,15], height=8, width=8, w2h_ratio=1.0, interpolation=1,
        #                            always_apply=False, p=1.0), 
        #                            A.RandomCrop(height=8, width=8, always_apply=False, p=1.0)],p=0.5),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625,
                           scale_limit=0.2, rotate_limit=45, p=0.2),
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
        A.Normalize(mean=norm_mean, std=norm_std, always_apply=True, p=1.0),
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
        transform=transforms,
    )
