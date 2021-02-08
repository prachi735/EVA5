from torchvision.datasets import CIFAR10
from torchvision import datasets
import torch
from PIL import Image

class CIFARData(CIFAR10):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, path, train, download, transforms=None):
        super().__init__(path, train=train, download=download)
        self.transforms = transforms

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def get_data(train_transforms, test_transforms):
    train = datasets.CIFAR10('./data', train=True, download=True,transform=train_transforms)
    test = datasets.CIFAR10('./data', train=False, download=True,transform=test_transforms)
    return train, test


def get_dataloader(data, shuffle=True, batch_size=128, num_workers=4, pin_memory=True):

  cuda = torch.cuda.is_available()

  dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=pin_memory) if cuda else dict(shuffle=True, batch_size=64)
  dataloader = torch.utils.data.DataLoader(data, ** dataloader_args)

  return dataloader
