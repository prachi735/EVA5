from torchvision.datasets import CIFAR10
import torch
import numpy as np


# class AlbumentationsDataset(Dataset):
#     """__init__ and __len__ functions are the same as in TorchvisionDataset"""

#     def __init__(self, rimages, labels, transform=None):
#         self.rimages = rimages
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.rimages)

#     def __getitem__(self, idx):
#         label = self.labels[idx]
#         image = self.rimages[idx]
#         if self.transform:
#             augmented = self.transform(image=image)
#             image = augmented['image']
#         return image, label


class Cifar10AlbuDataset(CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        CIFAR10.__init__(self,root=root, train=train,
                         download=download, transform=transform)

    def __len__(self):
      return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


# def get_data(train_transforms, test_transforms, alb_dataset=True):
#     train_set = datasets.CIFAR10(
#         root='./data', train=True, download=True)
#     test_set = datasets.CIFAR10(
#         root='./data', train=False, download=True)
#     train_set = AlbumentationsDataset(
#         rimages=train_set.data,
#         labels=train_set.targets,
#         transform=train_transforms,
#     )

#     test_set = AlbumentationsDataset(
#         rimages=test_set.data,
#         labels=test_set.targets,
#         transform=test_transforms,
#     )
  #     return train_set, test_set


def get_dataloader(data, shuffle=True, batch_size=128, num_workers=4, pin_memory=True):

    cuda = torch.cuda.is_available()

    dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                           pin_memory=pin_memory) if cuda else dict(shuffle=True, batch_size=64)
    dataloader = torch.utils.data.DataLoader(data, ** dataloader_args)

    return dataloader


# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1


# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))
