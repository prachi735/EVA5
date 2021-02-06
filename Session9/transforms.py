from torchvision import transforms
import albumentations as A

def get_transforms():
    train_transforms = transforms.Compose([
        #  transforms.Resize((28, 28)),
        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        #transforms.RandomRotation((-5.0, 5.0), fill=(1,)),
        # transforms.RandomAffine((-5.0,5.0),fillcolor=1),
        #transforms.RandomPerspective(),
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
