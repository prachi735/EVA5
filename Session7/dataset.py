import torchvision

def get_data(name):
    if name == 'MNIST':
        train = datasets.MNIST('./data', train=True, download=True)
        test = datasets.MNIST('./data', train=True, download=True)
    else if name == 'CIFAR10':
        train_set = datasets.CIFAR10(root='./data', train=True,download=True)
        test_set  = datasets.CIFAR10(root='./data', train=False,download=True)
    
    return(train_set,test_set)
    
def get_transforms(mean, std):
    
    train = transforms.Compose([
                                        #  transforms.Resize((28, 28)),
                                        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                        transforms.RandomRotation(
                                            (-5.0, 5.0), fill=(1,)),
                                        # transforms.RandomAffine((-5.0,5.0),fillcolor=1),
                                        #transforms.RandomPerspective(),
                                        transforms.ToTensor(),
                                        # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
                                        transforms.Normalize(
                                            (mean,), (std,))
                                        ])
    test = transforms.Compose([
        #  transforms.Resize((28, 28)),
        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        transforms.ToTensor(),
        # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
        transforms.Normalize((mean,), (std,))
    ])
    return train,test


def get_dataloader(train,test,**dataloader_args):
    # train dataloader
    train = torch.utils.data.DataLoader(train, ** dataloader_args)

    # test dataloader
    test = torch.utils.data.DataLoader(test, **dataloader_args)
    
    return train, test


def get_data_stats(data):
    #Dataset stats     
    return {' - Numpy Shape': data.cpu().numpy().shape,
    'Tensor Shape': data.size(),
    'min': torch.min(data),
    'max': torch.max(data),
    'mean': torch.Tensor.float(data).mean(),
    'std': torch.Tensor.float(data).std(),
    'var': torch.Tensor.float(data).var()}


def get_data(data_loader):
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    return images, lables
