def get_transforms(type):
    if type = 'train':
        return transforms.Compose([
                                        #  transforms.Resize((28, 28)),
                                        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                        transforms.RandomRotation(
                                            (-5.0, 5.0), fill=(1,)),
                                        # transforms.RandomAffine((-5.0,5.0),fillcolor=1),
                                        #transforms.RandomPerspective(),
                                        transforms.ToTensor(),
                                        # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
                                        transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                        ])
    if type = 'test':
        return transforms.Compose([
        #  transforms.Resize((28, 28)),
        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        transforms.ToTensor(),
        # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def get_dataloader(**dataloader_args):
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train, ** dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)


def get_data_stats(data):
    #Dataset stats & sampledata
    
    print(' - Numpy Shape:', data.cpu().numpy().shape)
    print(' - Tensor Shape:', data.size())
    print(' - min:', torch.min(data))
    print(' - max:', torch.max(data))
    print(' - mean:', torch.Tensor.float(data).mean())
    print(' - std:', torch.Tensor.float(data).std())
    print(' - var:', torch.Tensor.float(data).var())


def get_data(data_loader)
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    return images, lables