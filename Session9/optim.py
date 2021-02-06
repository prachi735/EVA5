import torch.optim as optim


def get_optimizer(model, lr=0.01,
                  momentum=0.9, weight_decay=5e-4):

    return optim.SGD(model.parameters(), lr=0.01,momentum=momentum, weight_decay=weight_decay)


def get_scheduler(optimizer, T_max=200):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
