import torch.nn as nn
import torch
import torch.optim as optim
from torchsummary import summary
from torch.optim import lr_scheduler
from model import ResNet_18

def get_optimizer(model, lr=0.01,
                  momentum=0.9, weight_decay=5e-4):

    return optim.SGD(model.parameters(), lr=0.01,momentum=momentum, weight_decay=weight_decay)


def get_scheduler(optimizer, lr_policy):
    if lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # else:
    #     return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def get_loss_function():
    return nn.CrossEntropyLoss()


def get_model_summary(model, input_size=(3, 32, 32)):
    return summary(model, input_size=input_size)


def get_previous_model(model_path,device):
    model =  ResNet_18()
    model.load_state_dict(torch.load(model_path))
    return model.to(device)



def get_learning_rate():
    lr = 0.01
    return lr
