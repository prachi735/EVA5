import torch.nn as nn
import torch
import torch.optim as optim
from torchsummary import summary
from model import ResNet_18

def get_optimizer(model, lr=0.01,
                  momentum=0.9, weight_decay=5e-4):

    return optim.SGD(model.parameters(), lr=0.01,momentum=momentum, weight_decay=weight_decay)


def get_scheduler(optimizer, T_max=200):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)


def get_loss_function():
    return nn.CrossEntropyLoss()


def get_model_summary(model, input_size=(3, 32, 32)):
    return summary(model, input_size=input_size)


def get_previous_model(model_path,device):
    model =  ResNet_18()
    model.load_state_dict(torch.load(model_path))
    return model.to(device)

