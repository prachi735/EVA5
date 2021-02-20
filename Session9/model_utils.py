import torch.nn as nn
import torch
import torch.optim as optim
from torchsummary import summary
from torch.optim import lr_scheduler
from model import ResNet_18

def get_optimizer(model, lr=0.01,
                  momentum=0.9, weight_decay=5e-4):

    return optim.SGD(model.parameters(), lr=0.01,momentum=momentum, weight_decay=weight_decay)


def get_scheduler(args, optimizer, datasets):
    if args.scheduler == 'step':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=eval(args.milestones), gamma=args.lr_decay)
    elif args.scheduler == 'poly':
        total_step = (len(datasets['train']) / args.batch + 1) * args.epochs
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lambda x: (1-x/total_step) ** args.power)
    elif args.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=args.lr_decay, patience=args.patience)
    elif args.scheduler == 'constant':
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda x: 1)
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, args.T_max, args.min_lr)
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
