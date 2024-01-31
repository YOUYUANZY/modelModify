import torch
from torchsummary import summary

from nets.facenet import Facenet
from nets.arcface import Arcface


def modelSummary(config):
    if config.model == 'facenet':
        FR = Facenet(backbone=config.backbone, mode='predict')
    elif config.model == 'arcface':
        FR = Arcface(backbone=config.backbone, mode='predict')
    else:
        raise ValueError('modelSummary error,unsupported model')
    device = torch.device('cuda:0')
    FR = FR.to(device)
    FR.cuda()
    x = config.input
    summary(FR, (3, x, x))
