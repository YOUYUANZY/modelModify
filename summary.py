import torch
from torchsummary import summary

from nets.facenet import Facenet


def modelSummary(config):
    FR = Facenet(backbone=config.backbone, mode='predict')
    device = torch.device('cuda:0')
    FR = FR.to(device)
    FR.cuda()
    x = config.input
    summary(FR, (3, x, x))
