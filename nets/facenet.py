import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

from Attention.Triplet import Triplet
from nets.mobilenet import MobileNetV1


class mobilenet(nn.Module):
    def __init__(self):
        super(mobilenet, self).__init__()
        self.model = MobileNetV1()
        del self.model.fc
        del self.model.avg

    def forward(self, x):
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        return x


class Facenet(nn.Module):
    def __init__(self, backbone="mobilenet", attention='CBAM', dropout_keep_prob=0.5, embedding_size=128,
                 num_classes=None, mode="train"):
        super(Facenet, self).__init__()
        if backbone == "mobilenet":
            self.backbone = mobilenet()
            flat_shape = 1024
        else:
            raise ValueError('Unsupported backbone - `{}`.'.format(backbone))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size, bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if attention == 'CBAM':
            self.attention = CBAM(planes=flat_shape)
        elif attention == 'APNB':
            self.attention = APNB(channel=flat_shape)
        elif attention == 'AFNB':
            self.attention = AFNB(channel=flat_shape)
        elif attention == 'GCNet':
            self.attention = GCNet(inplanes=flat_shape, ratio=0.25)
        elif attention == 'SE':
            self.attention = SE(in_chnls=flat_shape, ratio=16)
        elif attention == 'scSE':
            self.attention = scSE(channel=flat_shape, ratio=16)
        elif attention == 'Triplet':
            self.attention = Triplet()
        else:
            self.attention = None
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x, mode="predict"):
        x = self.backbone(x)
        # attention
        if self.attention is not None:
            x = self.attention(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)

        if mode == 'predict':
            return x
        cls = self.classifier(before_normalize)
        return x, cls


if __name__ == '__main__':
    a = Facenet(mode='predict', attention="AFNB")
    # for name, value in a.named_parameters():
    #     print(name)
    device = torch.device('cuda:0')
    a = a.to(device)
    a.cuda()
    summary(a, (3, 112, 112))
