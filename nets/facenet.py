import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
from torchvision.models import MobileNetV2

from nets.mobilefacenet_modify import MobileFaceNet
from nets.mobilenet_v1 import MobileNetV1
from nets.mobilenet_v3 import MobileNetV3_Large


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


class mobilenet_v2(nn.Module):
    def __init__(self):
        super(mobilenet_v2, self).__init__()
        self.model = MobileNetV2()
        del self.model.classifier

    def forward(self, x):
        x = self.model.features(x)
        # x = x.mean(3).mean(2)
        return x


class mobilenet_v3_L(nn.Module):
    def __init__(self):
        super(mobilenet_v3_L, self).__init__()
        self.model = MobileNetV3_Large()
        del self.model.gap
        del self.model.drop
        del self.model.linear4

    def forward(self, x):
        x = self.model.hs1(self.model.bn1(self.model.conv1(x)))
        x = self.model.bneck(x)

        x = self.model.hs2(self.model.bn2(self.model.conv2(x)))
        return x

class mobilefacenet(nn.Module):
    def __init__(self):
        super(mobilefacenet, self).__init__()
        self.model = MobileFaceNet(128)
        del self.model.features
        del self.model.last_bn

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2_dw(x)
        x = self.model.conv_23(x)
        x = self.model.conv_3(x)
        x = self.model.conv_34(x)
        x = self.model.conv_4(x)
        x = self.model.conv_45(x)
        x = self.model.conv_5(x)

        x = self.model.sep(x)
        x = self.model.sep_bn(x)
        x = self.model.prelu(x)

        x = self.model.GDC_dw(x)
        x = self.model.GDC_bn(x)
        return x


class Facenet(nn.Module):
    def __init__(self, backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128,
                 num_classes=None, mode="train"):
        super(Facenet, self).__init__()
        if backbone == "mobilenet":
            self.backbone = mobilenet()
            flat_shape = 1024
        elif backbone == "mobilenetv2":
            self.backbone = mobilenet_v2()
            flat_shape = 1280
        elif backbone == "mobilenetv3_L":
            self.backbone = mobilenet_v3_L()
            flat_shape = 960
        elif backbone == "mobilefacenet":
            self.backbone = mobilefacenet()
            flat_shape = 512
        else:
            raise ValueError('Unsupported backbone - `{}`.'.format(backbone))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size, bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x, mode="predict"):
        x = self.backbone(x)
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
    a = Facenet(backbone='mobilenet', mode='predict')
    # for name, value in a.named_parameters():
    #     print(name)
    device = torch.device('cuda:0')
    a = a.to(device)
    a.cuda()
    summary(a, (3, 224, 224))
