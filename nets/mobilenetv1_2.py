import torch
import torch.nn as nn
from torchsummary import summary


def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )


# Depthwise Convolution
def conv_dw(inp, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(),
    )


# Pointwise Convolution
def conv_pw(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


# Depthwise Separable Convolution
def conv_DW(inp, oup, stride=1):
    return nn.Sequential(
        conv_dw(inp, stride),
        conv_pw(inp, oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


# 倒残差块
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(inp * 6)
        # 倒残差
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return x + self.conv(x)


class MobileNetV1_2(nn.Module):
    def __init__(self):
        super(MobileNetV1_2, self).__init__()
        self.bn1 = conv_bn(3, 32, 2)
        self.dw1 = conv_dw(32, 1)
        self.pw1 = conv_pw(32, 16)

        self.stage = [InvertedResidual(16, 16, 1),
                      conv_bn(16, 32, 1),
                      InvertedResidual(32, 32, 1),
                      conv_bn(32, 64, 2),
                      InvertedResidual(64, 64, 1),
                      conv_bn(64, 64, 2),
                      InvertedResidual(64, 64, 1),
                      InvertedResidual(64, 64, 1),
                      InvertedResidual(64, 64, 1),
                      conv_bn(64, 96, 2),
                      InvertedResidual(96, 96, 1),
                      InvertedResidual(96, 96, 1),
                      conv_DW(96, 128, 1),
                      conv_DW(128, 128, 1),
                      conv_DW(128, 256, 1),
                      conv_DW(256, 256, 1),
                      conv_DW(256, 512, 1),
                      conv_DW(512, 512, 1),
                      conv_DW(512, 1024, 2),
                      conv_1x1_bn(1024, 1024)
                      ]
        self.stage = nn.Sequential(*self.stage)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.bn1(x)
        x = self.dw1(x)
        x = self.pw1(x)
        x = self.stage(x)
        x = self.avg(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    a = MobileNetV1_2()
    device = torch.device('cuda:0')
    a = a.to(device)
    a.cuda()
    summary(a, (3, 224, 224))
