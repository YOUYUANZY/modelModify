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
        nn.ReLU6(),
    )


# Depthwise Separable Convolution
def conv_DW(inp, oup, stride=1):
    return nn.Sequential(
        conv_dw(inp, stride),
        conv_pw(inp, oup)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class MobileNetV1_md(nn.Module):
    def __init__(self):
        super(MobileNetV1_md, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_DW(32, 64, 1),

            conv_DW(64, 128, 2),
            conv_bn(128, 128, 1),

            conv_DW(128, 256, 2),
            conv_DW(256, 256, 1),
        )
        self.stage2 = nn.Sequential(
            conv_DW(256, 512, 2),
            conv_DW(512, 512, 1),
            conv_DW(512, 512, 1),
            conv_DW(512, 512, 1),
        )
        self.stage3 = nn.Sequential(
            conv_DW(512, 1024, 2),
            conv_1x1_bn(1024, 1024),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    a = MobileNetV1_md()
    device = torch.device('cuda:0')
    a = a.to(device)
    a.cuda()
    summary(a, (3, 224, 224))
