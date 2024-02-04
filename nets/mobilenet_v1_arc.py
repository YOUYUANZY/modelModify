import torch
import torch.nn as nn
from torchsummary import summary


def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )


# 深度可分离卷积
def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(),
    )


class MobileNetV1_arc(nn.Module):
    def __init__(self,embedding_size):
        super(MobileNetV1_arc, self).__init__()
        self.stage1 = nn.Sequential(
            # 160,160,3 -> 80,80,32
            conv_bn(3, 32, 2),
            # 80,80,32 -> 80,80,64
            conv_dw(32, 64, 1),

            # 80,80,64 -> 40,40,128
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),

            # 40,40,128 -> 20,20,256
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )
        self.stage2 = nn.Sequential(
            # 20,20,256 -> 10,10,512
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        self.stage3 = nn.Sequential(
            # 10,10,512 -> 5,5,1024
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )

        self.GDC_dw = nn.Conv2d(1024, 1024, kernel_size=7, bias=False, groups=512)
        self.GDC_bn = nn.BatchNorm2d(1024)

        self.features = nn.Conv2d(1024, embedding_size, kernel_size=1, bias=False)
        self.last_bn = nn.BatchNorm2d(embedding_size)

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

        x = self.GDC_dw(x)
        x = self.GDC_bn(x)

        x = self.features(x)
        x = self.last_bn(x)
        return x


if __name__ == '__main__':
    a = MobileNetV1(128)
    device = torch.device('cuda:0')
    a = a.to(device)
    a.cuda()
    summary(a, (3, 224, 224))
