import torch.nn as nn
from ..layers.conv_module import Convolutional


class Residual_block(nn.Module):
    def __init__(self, filters_in, filters_out, filters_medium):

        super(Residual_block, self).__init__()
        self.__conv1 = Convolutional(filters_in=filters_in, filters_out=filters_medium, kernel_size=1, stride=1, pad=0,
                                     norm="bn", activate="leaky")
        self.__conv2 = Convolutional(filters_in=filters_medium, filters_out=filters_out, kernel_size=3, stride=1, pad=1,
                                     norm="bn", activate="leaky")

    def forward(self, x):
        r = self.__conv1(x)
        r = self.__conv2(r)
        out = x + r

        return out

class FeatureUp_block(nn.Module):
    def __init__(self, filters_in, filters_out):
        super(FeatureUp_block, self).__init__()
        self.res = True if filters_in == filters_out else False
        self.conv = Convolutional(filters_in=filters_in, filters_out=filters_out, kernel_size=1, stride=1, pad=0,
                                      norm="bn", activate="leaky")
    def forward(self, x):
        if self.res:
            x = x + self.conv(x)
        else:
            x = self.conv(x)
        return x

class SE_block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class my_SE_block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(my_SE_block, self).__init__()
        self.fc = nn.Sequential(
            # nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.Conv2d(ch_in, ch_in//reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_in//reduction),
            nn.ReLU(inplace=True),
            # nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Conv2d(ch_in//reduction, ch_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.Sigmoid()
        )

    def forward(self, x):
        # b, c, _, _ = x.size()
        # y = self.avg_pool(x).view(b, c)
        y = self.fc(x)
        return x * y.expand_as(x)