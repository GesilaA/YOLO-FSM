import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.conv_module import Convolutional
# from ..layers.blocks_module import SE_block
from ..layers.blocks_module import my_SE_block as SE_block
import numpy as np

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SE_Conv(nn.Module):
    def __init__(self, ch_in, out_c, reduction=16):
        super(SE_Conv, self).__init__()
        self.se_conv = SE_Block(ch_in, reduction)
        self.dc_conv = nn.Conv2d(ch_in, out_c, 1, 1, 0)

    def forward(self, x):
        x = self.se_conv(x)
        x = self.dc_conv(x)
        return x

def normal_dis(x, miu=5, delta=1):
    miu = miu
    delta = delta
    y = 1/(np.sqrt(2*np.pi)*delta)*torch.exp(-(x-miu)**2/(2*delta*delta))
    return y

class Channel_Selective_Conv(nn.Module):
    def __init__(self, in_c, out_c, med_c=None, rate=0.3):
        super(Channel_Selective_Conv, self).__init__()
        self.out_c = out_c
        if med_c is None:
            med_c = in_c // 2
        self.linear = nn.Sequential(
            #nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, med_c, 1, 1, 0, bias=True),
            nn.Conv2d(med_c, in_c, 1, 1, 0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_ = self.linear(x)
        xs = self.sigmoid(x_)
        _, xi = xs.topk(self.out_c, 1)
        out1 = torch.gather(x, 1, xi)
        out2 = torch.gather(x_,1, xi)
        return out1 * out2

    # x_ = self.linear(x)
    # xs = self.sigmoid(x_)
    # _, xi = xs.topk(self.out_c, 1)
    # out1 = torch.gather(x, 1, xi)
    # out2 = torch.gather(x_, 1, xi)
    # return out2

class Upsample(nn.Module):
    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

class ConvUpSample(nn.Module):
    def __init__(self, scale_factor=2):
        super(ConvUpSample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        pass

class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        """
        x1 means previous output; x2 means current output
        """
        out = torch.cat((x2, x1), dim=1)
        return out

class FPN_YOLOV3(nn.Module):
    """
    FPN for yolov3, and is different from original FPN or retinanet' FPN.
    """
    def __init__(self, fileters_in, fileters_out):
        super(FPN_YOLOV3, self).__init__()

        fi_0, fi_1, fi_2 = fileters_in
        fo_0, fo_1, fo_2 = fileters_out

        # large
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0, filters_out=512, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1,pad=0, norm="bn",
                          activate="leaky"),
        )
        self.__conv0_0 = Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1,
                                       pad=1, norm="bn", activate="leaky")
        self.__conv0_1 = Convolutional(filters_in=1024, filters_out=fo_0, kernel_size=1,
                                       stride=1, pad=0)


        self.__conv0 = Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                                      activate="leaky")
        self.__upsample0 = Upsample(scale_factor=2)
        self.__route0 = Route()

        # medium
        self.__conv_set_1 = nn.Sequential(
            #Channel_Selective_Conv(fi_1 + 256, 256),
            # SE_Conv(fi_1 + 256, 256),
            Convolutional(filters_in=fi_1+256, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
        )
        self.__conv1_0 = Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1,
                                       pad=1, norm="bn", activate="leaky")
        self.__conv1_1 = Convolutional(filters_in=512, filters_out=fo_1, kernel_size=1,
                                       stride=1, pad=0)


        self.__conv1 = Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                                     activate="leaky")
        self.__upsample1 = Upsample(scale_factor=2)
        self.__route1 = Route()

        # small
        self.__conv_set_2 = nn.Sequential(
            #Channel_Selective_Conv(fi_2+128, 128),
            #SE_Conv(fi_2 + 128, 128),
            Convolutional(filters_in=fi_2+128, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
        )
        self.__conv2_0 = Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1,
                                       pad=1, norm="bn", activate="leaky")
        self.__conv2_1 = Convolutional(filters_in=256, filters_out=fo_2, kernel_size=1,
                                       stride=1, pad=0)

    def forward(self, x0, x1, x2):  # large, medium, small
        # large
        r0 = self.__conv_set_0(x0)
        out0 = self.__conv0_0(r0)
        out0 = self.__conv0_1(out0)

        # medium
        r1 = self.__conv0(r0)
        r1 = self.__upsample0(r1)
        x1 = self.__route0(x1, r1)
        r1 = self.__conv_set_1(x1)
        out1 = self.__conv1_0(r1)
        out1 = self.__conv1_1(out1)

        # small
        r2 = self.__conv1(r1)
        r2 = self.__upsample1(r2)
        x2 = self.__route1(x2, r2)
        r2 = self.__conv_set_2(x2)
        out2 = self.__conv2_0(r2)
        out2 = self.__conv2_1(out2)

        return out2, out1, out0  # small, medium, large
