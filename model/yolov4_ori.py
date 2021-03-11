import sys
sys.path.append("..")
# coding=utf-8
#================================================================
# Editor: PyCharm
# Time: 2020/10/11 21:06
# Author: Zhang Songhua
# Description: *
#================================================================


import torch
import torch.nn as nn
from collections import OrderedDict
from model.backbones.CSPDarknet import darknet53
import config.yolov3_config_voc as cfg
from model.head.yolo_head import Yolo_head
from model.layers.conv_module import Convolutional
import torchsummary as summary
import numpy as np

class Channel_Selective_Conv(nn.Module):
    def __init__(self, in_c, out_c, med_c=None, rate=0.3):
        super(Channel_Selective_Conv, self).__init__()
        self.out_c = out_c
        if med_c is None:
            med_c = in_c // 2
        self.linear = nn.Sequential(
            nn.Conv2d(in_c, med_c, 1, 1, 0, bias=True),
            nn.Conv2d(med_c, in_c, 1, 1, 0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_ = self.linear(x)
        xs = self.sigmoid(x_)
        _, xi = xs.topk(self.out_c, 1)
        out1 = torch.gather(x, 1, xi)
        out2 = torch.gather(x_, 1, xi)
        return out1 * out2


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
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


class SE_Conv(nn.Module):
    def __init__(self, ch_in, out_c, reduction=16):
        super(SE_Conv, self).__init__()
        self.se_conv = SE_Block(ch_in, reduction)
        self.dc_conv = nn.Conv2d(ch_in, out_c, 1, 1, 0)

    def forward(self, x):
        x = self.se_conv(x)
        x = self.dc_conv(x)
        return x


def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


# ---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
# ---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


# ---------------------------------------------------#
#   卷积 + 上采样
# ---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


# ---------------------------------------------------#
#   三次卷积块
# ---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        # Channel_Selective_Conv(in_filters, filters_list[0]),
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


# ---------------------------------------------------#
#   五次卷积块
# ---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        # Channel_Selective_Conv(in_filters, filters_list[0]),
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


# ---------------------------------------------------#
#   最后获得yolov4的输出
# ---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        #  backbone
        self.backbone = darknet53(None)

        # Neck
        self.conv1 = make_three_conv([512, 1024], 1024)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512, 1024], 2048)

        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = nn.Sequential(
            # conv2d(512, 256, 1),
            Channel_Selective_Conv(512, 256),
        )
        self.make_five_conv1 = make_five_conv([256, 512], 512)

        self.upsample2 = Upsample(256, 128)
        self.conv_for_P3 = nn.Sequential(
            # conv2d(256, 128, 1),
            Channel_Selective_Conv(256, 128),
        )
        self.make_five_conv2 = make_five_conv([128, 256], 256)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # 4+1+num_classes
        final_out_filter2 = num_anchors * (5 + num_classes)
        self.yolo_head3 = yolo_head([256, final_out_filter2], 128)

        self.down_sample1 = conv2d(128, 256, 3, stride=2)
        self.make_five_conv3 = make_five_conv([256, 512], 512)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter1 = num_anchors * (5 + num_classes)
        self.yolo_head2 = yolo_head([512, final_out_filter1], 256)

        self.down_sample2 = conv2d(256, 512, 3, stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024], 1024)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter0 = num_anchors * (5 + num_classes)
        self.yolo_head1 = yolo_head([1024, final_out_filter0], 512)

    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)

        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        P5 = self.conv2(P5)

        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4, P5_upsample], axis=1)
        P4 = self.make_five_conv1(P4)

        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3, P4_upsample], axis=1)
        P3 = self.make_five_conv2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], axis=1)
        P4 = self.make_five_conv3(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], axis=1)
        P5 = self.make_five_conv4(P5)

        out2 = self.yolo_head3(P3)
        out1 = self.yolo_head2(P4)
        out0 = self.yolo_head1(P5)
        return out0, out1, out2
class YoloBody_ORI(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody_ORI, self).__init__()
        #  backbone
        self.backbone = darknet53(None)

        self.conv1 = make_three_conv([512,1024],1024)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512,1024],2048)

        self.upsample1 = Upsample(512,256)
        self.conv_for_P4 = conv2d(512,256,1)
        self.make_five_conv1 = make_five_conv([256, 512],512)

        self.upsample2 = Upsample(256,128)
        self.conv_for_P3 = conv2d(256,128,1)
        self.make_five_conv2 = make_five_conv([128, 256],256)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # 4+1+num_classes
        final_out_filter2 = num_anchors * (5 + num_classes)
        self.yolo_head3 = yolo_head([256, final_out_filter2],128)

        self.down_sample1 = conv2d(128,256,3,stride=2)
        self.make_five_conv3 = make_five_conv([256, 512],512)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter1 =  num_anchors * (5 + num_classes)
        self.yolo_head2 = yolo_head([512, final_out_filter1],256)


        self.down_sample2 = conv2d(256,512,3,stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024],1024)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter0 =  num_anchors * (5 + num_classes)
        self.yolo_head1 = yolo_head([1024, final_out_filter0],512)


    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)

        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        P5 = self.conv2(P5)

        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4,P5_upsample],axis=1)
        P4 = self.make_five_conv1(P4)

        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3,P4_upsample],axis=1)
        P3 = self.make_five_conv2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample,P4],axis=1)
        P4 = self.make_five_conv3(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample,P5],axis=1)
        P5 = self.make_five_conv4(P5)

        out2 = self.yolo_head3(P3)
        out1 = self.yolo_head2(P4)
        out0 = self.yolo_head1(P5)

        return out0, out1, out2
class Yolov4(nn.Module):
    def __init__(self):
        super(Yolov4, self).__init__()
        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__out_channel = cfg.MODEL["ANCHORS_PER_SCLAE"] * (self.__nC + 5)
#        self.__backnone = YoloBody(3, 20)
        self.__backnone = YoloBody_ORI(3, 20)
        self.__head_s = Yolo_head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        self.__head_m = Yolo_head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])
        self.__head_l = Yolo_head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])


    def forward(self, x):
        out = []
        x_l, x_m, x_s = self.__backnone(x)
        s = self.__head_s(x_s)
        m = self.__head_m(x_m)
        l = self.__head_l(x_l)
        out.append(s)
        out.append(m)
        out.append(l)

        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)

    def __init_weights(self):
        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                print("initing {}".format(m))

    def load_darknet_weights(self, weight_file, cutoff=52):
        print('load pretrained weights : ', weight_file)

        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.float32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            pass

if __name__ == '__main__':
    from thop import profile

    model = Yolov4()
    summary.summary(model, (3, 416, 416), 1, 'cpu')
    para = torch.randn(1, 3, 416, 416)
    flops, params = profile(model, (para,))
    print(flops)
    print(params)
