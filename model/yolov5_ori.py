import sys

sys.path.append("..")
# coding=utf-8
# ================================================================
# Editor: PyCharm
# Time: 2021/01/05 21:06
# Author: Zhang Songhua
# Description: *
# ================================================================


import torch
import torch.nn as nn
from collections import OrderedDict
from model.backbones.CSPDarknet import darknet53
import config.yolov3_config_voc as cfg
from model.head.yolo_head import Yolo_head
from model.layers.conv_module import Convolutional
import torchsummary as summary
import numpy as np
import torch.nn.functional as F
import math
'''
https://blog.csdn.net/Q1u1NG/article/details/107511465 
https://zhuanlan.zhihu.com/p/143747206
参考结构图
同时也参考u版的yolov5s的model打印出来的结构'''


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

class Conv(nn.Module):
    def __init__(self,c_in,c_out,k=1,s=1,p=0):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(c_in,c_out,k,s,p,bias=False),
                                  nn.BatchNorm2d(c_out),
                                  nn.Hardswish())
    def forward(self,x):
        return self.conv(x)

class Focus(nn.Module): #
    def __init__(self,c_in=12,c_out=32):
        super(Focus, self).__init__()
        self.conv_f = Conv(c_in,c_out,3,1,1)
    def forward(self, x):  # x(b,c,h,w) -> y(b,4c,h/2,w/2) 切片后合并。
        return self.conv_f(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1))


class Bottleneck(nn.Module): #实际是借鉴resnet残差结构，类似的，注意中间的部分要保证特征图大小不变。 通道数量输入输出一致。
    def __init__(self,c_):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            Conv(c_,c_), #此处，通道大小未变，并未如往常的残差结构缩小通道减少计算量，主要是为了融合通道。
            Conv(c_,c_,3,1,1)
        )
    def forward(self,x):
         return self.bottleneck(x)+x
class BL(nn.Module): #concat之后的batchnormal+激活。
    def __init__(self,c_):
        super(BL, self).__init__()
        self.bl = nn.Sequential(
            nn.BatchNorm2d(c_),
            nn.LeakyReLU(0.1,True)
        )
    def forward(self,x):
        return self.bl(x)

class Bottleneckcsp(nn.Module): #c_ = c_out*e,e=0.5,默认缩放其通道的因子。 csp的实现不是直接的对输入的特征图拆分，而是通过1*1卷积的方式进行通道融合且减少，默认是一半。
    def __init__(self,c_in,c_out,n=3,depth_multiple=0.33,e=0.5): #n为重复bottleneck的数量。
        super(Bottleneckcsp, self).__init__()
        c_ = int(c_in*e)
        self.n = max(round(n*depth_multiple),1)
        self.conv1 = Conv(c_in,c_) #bottleneck之前的那个卷积，融合并减少通道的作用。
        self.bottleneck = nn.ModuleList([Bottleneck(c_) for i in range(self.n)])
        self.conv2 = nn.Conv2d(c_,c_,1,1,bias=False) #bottleneck之后的那个卷积 作用，融合通道。
        self.conv3 = nn.Conv2d(c_in,c_,1,1,bias=False) #跳跃连接的卷积，目的，调整通道和bottleneck的输出通道一致。也有融合并减少通道的作用。
        #然后concat在forward中体现。
        self.bl = BL(c_*2)
        self.conv4 = Conv(c_*2,c_out,1,1) #确定输出通道数量。

    def forward(self,x):
        y = self.conv1(x)
        for i in range(self.n):
            y = self.bottleneck[i](y)
        y = self.conv2(y)
        y_shortcut = self.conv3(x)
        y = torch.cat((y,y_shortcut),dim=1) #通道上，拼接
        y = self.bl(y)
        y = self.conv4(y)
        return y

'''#注意spp模块是借鉴sppnet的思想，并不会像空间金字塔池化一样输出固定大小的特征图。spp模块作用是全局和局部特征融合，显著的分离了最重要的上下文特征。
最后不改变输入特征图大小，也就是说输入特征图h*w，输出还是h*w'''
class SPP(nn.Module):
    def __init__(self,c_in,c_out):
        super(SPP, self).__init__()
        self.conv1 = Conv(c_in,c_in//2,1,1)
        self.maxpool = nn.ModuleList([nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False),
                                      nn.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False),
                                      nn.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)])
        ##然后concat在forward中体现。
        self.conv2 = Conv(c_in*2,c_out,1,1)

    def forward(self,x):
        y = self.conv1(x)
        max_pool = [y]
        for m in self.maxpool:
            max_pool.append(m(y))
        y = torch.cat(max_pool,dim=1)
        y = self.conv2(y)
        return y
'''利用上面的模块写主网络'''
class MainNet(nn.Module):
    '''
    na是锚框anchor数量，nc是分类数量，depth_multiple是深度系数，width_multiple是宽度系数。
    yolov5s:
    depth_multiple: 0.33  # model depth multiple
    width_multiple: 0.50  # layer channel multiple
    yolov5m:
    depth_multiple: 0.67
    width_multiple: 0.75
    yolov5l:
    depth_multiple: 1.0
    width_multiple: 1.0
    yolov5x:
    depth_multiple: 1.33
    width_multiple: 1.25
    '''
    def __init__(self,na=3,nc=10,depth_multiple=0.33,width_multiple=0.5):#默认是yolov5s
        super(MainNet, self).__init__()
        c_init= math.ceil(64*width_multiple/8)*8
        # print('c_init',c_init) 32
        #backbone
        self.focus = Focus(12, c_init)
        self.conv1 = Conv(c_init, c_init * 2, 3, 2, 1)
        self.bottleneckCSP1 = Bottleneckcsp(c_init * 2, c_init * 2, n=3, depth_multiple=depth_multiple)
        self.conv2 = Conv(c_init * 2, c_init * 4, 3, 2, 1)
        self.bottleneckCSP2 = Bottleneckcsp(c_init * 4, c_init * 4, n=9, depth_multiple=depth_multiple)
        self.conv3 = Conv(c_init * 4, c_init * 8, 3, 2, 1)  # 128,256
        self.bottleneckCSP3 = Bottleneckcsp(c_init * 8, c_init * 8, n=9, depth_multiple=depth_multiple) #256,256
        # 此处输出给concat_fpn_1
        self.conv4 = Conv(c_init * 8, c_init * 16, 3, 2, 1)  # 256,512
        self.spp = SPP(c_init * 16,c_init * 16) #512,512
        self.bottleneckCSP4 = Bottleneckcsp(c_init * 16, c_init * 16, n=3, depth_multiple=depth_multiple) #512,512
        #输出给 conv_fpn_1

        #neck FPN
        self.conv_fpn_1 = Conv(c_init * 16,c_init * 8,1,1) #512,256
        #self.conv_fpn_1 = Channel_Selective_Conv(c_init * 16,c_init * 8)
        # 此处输出给 conv_pan_2合并
        #此处上采样在forward中体现
        #此处concat_fpn_1 #512
        self.bottleneck_fpn_1_3 = Bottleneckcsp(c_init * 16, c_init * 8, n=3, depth_multiple=depth_multiple) #512,256
        self.conv_fpn_2 = Conv(c_init * 8,c_init * 4,1,1)#256,128
        #self.conv_fpn_2 = Channel_Selective_Conv(c_init * 8,c_init * 4)
        #此处输出给 conv_pan_1合并
        #此处上采样
        #此处concat_fpn_2 #256
        #此处concat_fpn_2合并后，输出给 bottleneck_pan_1_3

        #neck pan
        self.bottleneck_pan_1_3 = Bottleneckcsp(c_init * 8, c_init * 4, n=3, depth_multiple=depth_multiple) #256,128
        #输出给 head_1
        self.conv_pan_1 = Conv(c_init * 4,c_init * 4,3,2,1) #128,128
        #此处concat_pan_1 #256
        self.bottleneck_pan_2_3 = Bottleneckcsp(c_init * 8, c_init * 8, n=3, depth_multiple=depth_multiple) #256,256
        #此处输出给head_2
        self.conv_pan_2 = Conv(c_init * 8,c_init * 8,3,2,1) #256,256
        #此处concat_pan_2 #512
        self.bottleneck_pan_3_3 = Bottleneckcsp(c_init * 16, c_init * 16, n=3, depth_multiple=depth_multiple) #512,512
        #此处输出给head_3

        #head
        self.head_1 = nn.Conv2d(c_init * 4,na*(nc+5),1,1)
        self.head_2 = nn.Conv2d(c_init * 8, na * (nc+ 5), 1, 1)
        self.head_3 = nn.Conv2d(c_init * 16, na * (nc + 5), 1, 1)
    def forward(self,x):
        #backbone
        focus = self.focus(x) #此处输出给concat_fpn_2
        conv1 = self.conv1(focus)
        csp1 = self.bottleneckCSP1(conv1)
        conv2 = self.conv2(csp1)
        csp2 = self.bottleneckCSP2(conv2)
        conv3 = self.conv3(csp2)
        bottleneck_3_9 = self.bottleneckCSP3(conv3) #此处输出给concat_fpn_1
        conv4 = self.conv4(bottleneck_3_9)
        spp = self.spp(conv4)
        bottleneck_4_3 = self.bottleneckCSP4(spp) #输出给 conv_fpn_1
        #neck fpn
        conv_fpn_1 = self.conv_fpn_1(bottleneck_4_3) #此处输出给 conv_pan_2合并
        up_fpn_1 = F.interpolate(conv_fpn_1, scale_factor=2, mode='nearest')
        concat_fpn_1 = torch.cat((up_fpn_1,bottleneck_3_9),dim=1)
        bottleneck_fpn_1_3 = self.bottleneck_fpn_1_3(concat_fpn_1)
        conv_fpn_2 = self.conv_fpn_2(bottleneck_fpn_1_3) #此处输出给 conv_pan_1合并
        up_fpn_2 = F.interpolate(conv_fpn_2, scale_factor=2, mode='nearest')
        concat_fpn_2 = torch.cat((up_fpn_2,csp2),dim=1) #输出给bottleneck_pan_1_3
        #neck pan
        bottleneck_pan_1_3 = self.bottleneck_pan_1_3(concat_fpn_2) #输出给head_1
        conv_pan_1 = self.conv_pan_1(bottleneck_pan_1_3)
        concat_pan_1 = torch.cat((conv_pan_1,conv_fpn_2),dim=1)
        bottleneck_pan_2_3 = self.bottleneck_pan_2_3(concat_pan_1) #输出给head_2
        conv_pan_2 = self.conv_pan_2(bottleneck_pan_2_3)
        concat_pan_2 = torch.cat((conv_pan_2,conv_fpn_1),dim=1)
        bottleneck_pan_3_3 = self.bottleneck_pan_3_3(concat_pan_2) #输出给head_3
        #head
        head_1 = self.head_1(bottleneck_pan_1_3)
        head_2 = self.head_2(bottleneck_pan_2_3)
        head_3 = self.head_3(bottleneck_pan_3_3)

        return head_3,head_2,head_1

class Yolov5(nn.Module):
    def __init__(self, type='l'):
        super(Yolov5, self).__init__()
        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__out_channel = cfg.MODEL["ANCHORS_PER_SCLAE"] * (self.__nC + 5)
        '''
        na是锚框anchor数量，nc是分类数量，depth_multiple是深度系数，width_multiple是宽度系数。
        yolov5s:
        depth_multiple: 0.33  # model depth multiple
        width_multiple: 0.50  # layer channel multiple
        yolov5m:
        depth_multiple: 0.67
        width_multiple: 0.75
        yolov5l:
        depth_multiple: 1.0
        width_multiple: 1.0
        yolov5x:
        depth_multiple: 1.33
        width_multiple: 1.25
        '''
        self.type = type
        self.depth_m = 0.33
        self.width_m = 0.5
        if type == 's':
            self.depth_m = 0.33
            self.width_m = 0.5
        elif type == 'm':
            self.depth_m = 0.67
            self.width_m = 0.75
        elif type == 'l':
            self.depth_m = 1.0
            self.width_m = 1.0
        elif type == 'x':
            self.depth_m = 1.33
            self.width_m = 1.25
        self.__backnone = MainNet(na=3,nc=20,depth_multiple=self.depth_m,width_multiple=self.width_m)
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

#from thop import profile
import torchsummary as summary

if __name__ == '__main__':
    from thop import profile
    # print(input.shape
    # net = MainNet(depth_multiple=1,width_multiple=1)
    net = Yolov5('l')
    summary.summary(net, (3, 416, 416), 1, 'cpu')
    para = torch.randn(1, 3, 416, 416)
    flops, params = profile(net, (para,))
    print(flops)
    print(params)
#    head_1,head_2,head_3 = net(input)
#    print(head_1.shape,head_2.shape,head_3.shape)
    # print(net)









