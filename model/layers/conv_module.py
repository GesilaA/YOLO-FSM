import torch
import torch.nn as nn
import torch.nn.functional as F
from .activate import *

norm_name = {"bn": nn.BatchNorm2d}
activate_name = {
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "trelu": TReLU,
    "mish": Mish,
    "swish": Swish,
    "sigmoid": nn.Sigmoid,
}


class Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride, pad, norm=None, activate=None, groups=1):
        super(Convolutional, self).__init__()

        self.norm = norm
        self.activate = activate

        self.__conv = nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size,
                                stride=stride, padding=pad, bias=not norm, groups=groups)
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)

        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            elif activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            else:
                self.__activate = activate_name[activate]()



    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)

        return x

class Copy_Convolutional(nn.Module):
    def __init__(self, stride, copy_c):
        super(Copy_Convolutional, self).__init__()
        norm = copy_c.norm
        self.norm = norm
        activate = copy_c.activate
        self.activate = activate

        self.__conv = nn.Conv2d(in_channels=copy_c.filters_in, out_channels=copy_c.filters_out,
                                kernel_size=copy_c.kernel_size, stride=stride, padding=copy_c.pad, bias=not norm)
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=copy_c.filters_out)

        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)

    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)

        return x