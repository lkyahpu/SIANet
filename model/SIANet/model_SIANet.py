import math

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.SIANet.SCTransNet import SCTransNet, get_CTranS_config


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff1 = self.conv.weight.sum(2).sum(2)
            kernel_diff2 = kernel_diff1[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff2, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)
            return out_normal - self.theta * out_diff

class OneConv_cdc(nn.Module):
    def __init__(self, G0, G):
        super(OneConv_cdc, self).__init__()
        self.conv = Conv2d_cd(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        buffer = self.conv(x)
        output = self.relu(buffer)
        return torch.cat((x, output), dim=1)

class SIANet(nn.Module):
    def __init__(self, nf=16, block=Res_CBAM_block):
        super(SIANet, self).__init__()
        self.static_path = block(nf, nf*2)
        config_vit = get_CTranS_config()
        self.unet = SCTransNet(config_vit, nf)
        self.init_conv1 = OneConv_cdc(1, nf-1)

    def forward(self, x):
        b, c0, h, w = x.shape
        x = self.init_conv1(x)
        # out = self.static_path(x)
        out = self.unet(x)

        return out