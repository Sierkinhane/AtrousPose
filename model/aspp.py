# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)

class ASPP_ASP(nn.Module):
    def __init__(self, in_, out_=16):
        super(ASPP_ASP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(in_, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(128)

        self.conv_3x3_1 = nn.Conv2d(in_, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(128)

        self.conv_3x3_2 = nn.Conv2d(in_, 128, kernel_size=3, stride=1, padding=8, dilation=8)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(128)

        self.conv_3x3_3 = nn.Conv2d(in_, 128, kernel_size=3, stride=1, padding=16, dilation=16)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(128)

        self.bn_out = nn.BatchNorm2d(512)

    def forward(self, feature_map):

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        add1 = out_1x1
        add2 = add1+out_3x3_1
        add3 = add2+out_3x3_2
        add4 = add3+out_3x3_3
        out = F.relu(self.bn_out(torch.cat([add1, add2, add3, add4], 1))) # (shape: (batch_size, 1280, h/16, w/16))

        return out
