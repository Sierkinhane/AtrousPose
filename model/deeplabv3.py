# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from model.resnet import ResNet50_OS16
from model.aspp import ASPP_ASP
def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)

def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)

class AtrousPose(nn.Module):
    def __init__(self):
        super(AtrousPose, self).__init__()
        """
        mobile net
        """
        self.resnet = ResNet50_OS16()
        self.smooth_ups2 = self._lateral(1024, 2)
        self.smooth_ups3 = self._lateral(512, 1)
        self.aspp1 = ASPP_ASP(512, out_=16)
        self.h1 = nn.Sequential(
            conv(512, 512, kernel_size=3, padding=1,),
            conv(512, 512, kernel_size=3, padding=1,),
            conv(512, 512, kernel_size=3, padding=1,),
            conv(512, 512, kernel_size=1, padding=0, bn=False),
            conv(512, 16, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.p1 = nn.Sequential(
            conv(512, 512, kernel_size=3, padding=1,),
            conv(512, 512, kernel_size=3, padding=1,),
            conv(512, 512, kernel_size=3, padding=1,),
            conv(512, 512, kernel_size=1, padding=0, bn=False),
            conv(512, 32, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def _lateral(self, input_size, factor):
        layers = []
        layers.append(nn.Conv2d(input_size, 256, kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(Upsample(scale_factor=factor, mode='bilinear'))

        return nn.Sequential(*layers)
    def forward(self, x):
        # import time
        # s = time.time()
        feature_map, _16x = self.resnet(x)
        # _32x = self.smooth_ups1(_32x)
        _16x = self.smooth_ups2(_16x)
        feature_map = self.smooth_ups3(feature_map)
        cat_feat = F.relu(torch.cat([feature_map, _16x], 1))

        out = self.aspp1(cat_feat)
        heat1 = self.h1(out)
        paf1 = self.p1(out)

        # e = time.time()
        return paf1, heat1

def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))

if __name__ == "__main__":
    import cv2
    net = AtrousPose().cuda()
    image = cv2.imread('star.jpg')
    image = cv2.resize(image, (256, 256))
    image = torch.from_numpy(image).type(torch.FloatTensor).permute(2, 0, 1).reshape(1, 3, 256, 256).cuda()
    print(net)
    model_info(net)
    for i in range(30):
        vec1, heat1 = net(image)

    print(vec1.shape, heat1.shape)
