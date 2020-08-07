import numpy as np
import torch
from torch import nn

from cil.modules.blocks import DoubleConvBlock
from cil.utilities.utils import OutputType


class UNetEncoder(nn.Module):
    """ Shrink feature map with max pooling, then go through two size-keeping convolution block. """
    def __init__(self, in_channels, out_channels, bn_eps=1e-5, bn_momentum=0.1):
        super().__init__()

        self.max_pool = nn.MaxPool2d(2)
        self.double_conv = DoubleConvBlock(in_channels, out_channels, bn_eps=bn_eps, bn_momentum=bn_momentum)

    def forward(self, x):
        out = self.max_pool(x)
        out = self.double_conv(out)

        return out


class UNetDecoder(nn.Module):
    """ Enlarge feature map with interpolation or deconvolution, then go through two size-keeping convolution block. """
    def __init__(self, in_channels, out_channels, bilinear=True, bn_eps=1e-5, bn_momentum=0.1):
        super().__init__()

        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.double_conv = DoubleConvBlock(in_channels, out_channels, reduction=2,
                                               bn_eps=bn_eps, bn_momentum=bn_momentum)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.double_conv = DoubleConvBlock(in_channels, out_channels, bn_eps=bn_eps, bn_momentum=bn_momentum)

    def forward(self, x1, x2):
        """ x2 decides the side of output, and it should be greater than up-sampled x1. """
        x1 = self.upsample(x1)

        # Pad to match shapes
        diff_h, diff_w = np.subtract(x2.shape[2:], x1.shape[2:])
        x1 = nn.functional.pad(x1, pad=[diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        out = torch.cat([x2, x1], dim=1)
        out = self.double_conv(out)

        return out


class UNet(nn.Module):
    output_type = OutputType.LOGIT

    def __init__(self, in_channels, out_channels, bilinear=True, bn_eps=1e-5, bn_momentum=0.1):
        super(UNet, self).__init__()
        reduction = 2 if bilinear else 1

        self.encoder0 = DoubleConvBlock(in_channels, 64, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.encoder1 = UNetEncoder(64, 128, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.encoder2 = UNetEncoder(128, 256, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.encoder3 = UNetEncoder(256, 512, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.encoder4 = UNetEncoder(512, 1024 // reduction, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.decoder4 = UNetDecoder(1024, 512 // reduction, bilinear, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.decoder3 = UNetDecoder(512, 256 // reduction, bilinear, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.decoder2 = UNetDecoder(256, 128 // reduction, bilinear, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.decoder1 = UNetDecoder(128, 64, bilinear, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.decoder0 = DoubleConvBlock(64, out_channels, bn_eps=bn_eps, bn_momentum=bn_momentum)

    def forward(self, x):
        encoded0 = self.encoder0(x)
        encoded1 = self.encoder1(encoded0)
        encoded2 = self.encoder2(encoded1)
        encoded3 = self.encoder3(encoded2)
        encoded4 = self.encoder4(encoded3)
        decoded4 = self.decoder4(encoded4, encoded3)
        decoded3 = self.decoder3(decoded4, encoded2)
        decoded2 = self.decoder2(decoded3, encoded1)
        decoded1 = self.decoder1(decoded2, encoded0)
        out = self.decoder0(decoded1)

        return out
