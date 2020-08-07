import torch
from torch import nn

from cil.modules.blocks import DoubleConvBlock, DilatedBlock
from cil.modules.unet import UNetEncoder, UNetDecoder
from cil.utilities.utils import OutputType


class UNetSeg(nn.Module):
    """ A segmentation network which effectively make use of context information """
    output_type = OutputType.LOGIT

    def __init__(self, in_channels=3, out_channels=2, bilinear=True, bn_eps=1e-5, bn_momentum=0.1):
        super(UNetSeg, self).__init__()
        reduction = 2 if bilinear else 1

        # Encoders
        self.encoder0 = DoubleConvBlock(in_channels, 64, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.encoder1 = UNetEncoder(64, 128, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.encoder2 = UNetEncoder(128, 256, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.encoder3 = UNetEncoder(256, 512, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.encoder4 = UNetEncoder(512, 1024 // reduction, bn_eps=bn_eps, bn_momentum=bn_momentum)

        # Dilation: only at the bottom level
        self.refiner = DilatedBlock(1024 // reduction, 1024 // reduction, bn_eps=bn_eps, bn_momentum=bn_momentum)

        # Decoders
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

        feature = self.refiner(encoded4)

        decoded4 = self.decoder4(feature, encoded3)
        decoded3 = self.decoder3(decoded4, encoded2)
        decoded2 = self.decoder2(decoded3, encoded1)
        decoded1 = self.decoder1(decoded2, encoded0)
        out = self.decoder0(decoded1)

        return out


def _test():
    model = UNetSeg()

    for name, param in model.named_parameters():
        print(name, param.shape)


if __name__ == '__main__':
    _test()
