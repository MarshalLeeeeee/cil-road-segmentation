import torch
import torch.nn as nn


class conv3x3(nn.Module):
    """3x3 convolution"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(conv3x3, self).__init__()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1)

    def forward(self, x):
        return self.c(x)


class conv1x1(nn.Module):
    """1x1 convolution"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(conv1x1, self).__init__()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        return self.c(x)


class BasicConvBlock(nn.Module):
    """ A standard conv-bn-relu block. """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bn_eps=1e-5, bn_momentum=0.1):
        super(BasicConvBlock, self).__init__()
        # has_bn = True, has_relu = True, has_bias = False, norm_layer = nn.BatchNorm2d)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


class DoubleConvBlock(nn.Module):
    """ Two standard convolution blocks. The shape of output differs from input only in channel dimension """
    def __init__(self, in_channels, out_channels, reduction=1, bn_eps=1e-5, bn_momentum=0.1):
        super(DoubleConvBlock, self).__init__()
        mid_channels = in_channels // reduction

        self.conv1 = conv3x3(in_channels, mid_channels)
        self.bn1 = nn.BatchNorm2d(mid_channels, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class DilatedBlock(nn.Module):
    """ An attentioned convolution block which has larger reception fields provided by dilated convolution """
    def __init__(self, in_channels, out_channels, bn_eps=1e-5, bn_momentum=0.1):
        super(DilatedBlock, self).__init__()

        assert in_channels % 8 == 0
        channels = [in_channels // 2, in_channels // 4, in_channels // 8, in_channels // 8]

        self.dilate1 = nn.Conv2d(in_channels, channels[0], kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(in_channels, channels[1], kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(in_channels, channels[2], kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(in_channels, channels[3], kernel_size=3, dilation=6, padding=6)
        self.bn = nn.BatchNorm2d(in_channels, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        self.attention = AttentionBlock(in_channels, in_channels)
        self.conv_block = BasicConvBlock(2 * in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # Concatenated dilated convolution
        dilated1 = self.dilate1(x)
        dilated2 = self.dilate2(x)
        dilated3 = self.dilate3(x)
        dilated4 = self.dilate4(x)
        out = torch.cat([dilated1, dilated2, dilated3, dilated4], dim=1)
        out = self.bn(out)
        out = self.relu(out)

        # Adjust with attention
        out = self.attention(out)

        # Allow "skip" connections, and do final processing
        out = torch.cat([x, out], dim=1)
        out = self.conv_block(out)

        return out


class AttentionBlock(nn.Module):
    """ A convolution block with local(channel) attention. HW of output is the same as input.  """
    def __init__(self, in_channels, out_channels, bn_eps=1e-5, bn_momentum=0.1):
        super(AttentionBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv2 = conv1x1(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # local attention: weight by channels
        channel_weights = self.avg_pool(out)
        channel_weights = self.conv2(channel_weights)
        channel_weights = self.bn2(channel_weights)
        channel_weights = self.sigmoid(channel_weights)

        out = out * channel_weights

        return out


class FeatureFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, bn_eps=1e-5, bn_momentum=0.1):
        super(FeatureFusionBlock, self).__init__()

        # diff 1: use 1x1 conv at first
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # a bottleneck structure
        # todo what is the intuition? not applied
        bottleneck_channels = out_channels // reduction
        self.conv2 = conv1x1(out_channels, bottleneck_channels)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels, eps=bn_eps, momentum=bn_momentum)

        self.conv3 = conv1x1(bottleneck_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        """
        fuse two feature map which only differs in number of channels. c1 + c2 == in_channels

        :param x1: feature map of shape (n * c1 * h * w)
        :param x2: feature map of shape (n * c2 * h * w)
        :return: fused feature map
        """

        x = torch.cat([x1, x2], dim=1)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # local attention: weight by channels
        channel_weights = self.avg_pool(out)

        channel_weights = self.conv2(channel_weights)
        channel_weights = self.bn2(channel_weights)
        channel_weights = self.relu(channel_weights)

        channel_weights = self.conv3(channel_weights)
        channel_weights = self.bn3(channel_weights)
        channel_weights = self.sigmoid(channel_weights)

        # todo what is the intuition of the plus
        out = out + out * channel_weights

        return out


class SeBlock(nn.Module):

    def __init__(self, channel, se_reduction=16, bias=False):
        super(SeBlock, self).__init__()
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(channel, channel // se_reduction),
            nn.ReLU(inplace=True),
            conv1x1(channel // se_reduction, channel),
            nn.Sigmoid()
            )
        self.sse = nn.Sequential(
            conv1x1(channel, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)