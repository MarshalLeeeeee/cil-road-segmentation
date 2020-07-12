import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d

from config import config
from resnet import resnet101
from util import AttentionRefinement, FeatureFusion

class Backbone_Res101(nn.Module):
    def __init__(self, out_planes):
        super(Backbone_Res101, self).__init__()
        self.layers = []
       
        conv_channel = 128
        # use base model of resnet 101 from resnet.py
        self.context = resnet101(norm_layer=BatchNorm2d, bn_eps=config.bn_eps, bn_momentum=config.bn_momentum)
        self.context_refine = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        ConvBnRelu(2048,conv_channel, 1, 1, 0, has_bn=True, has_relu=True, has_bias=False, norm_layer=BatchNorm2d)
                )
        
        # ARM for ResBlock 2,3,4 of resnet output
        arms = [AttentionRefinement(2048, conv_channel, norm_layer=BatchNorm2d),
                AttentionRefinement(1024, conv_channel, norm_layer=BatchNorm2d),
                AttentionRefinement(512, conv_channel, norm_layer=BatchNorm2d)]
        self.arms = nn.ModuleList(arms)

         # Refinement of corresponding output
        refines = [ConvBnRelu(conv_channel, conv_channel, 3, 1, 1, has_bn=True, norm_layer=BatchNorm2d, has_relu=True, has_bias=False),
                   ConvBnRelu(conv_channel, conv_channel, 3, 1, 1, has_bn=True, norm_layer=BatchNorm2d, has_relu=True, has_bias=False),
                   ConvBnRelu(conv_channel, conv_channel, 3, 1, 1, has_bn=True, norm_layer=BatchNorm2d, has_relu=True, has_bias=False)]
        self.refines = nn.ModuleList(refines)

        # Refinement on first layer of resnet output
        self.res_top_refine = ConvBnRelu(256, conv_channel, 3, 1, 1, has_bn=True, norm_layer=BatchNorm2d, has_relu=True, has_bias=False)
        self.ffm = FeatureFusion(conv_channel*2, conv_channel, 1, BatchNorm2d)
        
        # classifier for final output
        self.class_refine = nn.Sequential(
                ConvBnRelu(conv_channel, conv_channel//2, 3, 1, 1, has_bn=True, has_relu=True, has_bias=False, norm_layer=BatchNorm2d),
                nn.Conv2d(conv_channel//2, out_planes, kernel_size=1, stride=1, padding=0)
            )

        self.layers.append(self.context)
        self.layers.append(self.class_refine)
        self.layers.append(self.context_refine)
        self.layers.append(self.arms)
        self.layers.append(self.ffm)
        self.layers.append(self.refines)
        self.layers.append(self.res_top_refine)

        self.loss = nn.CrossEntropyLoss(reduction='mean',ignore_index=255)
        
    def forward(self, x):
        context_out = self.context(x)
        context_out.reverse()

        bot_context = self.context_refine(context_out[0])
        bot_context = F.interpolate(bot_context,size=context_out[0].size()[2:],
                                    mode='bilinear', align_corners=True)

        last = bot_context

        for i, (fm,arm,refine) in enumerate(zip(context_out[:3],self.arms,self.refines)):
            fm = arm(fm)
            fm += last
            last = F.interpolate(fm, size=(context_out[i+1].size()[2:]),mode='bilinear',align_corners=True)
            last = refine(last)
        
        res_top = self.res_top_refine(context_out[3])
        res_combine = self.ffm(res_top, last)       
        
        result = self.class_refine(res_combine)

        # upsampling to original size of image
        result = F.interpolate(result, scale_factor=4, mode='bilinear', align_corners=True)

        return result
    
class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5, 
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)
            
    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
            
        return x






