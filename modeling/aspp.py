import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.spatial_path import *

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        elif backbone == 'resnet-18':
            inplanes = 512
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 3, 12, 18]#[1, 3, 12, 18]
            # dilations = [1, 3, 3, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())

        self.attention_refinement_module = AttentionRefinementModule(inplanes, inplanes)
        self.attention_conv = nn.Conv2d(inplanes, 256, 1, stride=1, bias=False)
        self.attention_bn = BatchNorm(256)
        self.attention_relu = nn.ReLU()

        # self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.conv1 = nn.Conv2d(256*3, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        # x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        # x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x5 = self.attention_refinement_module(x)
        x5 = self.attention_conv(x5)
        x5 = self.attention_bn(x5)
        x5 = self.attention_relu(x5)


        # x5 = self.global_avg_pool(x)
        # print("x5 size = ", x5.size())
        
        # x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        # print("x5 size 2 = ", x5.size())
        # exit()
        x = torch.cat((x2, x4, x5), dim=1)
        # x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

# class _ASPPModule(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
#         super(_ASPPModule, self).__init__()
#         self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
#                                             stride=1, padding=padding, dilation=dilation, bias=False)
#         self.bn = BatchNorm(planes)
#         self.relu = nn.ReLU()

#         self._init_weight()

#     def forward(self, x):
#         x = self.atrous_conv(x)
#         x = self.bn(x)

#         return self.relu(x)

#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, SynchronizedBatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

# class ASPP(nn.Module):
#     def __init__(self, backbone, output_stride, BatchNorm):
#         super(ASPP, self).__init__()
#         if backbone == 'drn':
#             inplanes = 512
#         elif backbone == 'mobilenet':
#             inplanes = 320
#         else:
#             inplanes = 2048
#         if output_stride == 16:
#             dilations = [1, 6, 12, 18]
#         elif output_stride == 8:
#             dilations = [1, 12, 24, 36]
#         else:
#             raise NotImplementedError

#         self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
#         self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
#         self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
#         self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

#         self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
#                                              nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
#                                              BatchNorm(256),
#                                              nn.ReLU())
#         self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
#         # self.conv1 = nn.Conv2d(256*4, 256, 1, bias=False)
#         self.bn1 = BatchNorm(256)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self._init_weight()

#     def forward(self, x):
#         x1 = self.aspp1(x)
#         x2 = self.aspp2(x)
#         x3 = self.aspp3(x)
#         x4 = self.aspp4(x)
#         x5 = self.global_avg_pool(x)
#         x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
#         x = torch.cat((x1, x2, x3, x4, x5), dim=1)
#         # x = torch.cat((x1, x2, x3, x5), dim=1)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         return self.dropout(x), self.dropout(x1)

#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 # m.weight.data.normal_(0, math.sqrt(2. / n))
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, SynchronizedBatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()


# def build_aspp(backbone, output_stride, BatchNorm):
#     return ASPP(backbone, output_stride, BatchNorm)