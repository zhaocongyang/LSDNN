import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import torch.utils.model_zoo as model_zoo

def conv_bn(inp, oup, stride, BatchNorm):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm(oup),
        nn.ReLU6(inplace=True)
    )


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio, BatchNorm):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.kernel_size = 3
        self.dilation = dilation

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, 1, 1, bias=False),
                BatchNorm(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, 1, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, 1, bias=False),
                BatchNorm(oup),
            )

    def forward(self, x):
        x_pad = fixed_padding(x, self.kernel_size, dilation=self.dilation)
        if self.use_res_connect:
            x = x + self.conv(x_pad)
        else:
            x = self.conv(x_pad)
        return x


class MobileNetV2(nn.Module):
    def __init__(self, output_stride=8, BatchNorm=None, width_mult=1., pretrained=True):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        current_stride = 1
        rate = 1
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.features = [conv_bn(3, input_channel, 2, BatchNorm)]
        current_stride *= 2
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            if current_stride == output_stride:
                stride = 1
                dilation = rate
                rate *= s
            else:
                stride = s
                dilation = 1
                current_stride *= s
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, stride, dilation, t, BatchNorm))
                else:
                    self.features.append(block(input_channel, output_channel, 1, dilation, t, BatchNorm))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        self._initialize_weights()

        if pretrained:
            self._load_pretrained_model()

        self.low_level_features = self.features[0:4]
        self.high_level_features = self.features[4:]

    def forward(self, x):
        low_level_feat = self.low_level_features(x)
        x = self.high_level_features(low_level_feat)
        return x, low_level_feat

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('http://jeff95.me/models/mobilenet_v2-6a65762b.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _initialize_weights(self):
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

if __name__ == "__main__":
    input = torch.rand(1, 3, 512, 512)
    model = MobileNetV2(output_stride=16, BatchNorm=nn.BatchNorm2d)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())

# '''MobileNetV3 in PyTorch.

# See the paper "Inverted Residuals and Linear Bottlenecks:
# Mobile Networks for Classification, Detection and Segmentation" for more details.
# '''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import init



# class hswish(nn.Module):
#     def forward(self, x):
#         out = x * F.relu6(x + 3, inplace=True) / 6
#         return out


# class hsigmoid(nn.Module):
#     def forward(self, x):
#         out = F.relu6(x + 3, inplace=True) / 6
#         return out


# class SeModule(nn.Module):
#     def __init__(self, in_size, reduction=4):
#         super(SeModule, self).__init__()
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(in_size // reduction),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(in_size),
#             hsigmoid()
#         )

#     def forward(self, x):
#         return x * self.se(x)


# class Block(nn.Module):
#     '''expand + depthwise + pointwise'''
#     def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
#         super(Block, self).__init__()
#         self.stride = stride
#         self.se = semodule

#         self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(expand_size)
#         self.nolinear1 = nolinear
#         self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
#         self.bn2 = nn.BatchNorm2d(expand_size)
#         self.nolinear2 = nolinear
#         self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_size)

#         self.shortcut = nn.Sequential()
#         if stride == 1 and in_size != out_size:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_size),
#             )

#     def forward(self, x):
#         out = self.nolinear1(self.bn1(self.conv1(x)))
#         out = self.nolinear2(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         if self.se != None:
#             out = self.se(out)
#         out = out + self.shortcut(x) if self.stride==1 else out
#         return out


# class MobileNetV3_Large(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(MobileNetV3_Large, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.hs1 = hswish()

#         self.bneck = nn.Sequential(
#             Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
#             Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
#             Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
#             Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
#             Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
#             Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
#             Block(3, 40, 240, 80, hswish(), None, 2),
#             Block(3, 80, 200, 80, hswish(), None, 1),
#             Block(3, 80, 184, 80, hswish(), None, 1),
#             Block(3, 80, 184, 80, hswish(), None, 1),
#             Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
#             Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
#             Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
#             Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
#             Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
#         )


#         self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn2 = nn.BatchNorm2d(960)
#         self.hs2 = hswish()
#         self.linear3 = nn.Linear(960, 1280)
#         self.bn3 = nn.BatchNorm1d(1280)
#         self.hs3 = hswish()
#         self.linear4 = nn.Linear(1280, num_classes)
#         self.init_params()

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def forward(self, x):
#         out = self.hs1(self.bn1(self.conv1(x)))
#         out = self.bneck(out)
#         out = self.hs2(self.bn2(self.conv2(out)))
#         out = F.avg_pool2d(out, 7)
#         out = out.view(out.size(0), -1)
#         out = self.hs3(self.bn3(self.linear3(out)))
#         out = self.linear4(out)
#         return out



# class MobileNetV3_Small(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(MobileNetV3_Small, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.hs1 = hswish()

#         self.bneck = nn.Sequential(
#             Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
#             Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
#             Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
#             Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
#             Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
#             Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
#             Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
#             Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
#             Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
#             Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
#             Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
#         )


#         self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn2 = nn.BatchNorm2d(576)
#         self.hs2 = hswish()
#         self.linear3 = nn.Linear(576, 1280)
#         self.bn3 = nn.BatchNorm1d(1280)
#         self.hs3 = hswish()
#         self.linear4 = nn.Linear(1280, num_classes)
#         self.init_params()

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def forward(self, x):
#         out = self.hs1(self.bn1(self.conv1(x)))
#         out = self.bneck(out)
#         out = self.hs2(self.bn2(self.conv2(out)))
#         out = F.avg_pool2d(out, 7)
#         out = out.view(out.size(0), -1)
#         out = self.hs3(self.bn3(self.linear3(out)))
#         out = self.linear4(out)
#         return out



# def test():
#     net = MobileNetV3_Small()
#     x = torch.randn(2,3,224,224)
#     y = net(x)
#     print(y.size())

# # test()
