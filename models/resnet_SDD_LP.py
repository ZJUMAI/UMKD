import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
class SPP(nn.Module):
    def __init__(self, M=None):
        super(SPP, self).__init__()
        self.pooling_4x4 = nn.AdaptiveAvgPool2d((4, 4))
        self.pooling_2x2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pooling_1x1 = nn.AdaptiveAvgPool2d((1, 1))

        self.M = M
        print(self.M)

    def forward(self, x):
        x_4x4 = self.pooling_4x4(x)
        x_2x2 = self.pooling_2x2(x_4x4)
        x_1x1 = self.pooling_1x1(x_4x4)

        x_4x4_flatten = torch.flatten(x_4x4, start_dim=2, end_dim=3)  # B X C X feature_num

        x_2x2_flatten = torch.flatten(x_2x2, start_dim=2, end_dim=3)

        x_1x1_flatten = torch.flatten(x_1x1, start_dim=2, end_dim=3)

        if self.M == '[1,2,4]':
            x_feature = torch.cat((x_1x1_flatten, x_2x2_flatten, x_4x4_flatten), dim=2)
        elif self.M == '[1,2]':
            x_feature = torch.cat((x_1x1_flatten, x_2x2_flatten), dim=2)
        else:
            raise NotImplementedError('ERROR M')

        x_strength = x_feature.permute((2, 0, 1))
        x_strength = torch.mean(x_strength, dim=2)


        return x_feature, x_strength


# 通过低通滤波器的方式进行特征提取
class LowPassModule(nn.Module):
    def __init__(self, in_channel, sizes=(1, 2, 3, 6)):
        # 低通滤波器的不同尺寸，可以通过不同的滤波器尺寸进行处理，大小分别为1x1，2x2，3x3和6x6。
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        # 为每种大小创建一个滤波器（使用 nn.AdaptiveAvgPool2d 进行池化），以适应输入特征图的大小。
        self.relu = nn.ReLU()
        ch =  in_channel // 4
        self.channel_splits = [ch, ch, ch, ch]
        # 将输入特征图的通道分为四部分，每个部分的通道数是 in_channel // 4
    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3) # 获取输入特征图的高度和宽度
        feats = torch.split(feats, self.channel_splits, dim=1) # 将输入的 feats 按照 self.channel_splits 划分成多个部分
        priors = [F.upsample(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear') for i in range(4)] # 对每个部分应用 stages[i]，即进行自适应池化后，再通过双线性插值将其上采样至输入图像的原始大小。
        bottle = torch.cat(priors, 1) # 将所有上采样后的部分沿着通道维度拼接。
        return self.relu(bottle)

# 卷积&归一化层
class Conv2d_BN(nn.Module):
    """Convolution with BN module."""

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            pad=0,
            dilation=1,
            groups=1,
            bn_weight_init=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    stride,
                                    pad,
                                    dilation,
                                    groups,
                                    bias=False)
        self.bn = norm_layer(out_ch) # 批归一化层
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity(
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x

# 深度可分离卷积
class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        #   depthwise and pointwise convolution, downsample by 2
        # groups=channel_in：设置 groups=channel_in 使得每个输入通道与一个独立的卷积核进行卷积（深度卷积），每个通道都有自己的卷积核，不进行通道之间的混合。
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            # 执行深度卷积（Depthwise Convolution） 设置 groups=channel_in 使得每个输入通道与一个独立的卷积核进行卷积（深度卷积），每个通道都有自己的卷积核，不进行通道之间的混合。
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            # 是逐点卷积（Pointwise Convolution）。它使用 1×1 卷积核对每个通道的输出进行卷积，以实现通道之间的混合，改变特征图的通道数。
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            # 批量归一化层（BatchNorm2d）。它的作用是对每个通道进行标准化，
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            # 第三个卷积层，执行深度卷积（Depthwise Convolution）。这个层与第一个卷积层相似，但步幅是 1，没有进行下采样。这样可以保留特征图的空间尺寸。
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            # 第四个卷积层，使用 1×1 的卷积核进行逐点卷积，将通道数从 channel_in 转换到 channel_out，即生成最终的输出通道数。
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

# 针对学生模型的低通滤波器模块
class StuLowPassModule(nn.Module):
    def __init__(self, in_channel, kernel_size=3, stride = 1, sizes=(1, 2, 3, 6)):
        super().__init__()

        # 低通滤波器的卷积部分，保证和自适应卷积的大小一致
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=1)
        # 低通滤波器部分
        self.low_pass = LowPassModule(in_channel=in_channel, sizes=sizes)
        # 修改后的自适应卷积部分：步幅和卷积核大小为 s x s
        self.strided_conv = nn.Conv2d(
            in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=1
        )
        # 归一化操作：使用 LayerNorm
        self.aggregate = Conv2d_BN(in_channel*2, in_channel, act_layer=nn.Hardswish)
        # 深度分离卷积操作：使用 SepConv
        self.sepconv = SepConv(channel_in=in_channel, channel_out=in_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, feats):
        # print('input shape:', feats.shape)
        # 低通滤波器处理
        feats_1 = self.conv(feats)
        # print("conv1 Shape:", feats_1.shape)
        # 低通滤波器处理
        low_pass_feats = self.low_pass(feats_1)
        # print("low pass feats Shape:", low_pass_feats.shape)
        # 自适应卷积处理
        strided_feats = self.strided_conv(feats)
        # print("Strided Feats Shape:", strided_feats.shape)
        # 进行拼接操作
        fused_feats = torch.cat((low_pass_feats, strided_feats), dim=1)
        # print("Fused Feats Shape:", fused_feats.shape)
        # 聚合加归一化
        out_feats = self.aggregate(fused_feats)
        # print('add & norm_out_feats:', out_feats.shape)
        # 深度可分离卷积
        out_feats = self.sepconv(out_feats)
        # print('sepconv:', out_feats.shape)
        return out_feats


__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            # bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            # bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            print("ResNet unknown block error !!!")

        return [bn2, bn3, bn4]

    def get_stage_channels(self):
        return [256, 512, 1024, 2048]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        stem = x
        x = self.relu(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        feat4 = F.relu(feat4)

        x = self.avgpool(feat4)
        x = x.view(x.size(0), -1)
        avg = x
        out = self.fc(x)

        feats = {}
        feats["pooled_feat"] = avg
        feats["feats"] = [
            F.relu(stem),
            F.relu(feat1),
            F.relu(feat2),
            F.relu(feat3),
            F.relu(feat4),
        ]
        feats["preact_feats"] = [stem, feat1, feat2, feat3, feat4]

        return out, feats


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model




class ResNet_SDD(nn.Module):
    def __init__(self, block, layers, num_classes=1000, M=None):
        self.inplanes = 64
        self.M=M
        super(ResNet_SDD, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])                                                                                
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.stu_low_module_1 = StuLowPassModule(in_channel=64, kernel_size=3, stride=1)  # 使用 3x3 的卷积核和 1x1 步幅     
        # self.stu_low_module_2 = StuLowPassModule(in_channel=128, kernel_size=3, stride=1)  # 使用 3x3 的卷积核和 1x1 步幅 
        # self.stu_low_module_3 = StuLowPassModule(in_channel=256, kernel_size=3, stride=1)  # 使用 3x3 的卷积核和 1x1 步幅
        # self.stu_low_module_4 = StuLowPassModule(in_channel=512, kernel_size=3, stride=1)  # 使用 3x3 的卷积核和 1x1 步幅 

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.spp = SPP(M=self.M)
        self.num_classes = num_classes
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            # bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            # bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            print("ResNet unknown block error !!!")

        return [bn2, bn3, bn4]

    def get_stage_channels(self):
        return [256, 512, 1024, 2048]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        stem = x
        x = self.relu(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)# print('feat1 shape:', feat1.shape) # [8, 256, 56, 56] / [8, 64, 56, 56]
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        low_pass_feats = self.stu_low_module_1(feat1)
        # low_pass_feats = self.stu_low_module_2(feat2)
        # low_pass_feats = self.stu_low_module_3(feat3)
        # low_pass_feats = self.stu_low_module_4(feat4)

        feat4 = F.relu(feat4)# print('feat4 shape:', feat4.shape) # [8, 2048, 7, 7] / [8, 512, 7, 7]
        x_spp, x_strength = self.spp(feat4)# print('x_spp shape:', x_spp.shape) # [8, 2048, 21] / [8, 512, 21]
        # feature_num = x_spp.shape[-1]
        # patch_score = torch.zeros(x_spp.shape[0], self.class_num, feature_num)
        # patch_strength = torch.zeros(x_spp.shape[0], feature_num)
        x_spp = x_spp.permute((2, 0, 1))# print('2x_spp shape:', x_spp.shape) # [21, 8, 2048] / [21, 8, 512]
        m, b, c = x_spp.shape[0], x_spp.shape[1], x_spp.shape[2]# print('m, b, c:', m, b, c) # 21 8 2048 / 21 8 512
        x_spp = torch.reshape(x_spp, (m * b, c))# print('3x_spp shape:', x_spp.shape) # [168, 2048] / [168, 512]
        patch_score = self.fc(x_spp)# print('patch_score shape:', patch_score.shape) # [168, 5] / [168, 5]
        patch_score = torch.reshape(patch_score, (m, b, self.fc.out_features))# print('patch_score shape:', patch_score.shape) # [21, 8, 5] / [21, 8, 5]
        patch_score = patch_score.permute((1, 2, 0)) # [8, 5, 21] / [8, 5, 21]

        x = self.avgpool(feat4)
        x = x.view(x.size(0), -1)
        avg = x
        out = self.fc(x)

        feats = {}
        feats["pooled_feat"] = avg
        feats["feats"] = [
            F.relu(stem),
            F.relu(feat1),
            F.relu(feat2),
            F.relu(feat3),
            F.relu(feat4),
        ]
        feats["preact_feats"] = [stem, feat1, feat2, feat3, feat4]

        return out, patch_score, low_pass_feats


def resnet18_sdd_lp(pretrained=False, num_classes=None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_SDD(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_model = models.resnet18(pretrained=False)  # 加载原始的预训练模型
        pretrained_model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        pretrained_dict = pretrained_model.state_dict()  # 获取预训练权重字典
        model_dict = model.state_dict()  # 获取当前模型的权重字典
        # 过滤掉与当前模型不匹配的层
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新当前模型的权重
        model_dict.update(pretrained_dict)
        # 加载更新后的权重
        model.load_state_dict(model_dict)    
    if num_classes is not None:
        model.fc = nn.Linear(512, num_classes)
    return model


def resnet34_sdd_lp(pretrained=False, num_classes=None, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_SDD(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_model = models.resnet34(pretrained=False)  # 加载原始的预训练模型
        pretrained_model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        pretrained_dict = pretrained_model.state_dict()  # 获取预训练权重字典
        model_dict = model.state_dict()  # 获取当前模型的权重字典
        # 过滤掉与当前模型不匹配的层
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新当前模型的权重
        model_dict.update(pretrained_dict)
        # 加载更新后的权重
        model.load_state_dict(model_dict)    
    if num_classes is not None:
        model.fc = nn.Linear(512, num_classes)
    return model


def resnet50_sdd_lp(pretrained=False, num_classes=None, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_SDD(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_model = models.resnet50(pretrained=False)  # 加载原始的预训练模型
        pretrained_model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        pretrained_dict = pretrained_model.state_dict()  # 获取预训练权重字典
        model_dict = model.state_dict()  # 获取当前模型的权重字典
        # 过滤掉与当前模型不匹配的层
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新当前模型的权重
        model_dict.update(pretrained_dict)
        # 加载更新后的权重
        model.load_state_dict(model_dict)    
    if num_classes is not None:
        model.fc = nn.Linear(2048, num_classes)
    return model


def resnet101_sdd(pretrained=False, num_classes=None, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_SDD(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152_sdd(pretrained=False, num_classes=None, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_SDD(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model


# def multi_dkd(out_s_multi, out_t_multi, target, alpha, beta, temperature):
# import os
# import sys 
# sys.path.append("/data4/tongshuo/Grading/CommonFeatureLearning")
# from loss.SDD_DKD import multi_dkd
# alpha = 1
# beta = 8
# temperature = 1
# num_classes = 5
# # 示例输入
# input_data = torch.randn(8, 3, 224, 224)  # 一张224x224的RGB图像
# target = torch.randint(0, num_classes, (8,))
# # 创建带有 MCDropout 的 ResNet-34 模型
# model_t = resnet50_sdd(pretrained=False, num_classes=5, M = '[1,2,4]')
# model_s = resnet18_sdd(pretrained=False, num_classes=5, M = '[1,2,4]')
# model_t.load_state_dict(torch.load('/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/resnet50_aptos01234_max_acc_ce_notran.pth')['model_state'])
# model_s
# # 进行多次前向传播并获取均值和方差
# output, patch_score_t = model_t(input_data)
# output, patch_score_s = model_s(input_data)

# # 输出结果和低通滤波
# print(f'Output Shape: {output.shape}')
# print(f'Low Pass Shape: {patch_score_t.shape, patch_score_s.shape}')

# loss = multi_dkd(patch_score_s, patch_score_t, target, alpha, beta, temperature)

# print('loss:',loss)