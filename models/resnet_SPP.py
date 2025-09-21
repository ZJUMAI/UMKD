import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
        out = self.relu(out)

        return out


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
# # 定义测试数据
# B, C, H, W = 2, 64, 8, 8  # Batch size, Channels, Height, Width
# feats = torch.randn(B, C, H, W)  # 随机生成一个图像特征张量
# # 实例化 StuLowPassModule
# stu_low_module = StuLowPassModule(in_channel=C, kernel_size=3, stride=1)  # 使用 3x3 的卷积核和步幅                                                                                       
# # 通过 StuLowPassModule 进行前向传播
# output = stu_low_module(feats)

class SPPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6)):
        # 低通滤波器的不同尺寸，可以通过不同的滤波器尺寸进行处理，大小分别为1x1，2x2，3x3和6x6。
        super(SPPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        # 为每种大小创建一个滤波器（使用 nn.AdaptiveAvgPool2d 进行池化），以适应输入特征图的大小。
        # self.relu = nn.ReLU()
        self.sepconv = SepConv(channel_in=in_channels*4, channel_out=in_channels, kernel_size=3, stride=1, padding=1)

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    # def forward(self, feats):
    #     h, w = feats.size(2), feats.size(3) # 获取输入特征图的高度和宽度
    #     priors = [self.stages[i](feats) for i in range(4)]
    #     # priors = [F.upsample(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear') for i in range(4)] # 对每个部分应用 stages[i]，即进行自适应池化后，再通过双线性插值将其上采样至输入图像的原始大小。
    #     # bottle = torch.cat(priors, 1) # 将所有上采样后的部分沿着通道维度拼接。
    #     for i in range(len(priors)):
    #         priors[i] = torch.flatten(priors[i], start_dim=2, end_dim=3)  # B X C X feature_num
    #     # merged_tensor = torch.cat(priors, dim=2)
    #     merged_tensor = torch.cat(priors, dim=2).unsqueeze(-1)
    #     out_feats = self.sepconv(merged_tensor).squeeze()
    #     return out_feats
    def forward(self, feats):
        h, w = feats.size(2), feats.size(3) # 获取输入特征图的高度和宽度
        # priors = [self.stages[i](feats) for i in range(4)]
        priors = [F.upsample(input=self.stages[i](feats), size=(h, w), mode='bilinear') for i in range(4)] # 对每个部分应用 stages[i]，即进行自适应池化后，再通过双线性插值将其上采样至输入图像的原始大小。
        bottle = torch.cat(priors, 1) # 将所有上采样后的部分沿着通道维度拼接。
        out_feats = self.sepconv(bottle)
        # priors = [F.upsample(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear') for i in range(4)] # 对每个部分应用 stages[i]，即进行自适应池化后，再通过双线性插值将其上采样至输入图像的原始大小。
        # bottle = torch.cat(priors, 1) # 将所有上采样后的部分沿着通道维度拼接。
        # for i in range(len(priors)):
        #     priors[i] = torch.flatten(priors[i], start_dim=2, end_dim=3)  # B X C X feature_num
        # merged_tensor = torch.cat(priors, dim=2).unsqueeze(-1)
        # out_feats = self.sepconv(merged_tensor).squeeze()
        return out_feats

class SPP_LPModule(nn.Module):
    #######(1, 2, 3, 6)替换为(1, 2, 4)#########
    def __init__(self, sizes=(1, 2, 4)):
        # 低通滤波器的不同尺寸，可以通过不同的滤波器尺寸进行处理，大小分别为1x1，2x2，3x3和6x6。
        super().__init__()
        self.sizes = sizes
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        # 为每种大小创建一个滤波器（使用 nn.AdaptiveAvgPool2d 进行池化），以适应输入特征图的大小。
        self.relu = nn.ReLU()
    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feats):
#########输入特征不进行双线性插值和上采样############
        priors = [self.stages[i](feats) for i in range(len(self.sizes))]
#########输入特征进行双线性插值和上采样############
        # h, w = feats.size(2), feats.size(3) # 获取输入特征图的高度和宽度
        # priors = [F.upsample(input=self.stages[i](feats), size=(h, w), mode='bilinear') for i in range(4)] # 对每个部分应用 stages[i]，即进行自适应池化后，再通过双线性插值将其上采样至输入图像的原始大小。
        # return self.relu(priors[0]), self.relu(priors[1]), self.relu(priors[2]), self.relu(priors[3]) # 返回四个不同池化尺度的部分
        return self.relu(priors[0]), self.relu(priors[1]), self.relu(priors[2]) # 返回三个不同池化尺度的部分

class StuSPPLowPassModule(nn.Module):
    def __init__(self, in_channel, kernel_size=3, stride = 1, sizes=(1, 2, 4)):
        super().__init__()
        # 低通滤波器的卷积部分，保证和自适应卷积的大小一致
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=1)
        # 低通滤波器部分，返回不同尺度的浅层特征
        self.low_pass = SPP_LPModule(sizes=sizes)
        # 归一化操作：使用 LayerNorm
        self.aggregate = Conv2d_BN(in_channel, in_channel, act_layer=nn.Hardswish)
        # 深度分离卷积操作：使用 SepConv
        self.sepconv = SepConv(channel_in=in_channel, channel_out=in_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, feats):
###########低通滤波前的卷积操作###########
        # feats = self.conv(feats)
        # # 低通滤波器处理
        # low_pass_feats_1, low_pass_feats_2, low_pass_feats_3, low_pass_feats_6 = self.low_pass(feats)
        # # 聚合加归一化
        # low_pass_feats_1 = self.aggregate(low_pass_feats_1)
        # low_pass_feats_2 = self.aggregate(low_pass_feats_2)
        # low_pass_feats_3 = self.aggregate(low_pass_feats_3)
        # low_pass_feats_6 = self.aggregate(low_pass_feats_6)
        # # 深度可分离卷积
        # low_pass_feats_1 = self.sepconv(low_pass_feats_1)
        # low_pass_feats_2 = self.sepconv(low_pass_feats_2)
        # low_pass_feats_3 = self.sepconv(low_pass_feats_3)
        # low_pass_feats_6 = self.sepconv(low_pass_feats_6)
        # return low_pass_feats_1, low_pass_feats_2, low_pass_feats_3, low_pass_feats_6
        feats = self.conv(feats)
        # 低通滤波器处理
        low_pass_feats_1, low_pass_feats_2, low_pass_feats_4 = self.low_pass(feats)
        # 聚合加归一化
        low_pass_feats_1 = self.aggregate(low_pass_feats_1)
        low_pass_feats_2 = self.aggregate(low_pass_feats_2)
        low_pass_feats_4 = self.aggregate(low_pass_feats_4)
        # 深度可分离卷积
        low_pass_feats_1 = self.sepconv(low_pass_feats_1)
        low_pass_feats_2 = self.sepconv(low_pass_feats_2)
        low_pass_feats_4 = self.sepconv(low_pass_feats_4)
        return low_pass_feats_1, low_pass_feats_2, low_pass_feats_4

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #最大池化层，池化窗口为 3x3，步长为 2，padding 为 1，进一步减少特征图的尺寸
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #网络的四个主要层，每一层堆叠多个残差块，_make_layer
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.stu_low_module_1 = StuSPPLowPassModule(in_channel = 64)                                                                                   
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.stu_low_module_2 = SPPModule()                                                                                       
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.stu_low_module_3 = SPPModule()                                                                                      
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.stu_low_module_4 = SPPModule()                                                                                
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

#####_make_layer 方法用于生成网络中的每一层（layer1、layer2、layer3、layer4）
#####它会创建多个残差块（block），并且如果需要，添加下采样（downsample）操作
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # print('layer1_out:', x.shape)
######### 添加学生的低通滤波模块 ###########
        # layer1_low_pass_feats_1, layer1_low_pass_feats_2, layer1_low_pass_feats_3, layer1_low_pass_feats_6 = self.stu_low_module_1(x)
        layer1_low_pass_feats_1, layer1_low_pass_feats_2, layer1_low_pass_feats_4 = self.stu_low_module_1(x)
        # print('layer1_low_pass_feats:', layer1_low_pass_feats.shape)
##########################################
        x = self.layer2(x)
######### 添加学生的低通滤波模块 ###########
        # layer2_low_pass_feats = self.stu_low_module_2(x)
##########################################
        x = self.layer3(x)
######### 添加学生的低通滤波模块 ###########
        # layer3_low_pass_feats = self.stu_low_module_3(x)
##########################################
        x = self.layer4(x)
######### 添加学生的低通滤波模块 ###########
        # layer4_low_pass_feats = self.stu_low_module_4(x)
##########################################
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, layer1_low_pass_feats_1, layer1_low_pass_feats_2, layer1_low_pass_feats_4


def resnet18_LP(pretrained=False, num_classes=None, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #[2, 2, 2, 2]列表表示 ResNet-18 中每个阶段的 block 数量
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    if num_classes is not None:
        model.fc = nn.Linear(512, num_classes)
    return model



def resnet34_LP(pretrained=False, num_classes=None, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    if num_classes is not None:
        model.fc = nn.Linear(512, num_classes)
    return model



def resnet50_LP(pretrained=False, num_classes=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    if num_classes is not None:
        model.fc = nn.Linear(2048, num_classes)
    return model



def resnet101_LP(pretrained=False, num_classes=None, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    if num_classes is not None:
        model.fc = nn.Linear(2048, num_classes)
    return model



def resnet152_LP(pretrained=False, num_classes=None, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    if num_classes is not None:
        model.fc = nn.Linear(2048, num_classes)
    return model


# # 示例输入
# input_data = torch.randn(128, 3, 224, 224)  # 一张224x224的RGB图像

# # 创建带有 MCDropout 的 ResNet-34 模型
# model = resnet18(pretrained=False, num_classes=5)

# # model.load_state_dict(torch.load('/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/resnet50_aptos01234_max_acc_ce_notran.pth')['model_state'])

# # 进行多次前向传播并获取均值和方差
# output, layer1_low_pass_feats = model(input_data)

# # 输出结果和低通滤波
# print(f'Output Shape: {output.shape}')
# print(f'Low Pass Shape: {layer1_low_pass_feats.shape}')