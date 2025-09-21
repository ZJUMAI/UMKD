import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResBlock(nn.Module):
    """ Residual Blocks
    """
    def __init__(self, inplanes, planes, stride=1, momentum=0.1):
#########inplanes：输入的通道数（即输入特征图的通道数）。
#########planes：输出的通道数（即该层的卷积层输出通道数）。
#########stride：卷积操作的步幅（默认是 1）。步幅决定了卷积操作时特征图尺寸的缩小程度。
#########momentum：用于批归一化（batch normalization）的动量（默认是 0.1）
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = nn.BatchNorm2d(planes, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes, momentum=momentum)
        if stride > 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                #nn.BatchNorm2d(planes, momentum=momentum)
            )
        else:
            self.downsample = None
#########残差连接中的下采样（即将输入的特征图大小减小或通道数增加的情况）
#########减少空间维度、增大感受野、减少计算和内存消耗，并对齐不同层次的特征图

        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        #out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class CFL_ConvBlock(nn.Module):
    """Common Feature Blocks for Convolutional layer
    卷积层的公共特征学习 
    This module is used to capture the common features of multiple teachers and calculate mmd with features of student.
    它用于捕捉多个教师模型的共同特征，并与学生模型的特征计算最大均值差异
    **Parameters:**
            channel_s:学生特征的通道数。
            channel_t:教师特征的通道数列表。
            channel_h:隐藏共同特征的通道数。
        - **channel_s** (int): channel of student features
        - **channel_t** (list or tuple): channel list of teacher features
        - **channel_h** (int): channel of hidden common features
    """
                    #   512,      [512, 512],     128   
    def __init__(self, channel_s, channel_t, channel_h):
        super(CFL_ConvBlock, self).__init__()

        self.align_t = nn.ModuleList()
        for ch_t in channel_t:
            self.align_t.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=ch_t, out_channels=2*channel_h,
                              kernel_size=1, bias=False),
                    nn.ReLU(inplace=True)
                )
            )

        self.align_s = nn.Sequential(
            nn.Conv2d(in_channels=channel_s, out_channels=2*channel_h,
                      kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        #特征提取器。是一个由多个残差块（ResBlock）组成的序列，用于从对齐后的特征中提取共同的隐藏特征。
        self.extractor = nn.Sequential(
            ResBlock(inplanes=2*channel_h, planes=channel_h, stride=1),
            ResBlock(inplanes=channel_h, planes=channel_h, stride=1),
            ResBlock(inplanes=channel_h, planes=channel_h, stride=1),
        )

        #教师特征解码器。教师特征映射回原始空间
        self.dec_t = nn.ModuleList()
        for ch_t in channel_t:
            self.dec_t.append(
                nn.Sequential(
                    nn.Conv2d(channel_h, ch_t, kernel_size=3,
                              stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch_t, ch_t, kernel_size=1,
                              stride=1, padding=0, bias=False)
                )
            )
    #初始化网络权重的函数
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


    def forward(self, fs, ft):
        #将教师特征通过对应的对齐层进行对齐
        aligned_t = [self.align_t[i](ft[i]) for i in range(len(ft))]
        #将学生特征通过对齐层进行对齐
        aligned_s = self.align_s(fs)

        #对齐后的每个教师特征通过特征提取器提取隐藏特征
        ht = [self.extractor(f) for f in aligned_t]
        #对齐后的学生特征通过特征提取器提取隐藏特征
        hs = self.extractor(aligned_s)

        #教师的解码特征，用于重构损失
        ft_ = [self.dec_t[i](ht[i]) for i in range(len(ht))]

        #最终返回学生、教师的隐藏特征；教师的解码特征、原始教师特征
        return (hs, ht), (ft_, ft)
