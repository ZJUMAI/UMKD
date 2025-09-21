import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo


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


class ResNetWithMCDropout(nn.Module):

    def __init__(self, block, layers, num_classes=1000, dropout_rate=0.5):
        self.inplanes = 64
        super(ResNetWithMCDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #最大池化层，池化窗口为 3x3，步长为 2，padding 为 1，进一步减少特征图的尺寸
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #网络的四个主要层，每一层堆叠多个残差块，_make_layer
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # Adding a Dropout layer before the fully connected layer
        self.dropout = nn.Dropout(p=dropout_rate)
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

    def forward(self, x, mc_dropout=False):
        if mc_dropout:
            self.train()  # Ensure the model is in training mode to enable dropout
        else:
            self.eval()  # Put the model in evaluation mode to disable dropout

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Apply dropout before the final fully connected layer
        x = self.dropout(x)
        x = self.fc(x)

        return x
# Monte Carlo Dropout inference method
    def mc_dropout_forward(self, x, num_samples=10):
        # self.train()  # Ensure model is in training mode for dropout
        outputs = []
        for _ in range(num_samples):
            output = self(x, mc_dropout=True)  # Run a forward pass with dropout enabled
            outputs.append(output)

        # Stack the outputs and compute the mean and variance
        outputs = torch.stack(outputs, dim=0)
        mean_output = outputs.mean(dim=0)  # Mean of the outputs
        uncertainty = outputs.var(dim=0)  # Variance as uncertainty

        return mean_output, uncertainty.sum()

def resnet18(pretrained=False, num_classes=None, dropout_rate=0.5, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #[2, 2, 2, 2]列表表示 ResNet-18 中每个阶段的 block 数量
    model = ResNetWithMCDropout(BasicBlock, [2, 2, 2, 2], dropout_rate=dropout_rate, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    if num_classes is not None:
        model.fc = nn.Linear(512, num_classes)
        # 在ResNet的最后加入Dropout层
        # model.fc = nn.Sequential(
        #     nn.Dropout(p=dropout_rate),  # Dropout层，p是丢弃的概率
        #     nn.Linear(512, num_classes)  # 全连接层
        # )
    return model



def resnet34(pretrained=False, num_classes=None, dropout_rate=0.5, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetWithMCDropout(BasicBlock, [3, 4, 6, 3], dropout_rate=dropout_rate, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    if num_classes is not None:
        model.fc = nn.Linear(512, num_classes)
    return model



def resnet50(pretrained=False, num_classes=None, dropout_rate=0.5, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetWithMCDropout(Bottleneck, [3, 4, 6, 3], dropout_rate=dropout_rate, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    if num_classes is not None:
        model.fc = nn.Linear(2048, num_classes)
    return model



def resnet101(pretrained=False, num_classes=None, dropout_rate=0.5, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetWithMCDropout(Bottleneck, [3, 4, 23, 3], dropout_rate=dropout_rate, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    if num_classes is not None:
        model.fc = nn.Linear(2048, num_classes)
    return model



def resnet152(pretrained=False, num_classes=None, dropout_rate=0.5, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetWithMCDropout(Bottleneck, [3, 8, 36, 3], dropout_rate=dropout_rate, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    if num_classes is not None:
        model.fc = nn.Linear(2048, num_classes)
    return model



# 示例输入
input_data = torch.randn(2, 3, 224, 224)  # 一张224x224的RGB图像

# 创建带有 MCDropout 的 ResNet-34 模型
model = resnet18(pretrained=False, num_classes=5, dropout_rate=0.5)

model.load_state_dict(torch.load('/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/resnet18_aptos01234_max_acc_ce_notran.pth')['model_state'])

# 进行多次前向传播并获取均值和方差
mean_output, uncertainty = model.mc_dropout_forward(input_data, num_samples=10)
# def forward_hook(module, input, output):
#     module.output = output # keep feature maps
# model.layer1.register_forward_hook(forward_hook)
# xxx = model(input_data)
# ft2_SA = model.layer1.output
# print(ft2_SA.shape)
# 输出均值和方差
print(f'Mean Output Shape: {mean_output.shape, mean_output}')
print(f'Uncertainty Shape: {uncertainty.shape, uncertainty}')
