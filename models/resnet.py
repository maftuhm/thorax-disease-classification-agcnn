import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from utils import LSEPool2d
from utils.utils import reduce_weight_bias

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.controllerelu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.controllerelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.controllerelu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.controllerelu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.controllerelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.controllerelu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.controllerelu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, last_pool = 'lse', lse_pool_controller = 10, is_grayscale = True):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        in_channel = 3
        if is_grayscale:
            in_channel = 1

        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.controllerelu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # ----------------- additional last pooling layer -----------------
        if last_pool == 'lse':
            self.last_pool = LSEPool2d(controller = lse_pool_controller)
        elif last_pool == 'max':
            self.last_pool = nn.MaxPool2d(kernel_size = 7, stride = 1)
        elif last_pool == 'avg':
            self.last_pool = nn.AvgPool2d(kernel_size = 7, stride = 1)
        elif last_pool == 'adaptive_avg':
            self.last_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise Exception("pooling layer must be lse, max or avg")

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.controllerelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x)

        x = self.last_pool(features)
        flatten_pool = torch.flatten(x, 1)
        x = self.fc(flatten_pool)
        x = torch.sigmoid(x)

        result = {
            'score': x,
            'features': features,
            'pool': flatten_pool
        }

        return result

    def forward(self, x):
        return self._forward_impl(x)


def ResNet50(pretrained = True, num_classes = 14, is_grayscale = True, last_pool = 'lse', lse_pool_controller = 10, group_norm = False, **kwargs):

    if group_norm:
        norm_layer = lambda x: nn.GroupNorm(32, x)
    else:
        norm_layer = None

    model = ResNet(block = Bottleneck, layers = [3, 4, 6, 3], num_classes = num_classes, is_grayscale = is_grayscale,
                    last_pool = last_pool, lse_pool_controller = lse_pool_controller, norm_layer = norm_layer, **kwargs)

    if pretrained:
        print(" Loading state dict from", model_urls['resnet50'])
        # Pretrained ResNet base
        loaded_state_dict = load_state_dict_from_url(model_urls['resnet50'], progress = True)

        del loaded_state_dict['fc.weight'], loaded_state_dict['fc.bias']

        # loaded_state_dict['fc.weight'], loaded_state_dict['fc.bias'] = reduce_weight_bias(loaded_state_dict['fc.weight'].data, loaded_state_dict['fc.bias'].data, num_classes)

        if is_grayscale:
            loaded_state_dict['conv1.weight'] = loaded_state_dict['conv1.weight'].mean(axis=1, keepdim = True).data
            # loaded_state_dict['conv1.weight'] = (loaded_state_dict['conv1.weight'] ** 2).sum(axis = 1, keepdim=True).sqrt().data

        model_state_dict = model.state_dict()

        for key in list(model_state_dict.keys()):
            if key in list(loaded_state_dict.keys()):
                model_state_dict[key] = loaded_state_dict[key]

        # Subsample fc with size [1000, 2048] to [num_classes, 2048] by unfold
        # step_fold = (1000 // model.state_dict()['fc.weight'].size(0)) + 1
        # state_dict['fc.weight'] = state_dict['fc.weight'].unfold(0, 1, step_fold).flatten(1)  # (num_classes, 2048)
        # state_dict['fc.bias'] = state_dict['fc.bias'].unfold(0, 1, step_fold).flatten()  # (num_classes)

        # state_dict['fc.weight'] = model.state_dict()['fc.weight'].data
        # state_dict['fc.bias'] = model.state_dict()['fc.bias'].data

        model.load_state_dict(model_state_dict)
        del loaded_state_dict
        torch.cuda.empty_cache()
        print(" State dict is loaded")

    return model

def ResNet101(pretrained = True, num_classes = 14, is_grayscale = True, last_pool = 'lse', lse_pool_controller = 10, group_norm = False, **kwargs):

    if group_norm:
        norm_layer = lambda x: nn.GroupNorm(32, x)
    else:
        norm_layer = None

    model = ResNet(block = Bottleneck, layers = [3, 4, 23, 3], num_classes = num_classes, is_grayscale = is_grayscale,
                    last_pool = last_pool, lse_pool_controller = lse_pool_controller, norm_layer = norm_layer, **kwargs)

    if pretrained:
        print(" Loading state dict from", model_urls['resnet101'])
        # Pretrained ResNet base
        loaded_state_dict = load_state_dict_from_url(model_urls['resnet101'], progress = True)

        del loaded_state_dict['fc.weight'], loaded_state_dict['fc.bias']

        if is_grayscale:
            loaded_state_dict['conv1.weight'] = loaded_state_dict['conv1.weight'].mean(axis=1, keepdim = True).data

        model_state_dict = model.state_dict()

        for key in list(model_state_dict.keys()):
            if key in list(loaded_state_dict.keys()):
                model_state_dict[key] = loaded_state_dict[key]

        model.load_state_dict(model_state_dict)
        del loaded_state_dict, model_state_dict
        print(" State dict is loaded")

    return model