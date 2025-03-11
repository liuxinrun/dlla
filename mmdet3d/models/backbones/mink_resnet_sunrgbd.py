# Copyright (c) OpenMMLab. All rights reserved.
# Follow https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/resnet.py # noqa
# and mmcv.cnn.ResNet
try:
    import MinkowskiEngine as ME
    from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')
    # blocks are used in the static part of MinkResNet
    BasicBlock, Bottleneck = None, None

import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet3d.models.builder import BACKBONES


@BACKBONES.register_module()
class MinkResNet(BaseModule):
    r"""Minkowski ResNet backbone. See `4D Spatio-Temporal ConvNets
    <https://arxiv.org/abs/1904.08755>`_ for more details.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (ont): Number of input channels, 3 for RGB.
        num_stages (int, optional): Resnet stages. Default: 4.
        pool (bool, optional): Add max pooling after first conv if True.
            Default: True.
    """
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels,
                 max_channels=None,
                 num_stages=4,
                 pool=True,
                 norm='instance'):
        super(MinkResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        assert 4 >= num_stages >= 1
        block, stage_blocks = self.arch_settings[depth]
        stage_blocks = stage_blocks[:num_stages]
        self.max_channels = max_channels
        self.num_stages = num_stages
        self.pool = pool

        self.inplanes = 64
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=3, stride=2, dimension=3)
        norm1 = ME.MinkowskiInstanceNorm if norm == 'instance' \
            else ME.MinkowskiBatchNorm
        self.norm1 = norm1(self.inplanes)
        self.relu = ME.MinkowskiReLU(inplace=True)
        if self.pool:
            self.maxpool = ME.MinkowskiMaxPooling(
                kernel_size=2, stride=2, dimension=3)

        for i, _ in enumerate(stage_blocks):
            n_channels = 64 * 2**i
            if self.max_channels is not None:
                n_channels = min(n_channels, self.max_channels)
            setattr(
                self, f'layer{i + 1}',
                self._make_layer(block, n_channels, stage_blocks[i], stride=2))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=3),
                ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                dimension=3))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dimension=3))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of ResNet.

        Args:
            x (ME.SparseTensor): Input sparse tensor.

        Returns:
            list[ME.SparseTensor]: Output sparse tensors.
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        if self.pool:
            x = self.maxpool(x)
        outs = []
        for i in range(self.num_stages):
            x = getattr(self, f'layer{i + 1}')(x)
            outs.append(x)
        return outs


@BACKBONES.register_module()
class MinkFFResNet(MinkResNet):
    def forward(self, x, f):
        """Forward pass of ResNet.

        Args:
            x (ME.SparseTensor): Input sparse tensor.

        Returns:
            list[ME.SparseTensor]: Output sparse tensors.
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        if self.pool:
            x = self.maxpool(x)
        x = f(x)
        outs = []
        for i in range(self.num_stages):
            x = getattr(self, f'layer{i + 1}')(x)
            outs.append(x)
        return outs

class MinkFormerBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1, dimension=-1):
        super(MinkFormerBlock, self).__init__()
        assert dimension > 0

        kernel_ = 1 if stride == 1 else 3
        dropout = 0.1
        self.a_conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=kernel_, stride=stride, dilation=dilation, dimension=dimension)
        self.a_conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=5, stride=1, dilation=dilation, dimension=dimension)
        self.v_conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=kernel_, stride=stride, dilation=dilation, dimension=dimension)
        self.conv1 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.relu1 = ME.MinkowskiReLU(inplace=True)
        self.relu2 = ME.MinkowskiReLU(inplace=True)
        self.dropout1 = ME.MinkowskiDropout(dropout, inplace=True)
        self.dropout2 = ME.MinkowskiDropout(dropout, inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        a = self.a_conv1(x)
        a = self.a_conv2(a)

        v = self.v_conv1(x)
        out = a * v
        out = self.conv1(out)
        out = self.norm1(out)
        if self.downsample is not None:
          residual = self.downsample(x)
        out = self.dropout1(out) + residual
        residual = out
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout2(out) + residual
        out = self.relu2(out)

        return out

@BACKBONES.register_module()
class MinkFormer(BaseModule):

    arch_settings = {
        18: (MinkFormerBlock, (2, 2, 2, 2)),
        34: (MinkFormerBlock, (3, 4, 6, 3)),
        50: (MinkFormerBlock, (3, 4, 6, 3)),
        101: (MinkFormerBlock, (3, 4, 23, 3)),
        152: (MinkFormerBlock, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels,
                 max_channels=None,
                 num_stages=4,
                 pool=True,
                 norm='instance'):
        super(MinkFormer, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        assert 4 >= num_stages >= 1
        block, stage_blocks = self.arch_settings[depth]
        stage_blocks = stage_blocks[:num_stages]
        self.max_channels = max_channels
        self.num_stages = num_stages
        self.pool = pool

        self.inplanes = 64
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=3, stride=2, dimension=3)
        norm1 = ME.MinkowskiInstanceNorm if norm == 'instance' \
            else ME.MinkowskiBatchNorm
        self.norm1 = norm1(self.inplanes)
        self.relu = ME.MinkowskiReLU(inplace=True)
        if self.pool:
            self.maxpool = ME.MinkowskiMaxPooling(
                kernel_size=2, stride=2, dimension=3)
        for i, _ in enumerate(stage_blocks):
            n_channels = 64 * 2**i
            if self.max_channels is not None:
                n_channels = min(n_channels, self.max_channels)
            setattr(
                self, f'layer{i + 1}',
                self._make_layer(block, n_channels, stage_blocks[i], stride=2))

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=3),
                ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = []
        layers.append(block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                dimension=3))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dimension=3))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of ResNet.

        Args:
            x (ME.SparseTensor): Input sparse tensor.

        Returns:
            list[ME.SparseTensor]: Output sparse tensors.
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        if self.pool:
            x = self.maxpool(x)
        outs = []
        for i in range(self.num_stages):
            x = getattr(self, f'layer{i + 1}')(x)
            outs.append(x)
        return outs
