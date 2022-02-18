
# based on:
# https://github.com/cavalleria/cavaface.pytorch/blob/master/backbone/efficientnet.py
from inspect import isfunction
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import ReLU, ReLU6, PReLU
from torch.nn import Module
from torchkit.backbone.common import SEModule, GDC


class ConvBlock(Module):
    """ Convolution block
    """
    def __init__(self, in_c, out_c, kernel_size=1, stride=1, padding=0, groups=1,
                 use_bn=True, activation=(lambda: ReLU(inplace=True))):
        super(ConvBlock, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_c) if use_bn else None

        if activation is None:
            self.act = None
        elif isfunction(activation):
            self.act = activation
        elif isinstance(activation, str):
            if activation == "relu":
                self.act = ReLU(inplace=True)
            elif activation == "relu6":
                self.act = ReLU6(inplace=True)
            elif activation == "prelu":
                self.act = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x


def conv1x1_block(in_channels, out_channels, stride=1, padding=0,
                  use_bn=True, activation=(lambda: nn.ReLU(inplace=True))):
    """
    1x1 version of the standard convolution block.
    """
    return ConvBlock(in_c=in_channels, out_c=out_channels, kernel_size=1, stride=stride,
                     padding=padding, groups=1, use_bn=use_bn, activation=activation)


def conv3x3_block(in_channels, out_channels, stride=1, padding=1,
                  use_bn=True, activation=(lambda: nn.ReLU(inplace=True))):
    """
    3x3 version of the standard convolution block.
    """
    return ConvBlock(in_c=in_channels, out_c=out_channels, kernel_size=3, stride=stride,
                     padding=padding, groups=1, use_bn=use_bn, activation=activation)


def dwconv3x3_block(in_channels, out_channels, stride=1, padding=1,
                    use_bn=True, activation=(lambda: nn.ReLU(inplace=True))):
    """
    3x3 depthwise version of the standard convolution block.
    """
    return ConvBlock(in_c=in_channels, out_c=out_channels, kernel_size=3, stride=stride,
                     padding=padding, groups=out_channels, use_bn=use_bn, activation=activation)


def dwconv5x5_block(in_channels, out_channels, stride=1, padding=2,
                    use_bn=True, activation=(lambda: nn.ReLU(inplace=True))):
    """
    5x5 depthwise version of the standard convolution block.
    """
    return ConvBlock(in_c=in_channels, out_c=out_channels, kernel_size=5, stride=stride,
                     padding=padding, groups=out_channels, use_bn=use_bn, activation=activation)


def round_channels(channels, divisor=8):
    """
    Round weighted channel number (make divisible operation).
    Parameters:
    ----------
    channels : int or float
        Original number of channels.
    divisor : int, default 8
        Alignment value.
    Returns
    -------
    int
        Weighted number of channels.
    """
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels


def calc_tf_padding(x,
                    kernel_size,
                    stride=1,
                    dilation=1):
    """
    Calculate TF-same like padding size.
    Parameters:
    ----------
    x : tensor
        Input tensor.
    kernel_size : int
        Convolution window size.
    stride : int, default 1
        Strides of the convolution.
    dilation : int, default 1
        Dilation value for convolution layer.
    Returns
    -------
    tuple of 4 int
        The size of the padding.
    """
    height, width = x.size()[2:]
    oh = math.ceil(height / stride)
    ow = math.ceil(width / stride)
    pad_h = max((oh - 1) * stride + (kernel_size - 1) * dilation + 1 - height, 0)
    pad_w = max((ow - 1) * stride + (kernel_size - 1) * dilation + 1 - width, 0)
    return pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2


class EffiDwsConvUnit(nn.Module):
    """
    EfficientNet specific depthwise separable convolution block/unit with BatchNorms and activations at each convolution
    layers.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    bn_eps : float
        Small float added to variance in Batch norm.
    activation : str
        Name of activation function.
    tf_mode : bool
        Whether to use TF-like mode.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bn_eps,
                 activation,
                 tf_mode):
        super(EffiDwsConvUnit, self).__init__()
        self.tf_mode = tf_mode
        self.residual = (in_channels == out_channels) and (stride == 1)

        self.dw_conv = dwconv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            padding=(0 if tf_mode else 1),
            activation=activation)
        self.se = SEModule(
            channels=in_channels,
            reduction=4)
        self.pw_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        if self.tf_mode:
            x = F.pad(x, pad=calc_tf_padding(x, kernel_size=3))
        x = self.dw_conv(x)
        x = self.se(x)
        x = self.pw_conv(x)
        if self.residual:
            x = x + identity
        return x


class EffiInvResUnit(nn.Module):
    """
    EfficientNet inverted residual unit.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    exp_factor : int
        Factor for expansion of channels.
    se_factor : int
        SE reduction factor for each unit.
    bn_eps : float
        Small float added to variance in Batch norm.
    activation : str
        Name of activation function.
    tf_mode : bool
        Whether to use TF-like mode.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 exp_factor,
                 se_factor,
                 bn_eps,
                 activation,
                 tf_mode):
        super(EffiInvResUnit, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.tf_mode = tf_mode
        self.residual = (in_channels == out_channels) and (stride == 1)
        self.use_se = se_factor > 0
        mid_channels = in_channels * exp_factor
        dwconv_block_fn = dwconv3x3_block if kernel_size == 3 else (dwconv5x5_block if kernel_size == 5 else None)

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            activation=activation)
        self.conv2 = dwconv_block_fn(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            padding=(0 if tf_mode else (kernel_size // 2)),
            activation=activation)
        if self.use_se:
            self.se = SEModule(
                channels=mid_channels,
                reduction=(exp_factor * se_factor))
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        x = self.conv1(x)
        if self.tf_mode:
            x = F.pad(x, pad=calc_tf_padding(x, kernel_size=self.kernel_size, stride=self.stride))
        x = self.conv2(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv3(x)
        if self.residual:
            x = x + identity
        return x


class EffiInitBlock(nn.Module):
    """
    EfficientNet specific initial block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    activation : str
        Name of activation function.
    tf_mode : bool
        Whether to use TF-like mode.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_eps,
                 activation,
                 tf_mode):
        super(EffiInitBlock, self).__init__()
        self.tf_mode = tf_mode

        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=1,
            padding=(0 if tf_mode else 1),
            activation=activation)

    def forward(self, x):
        if self.tf_mode:
            x = F.pad(x, pad=calc_tf_padding(x, kernel_size=3, stride=2))
        x = self.conv(x)
        return x


class EfficientNet(nn.Module):
    """
    EfficientNet model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    kernel_sizes : list of list of int
        Number of kernel sizes for each unit.
    strides_per_stage : list int
        Stride value for the first unit of each stage.
    expansion_factors : list of list of int
        Number of expansion factors for each unit.
    dropout_rate : float, default 0.2
        Fraction of the input units to drop. Must be a number between 0 and 1.
    tf_mode : bool, default False
        Whether to use TF-like mode.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 kernel_sizes,
                 strides_per_stage,
                 expansion_factors,
                 dropout_rate=0.2,
                 tf_mode=False,
                 bn_eps=1e-5,
                 in_channels=3,
                 in_size=(112, 112),
                 embedding_size=512):
        super(EfficientNet, self).__init__()
        self.in_size = in_size
        activation = "prelu"

        self.features = nn.Sequential()
        self.features.add_module("init_block", EffiInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            bn_eps=bn_eps,
            activation=activation,
            tf_mode=tf_mode))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            kernel_sizes_per_stage = kernel_sizes[i]
            expansion_factors_per_stage = expansion_factors[i]
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                kernel_size = kernel_sizes_per_stage[j]
                expansion_factor = expansion_factors_per_stage[j]
                stride = strides_per_stage[i] if (j == 0) else 1
                if i == 0:
                    stage.add_module("unit{}".format(j + 1), EffiDwsConvUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        bn_eps=bn_eps,
                        activation=activation,
                        tf_mode=tf_mode))
                else:
                    stage.add_module("unit{}".format(j + 1), EffiInvResUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        exp_factor=expansion_factor,
                        se_factor=4,
                        bn_eps=bn_eps,
                        activation=activation,
                        tf_mode=tf_mode))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_block", conv1x1_block(
            in_channels=in_channels,
            out_channels=final_block_channels,
            activation=activation))
        in_channels = final_block_channels
       
        self.output = GDC(512, embedding_size)
        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def efficientnet(input_size, embedding_size=512, version='b1', **kwargs):
    """
    Create EfficientNet model with specific parameters.
    """
    assert input_size[0] in [112]
    if version.endswith('b') or version.endswith('c'):
        version = version[:-1]
        tf_mode = True
        bn_eps = 1e-3
    else:
        tf_mode = False
        bn_eps = 1e-5
    if version == "b0":
        in_size = (112, 112)
        depth_factor = 1.0
        width_factor = 1.0
        dropout_rate = 0.2
    elif version == "b1":
        in_size = (120, 120)
        depth_factor = 1.1
        width_factor = 1.0
        dropout_rate = 0.2
    elif version == "b2":
        in_size = (130, 130)
        depth_factor = 1.2
        width_factor = 1.1
        dropout_rate = 0.3
    elif version == "b3":
        in_size = (150, 150)
        depth_factor = 1.4
        width_factor = 1.2
        dropout_rate = 0.3
    elif version == "b4":
        in_size = (190, 190)
        depth_factor = 1.8
        width_factor = 1.4
        dropout_rate = 0.4
    elif version == "b5":
        in_size = (228, 228)
        depth_factor = 2.2
        width_factor = 1.6
        dropout_rate = 0.4
    elif version == "b6":
        in_size = (264, 264)
        depth_factor = 2.6
        width_factor = 1.8
        dropout_rate = 0.5
    elif version == "b7":
        in_size = (300, 300)
        depth_factor = 3.1
        width_factor = 2.0
        dropout_rate = 0.5
    elif version == "b8":
        in_size = (672, 672)
        depth_factor = 3.6
        width_factor = 2.2
        dropout_rate = 0.5
    else:
        raise ValueError("Unsupported EfficientNet version {}".format(version))

    init_block_channels = 32
    layers = [1, 2, 2, 3, 3, 4, 1]
    downsample = [1, 1, 1, 1, 0, 1, 0]
    channels_per_layers = [16, 24, 40, 80, 112, 192, 320]
    expansion_factors_per_layers = [1, 6, 6, 6, 6, 6, 6]
    kernel_sizes_per_layers = [3, 3, 5, 3, 5, 5, 3]
    strides_per_stage = [1, 2, 2, 2, 1, 2, 1]
    final_block_channels = 512

    layers = [int(math.ceil(li * depth_factor)) for li in layers]
    channels_per_layers = [round_channels(ci * width_factor) for ci in channels_per_layers]

    from functools import reduce
    channels = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                      zip(channels_per_layers, layers, downsample), [])
    kernel_sizes = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                          zip(kernel_sizes_per_layers, layers, downsample), [])
    expansion_factors = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                               zip(expansion_factors_per_layers, layers, downsample), [])
    strides_per_stage = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                               zip(strides_per_stage, layers, downsample), [])
    strides_per_stage = [si[0] for si in strides_per_stage]

    init_block_channels = round_channels(init_block_channels * width_factor)

    if width_factor > 1.0:
        assert (int(final_block_channels * width_factor) == round_channels(final_block_channels * width_factor))
        final_block_channels = round_channels(final_block_channels * width_factor)

    net = EfficientNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernel_sizes=kernel_sizes,
        strides_per_stage=strides_per_stage,
        expansion_factors=expansion_factors,
        dropout_rate=dropout_rate,
        tf_mode=tf_mode,
        bn_eps=bn_eps,
        in_size=in_size,
        embedding_size=embedding_size,
        **kwargs)

    return net


def EfficientNetB0(input_size):
    """  Constructs an EfficientNet-B0 model.
    """
    return efficientnet(input_size, embedding_size=512, version='b0')


def EfficientNetB1(input_size):
    """  Constructs an EfficientNet-B0 model.
    """
    return efficientnet(input_size, embedding_size=512, version='b1')
