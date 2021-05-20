
# based on:
# https://github.com/cavalleria/cavaface.pytorch/blob/master/backbone/mobilefacenet.py
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import PReLU
from torch.nn import Sequential
from torch.nn import Module
from torchkit.backbone.common import initialize_weights
from torchkit.backbone.common import LinearBlock, GNAP, GDC


class Conv_block(Module):
    """ Convolution block with no-linear activation layer
    """
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Depth_Wise(Module):
    """ Depthwise block
    """
    def __init__(self, in_c, out_c, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, residual=False):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, groups, (1, 1), (1, 1), (0, 0))
        self.conv_dw = Conv_block(groups, groups, kernel, stride, padding, groups=groups)
        self.project = LinearBlock(groups, out_c, (1, 1), (1, 1), (0, 0))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    """ Residual block
    """
    def __init__(self, channel, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(channel, channel,
                                      kernel=kernel,
                                      stride=stride,
                                      padding=padding,
                                      groups=groups,
                                      residual=True))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MobileFaceNet(Module):
    """ MobileFaceNet backbone
    """
    def __init__(self, input_size, embedding_size=512, output_name="GDC"):
        """ Args:
            input_size: input_size of backbone
            embedding_size: embedding_size of last feature
            output_name: support GDC or GNAP
        """
        super(MobileFaceNet, self).__init__()
        assert output_name in ["GNAP", 'GDC']
        assert input_size[0] in [112]
        self.conv1 = Conv_block(3, 64, (3, 3), (2, 2), (1, 1))
        self.conv2_dw = Conv_block(64, 64, (3, 3), (1, 1), (1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, (3, 3), (2, 2), (1, 1), groups=128)
        self.conv_3 = Residual(64,
                               num_block=4,
                               groups=128,
                               kernel=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, (3, 3), (2, 2), (1, 1), groups=256)
        self.conv_4 = Residual(128,
                               num_block=6,
                               groups=256,
                               kernel=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, (3, 3), (2, 2), (1, 1), groups=512)
        self.conv_5 = Residual(128,
                               num_block=2,
                               groups=256,
                               kernel=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, (1, 1), (1, 1), (0, 0))
        if output_name == "GNAP":
            self.output_layer = GNAP(512)
        else:
            self.output_layer = GDC(512, embedding_size)

        initialize_weights(self.modules())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        conv_features = self.conv_6_sep(out)
        out = self.output_layer(conv_features)
        return out
