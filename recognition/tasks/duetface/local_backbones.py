import os
import sys
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from torchkit.backbone import get_model
from torchkit.backbone.common import initialize_weights


class IntConv(nn.Module):
    def __init__(self, channels_in, channels_out, feature_channels_in, kernel_size):
        super(IntConv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.features_channels_in = feature_channels_in

        self.kernel = nn.Sequential(
            nn.Linear(self.features_channels_in, self.features_channels_in, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.features_channels_in, self.channels_in * self.kernel_size * self.kernel_size, bias=False),
        )
        self.conv = nn.Conv2d(channels_in, channels_out, 1, padding=(1 // 2), bias=True)
        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, inputs):
        main_inputs, embedding_inputs = inputs[0], inputs[1]
        b, c, h, w = main_inputs.size()
        kernel = self.kernel(embedding_inputs)
        kernel = kernel.view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(
            F.conv2d(main_inputs.view(1, -1, h, w), kernel, groups=b * c, padding=(self.kernel_size - 1) // 2))
        out = self.conv(out.view(b, -1, h, w))
        return out


class BasicBlockIntIR(nn.Module):
    """ BasicBlock for IRNet
    """
    def __init__(self, in_channel, depth, stride, feature_channel, kernel_size, stage=0, embedding=False):
        super(BasicBlockIntIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth))
        if embedding:
            self.embedding_layer = IntConv(depth, depth, feature_channel, kernel_size)
        else:
            self.embedding_layer = None

    def forward(self, x):
        main_x, embedding_x = x[0], x[1]
        shortcut = self.shortcut_layer(main_x)
        res = self.res_layer(main_x)
        main_x = shortcut + res
        if self.embedding_layer is not None:
            main_x = self.embedding_layer([main_x, embedding_x])
        return [main_x, embedding_x]


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + \
           [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2),
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    elif num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=6),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=8),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]
    elif num_layers == 200:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=24),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]

    return blocks


class ClientBackbone(nn.Module):
    def __init__(self, channels_in, channels_out, client_backbone_name='MobileFaceNet'):
        super(ClientBackbone, self).__init__()

        # sub_backbone_name = 'IR_18'
        self.client_backbone = get_model(client_backbone_name)
        self.client_backbone = self.client_backbone([112, 112])
        if client_backbone_name == 'MobileFaceNet':
            self.client_backbone = self._adjust_client_backbone_model(self.client_backbone, channels_in, channels_out)
        if client_backbone_name == 'IR_18':
            # deprecated
            backbone = nn.Sequential()
            backbone.add_module('input_layer', nn.Sequential(nn.Conv2d(channels_in, 64, (3, 3), 1, 1, bias=False),
                                                             nn.BatchNorm2d(64), nn.PReLU(64)))
            body = list(list(self.client_backbone.children())[2].children())
            for i in range(len(body)):
                backbone.add_module('body_{}'.format(str(i)), body[i])
            backbone.add_module('output_layer', self.client_backbone.output_layer)
            self.client_backbone = backbone

    def _adjust_client_backbone_model(self, backbone, in_channels, out_channels):
        # mainly to reshape the input and output layer
        if backbone is None:
            backbone = self.client_backbone

        body = list(backbone.children())[1:-1]
        input_layer = torch.nn.Sequential(
            nn.Conv2d(in_channels, 64, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        output_layer = nn.Sequential(
            nn.Conv2d(512, 512, (7, 7), (1, 1), groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(512, out_channels),
            nn.BatchNorm1d(out_channels)
        )
        backbone = nn.Sequential()
        backbone.add_module('input_layer', input_layer)
        for i in range(len(body)):
            backbone.add_module('body_{}'.format(str(i)), body[i])
        backbone.add_module('output_layer', output_layer)

        return backbone

    def forward(self, x):
        return self.client_backbone(x)


class ServerBackbone(nn.Module):
    def __init__(self, input_size, num_layers, in_channels, out_channels, feature_channels, kernel_size,
                 unit_module=None):
        super(ServerBackbone, self).__init__()
        assert input_size[0] in [112, 224], \
            "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [18, 34, 50, 100, 152, 200], \
            "num_layers should be 18, 34, 50, 100 or 152"

        self.input_layer = nn.Sequential(nn.Conv2d(in_channels, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64), nn.PReLU(64))
        blocks = get_blocks(num_layers)
        if unit_module is None:
            unit_module = BasicBlockIntIR
        output_channel = 512

        self.output_layer = nn.Sequential(nn.BatchNorm2d(output_channel),
                                          nn.Dropout(0.4), nn.Flatten(),
                                          nn.Linear(output_channel * 7 * 7, out_channels),
                                          nn.BatchNorm1d(out_channels, affine=False))

        modules = []
        stage = 0
        for block in blocks:
            for i in range(len(block)):
                # modification: append embedding layer at the end of each stage
                if i < len(block) - 1:
                    modules.append(
                        unit_module(block[i].in_channel, block[i].depth,
                                    block[i].stride, feature_channels, kernel_size))
                else:
                    modules.append(
                        unit_module(block[i].in_channel, block[i].depth,
                                    block[i].stride, feature_channels, kernel_size, stage=stage, embedding=True))
            stage += 1
        self.body = nn.Sequential(*modules)

        initialize_weights(self.modules())

    def forward(self, x):
        if len(x) == 2:
            main_x, embedding_x = x[0], x[1]

            main_x = self.input_layer(main_x)
            [main_x, embedding_x] = self.body([main_x, embedding_x])
        else:  # len(x) == 3
            main_x, embedding_x, inference_x = x[0], x[1], x[2]
            main_x = self.input_layer(main_x)
            [main_x, embedding_x, inference_x] = self.body([main_x, embedding_x, inference_x])
        main_x = self.output_layer(main_x)
        return main_x
