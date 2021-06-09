import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple
import numpy as np
import pdb


class Flatten(Module):
    '''
    This method is to flatten the features
    '''
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input, axis=1):
    '''
    This method is for l2 normalization
    '''
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        '''
        This method is to initialize IR module
        '''
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), 
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), 
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

def get_block(in_channel, depth, num_units, stride=2):
    '''
    This method is to obtain blocks
    '''
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks(num_layers):
    '''
    This method is to obtain blocks
    '''
    if num_layers == 50:
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
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks

class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir', use_type = "Rec"):
        '''
        This method is to initialize model
        if use for quality network, select self.use_type == "Qua"
        if use for recognition network, select self.use_type == "Rec"
        '''
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir': unit_module = bottleneck_IR
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.use_type = use_type
        if input_size[0] == 112:
            if use_type == "Qua":
                self.quality = Sequential(Flatten(),
                                      PReLU(512 * 7 * 7),
                                      Dropout(0.5, inplace=False),
                                      Linear(512 * 7 * 7, 1)
                                    )
            else:
                self.output_layer = Sequential(Flatten(),
                                      PReLU(512 * 7 * 7),
                                      Dropout(0.5, inplace=False),
                                      Linear(512 * 7 * 7, 512)
                                    )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self._initialize_weights()

    def forward(self, x):
        '''
        This method is to model forward
        '''
        x = self.input_layer(x)
        x = self.body(x)
        if self.use_type == "Qua":
            x = self.quality(x)
        else:
            x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        '''
        This method is to initialize model weights
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

def R50(input_size, use_type = "Rec"):
    '''
    This method is to create ResNet50 backbone
    if use for quality network, select self.use_type == "Qua"
    if use for recognition network, select self.use_type == "Rec"
    '''
    model = Backbone(input_size, 50, 'ir', use_type = use_type)
    return model
