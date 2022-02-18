import os
import sys
import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..'))
from common.utils.model_init import init_weights


def conv3x3(in_ch, out_ch, stride=1, padding=1, bias=True):
    '''
        the cnn module with specific parameters
        Args:
            in_ch (int): the channel numbers of input features
            out_ch (int): the channel numbers of output features
            strider (int): the stride paramters of Conv2d
            padding (int): the padding parameters of Conv2D
            bias (bool): the bool parameters of Conv2d
    '''
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=padding, bias=bias)


class Conv_block(nn.Module):

    def __init__(self, in_ch, out_ch, AdapNorm=False, AdapNorm_attention_flag=False, model_initial='kaiming'):
        '''
            Args:
                in_ch (int): the channel numbers of input features
                out_ch (int): the channel numbers of output features
                AdapNorm (bool): 
                    'True' allow the Conv_block to combine BN and IN
                    'False' allow the Conv_block to use BN
                AdapNorm_attention_flag (str):
                    '1layer' allow the Conv_block to use 1layer FC to generate the balance factor
                    '2layer' allow the Conv_block to use 2layer FC to generate the balance factor
                model_initial (str):
                    'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
        '''
        super(Conv_block, self).__init__()
        self.AdapNorm = AdapNorm
        self.AdapNorm_attention_flag = AdapNorm_attention_flag
        self.model_initial = model_initial

        self.conv = conv3x3(in_ch, out_ch)

        if self.AdapNorm:
            self.BN = nn.BatchNorm2d(out_ch)
            self.IN = nn.InstanceNorm2d(out_ch, affine=True)
            self.global_pool = nn.AdaptiveAvgPool2d(1)

            if self.AdapNorm_attention_flag is not None:
                if self.AdapNorm_attention_flag == '1layer':
                    self.AttentionNet = nn.Linear(out_ch, out_ch)
                elif self.AdapNorm_attention_flag == '2layer':
                    self.AttentionNet_fc1 = nn.Linear(out_ch, out_ch // 16 + 1)
                    self.AttentionNet_fc2 = nn.Linear(out_ch // 16 + 1, out_ch)

                self.sigmoid = nn.Sigmoid()

            else:
                self.gate_weight = torch.nn.Parameter(torch.Tensor(out_ch))
                self.gate_bias = torch.nn.Parameter(torch.Tensor(out_ch))
                self.gate_weight.data.fill_(0.5)
                self.gate_bias.data.fill_(0)
        else:
            self.IN = None
            self.BN = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

        # model initial
        init_weights(self.conv, init_type=self.model_initial)

    def load_params(self, z, params):
        if self.AdapNorm_attention_flag == '1layer':
            for k in params:
                if 'weight' in k:
                    AttentionNet_weight = params[k]
                elif 'bias' in k:
                    AttentionNet_bias = params[k]
            AdapNorm_factor = F.linear(z, AttentionNet_weight, AttentionNet_bias)

        elif self.AdapNorm_attention_flag == '2layer':
            for k in params:
                if 'weight' in k:
                    if 'fc1' in k:
                        AttentionNet_fc1_weight = params[k]
                    if 'fc2' in k:
                        AttentionNet_fc2_weight = params[k]
                elif 'bias' in k:
                    if 'fc1' in k:
                        AttentionNet_fc1_bias = params[k]
                    if 'fc2' in k:
                        AttentionNet_fc2_bias = params[k]
            AdapNorm_factor = F.linear(z, AttentionNet_fc1_weight, AttentionNet_fc1_bias)
            AdapNorm_factor = F.relu(AdapNorm_factor)
            AdapNorm_factor = F.linear(AdapNorm_factor, AttentionNet_fc2_weight, AttentionNet_fc2_bias)

        return AdapNorm_factor

    def forward(self, x, params=None):
        x = self.conv(x)
        if not self.AdapNorm:
            x = self.BN(x)
        else:
            if self.AdapNorm_attention_flag is not None:
                x_BN = self.BN(x)
                x_IN = self.IN(x)
                b, c, _, _ = x.size()
                z = self.global_pool(x).view(b, c)
                if params is None:
                    if self.AdapNorm_attention_flag == '1layer':
                        AdapNorm_factor = self.AttentionNet(z)
                    elif self.AdapNorm_attention_flag == '2layer':
                        AdapNorm_factor = self.AttentionNet_fc1(z)
                        AdapNorm_factor = F.relu(AdapNorm_factor)
                        AdapNorm_factor = self.AttentionNet_fc2(AdapNorm_factor)
                else:
                    AdapNorm_factor = self.load_params(z, params)

                AdapNorm_factor = self.sigmoid(AdapNorm_factor)
                if AdapNorm_factor.shape[1] == 1:
                    AdapNorm_factor = AdapNorm_factor.expand(b, c)
                AdapNorm_factor = AdapNorm_factor.view(b, c, 1, 1)

                x = AdapNorm_factor * x_BN + (1 - AdapNorm_factor) * x_IN

            else:
                x_BN = self.BN(x)
                x_IN = self.IN(x)
                shape1d = (1, x_bn.shape[1], 1, 1)
                if params is None:
                    AdapNorm_factor = self.gate_weight.clamp(0, 1)
                    x = AdapNorm_factor.view(shape1d) * x_BN + (1 - AdapNorm_factor.view(shape1d)) * x_IN

                else:
                    for k in params:
                        if 'gate_weight' in k:
                            self.gate_weight = params[k]
                        elif 'gate_bias' in k:
                            self.gate_bias = params[k]
                    AdapNorm_factor = self.gate_weight.clamp(0, 1)
                    x = AdapNorm_factor.view(shape1d) * x_BN + (1 - AdapNorm_factor.view(shape1d)) * x_IN

        x = self.relu(x)

        return x


class Basic_block(nn.Module):

    def __init__(self, in_ch, out_ch, AdapNorm=True, AdapNorm_attention_flag='1layer', model_initial='kaiming'):
        '''
            Basic_block contains three Conv_block

            Args:
                in_ch (int): the channel numbers of input features
                out_ch (int): the channel numbers of output features
                AdapNorm (bool): 
                    'True' allow the Conv_block to combine BN and IN
                    'False' allow the Conv_block to use BN
                AdapNorm_attention_flag:
                    '1layer' allow the Conv_block to use 1layer FC to generate the balance factor
                    '2layer' allow the Conv_block to use 2layer FC to generate the balance factor
                model_initial:
                    'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
        '''

        super(Basic_block, self).__init__()
        self.AdapNorm = AdapNorm
        self.AdapNorm_attention_flag = AdapNorm_attention_flag
        self.model_initial = model_initial

        self.conv_block1 = Conv_block(in_ch, 128, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.conv_block2 = Conv_block(128, 196, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.conv_block3 = Conv_block(196, out_ch, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)

        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x, params=None):
        if params is not None:
            params_conv_block1 = {}
            params_conv_block2 = {}
            params_conv_block3 = {}
            for k in params:
                if 'conv_block1' in k:
                    params_conv_block1[k] = params[k]
                elif 'conv_block2' in k:
                    params_conv_block2[k] = params[k]
                elif 'conv_block3' in k:
                    params_conv_block3[k] = params[k]

            x = self.conv_block1(x, params_conv_block1)
            x = self.conv_block2(x, params_conv_block2)
            x = self.conv_block3(x, params_conv_block3)
        else:
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
        x = self.max_pool(x)
        return x
