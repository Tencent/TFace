import os
import sys
import torch
from torch import nn
import torch.nn.functional as F

from .base_block import Conv_block, Basic_block

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..'))
from common.utils.model_init import init_weights


class FeatExtractor(nn.Module):
    '''
        Args:
            in_ch (int): the channel numbers of input features
            AdapNorm (bool): 
                'True' allow the Conv_block to combine BN and IN
                'False' allow the Conv_block to use BN
            AdapNorm_attention_flag:
                '1layer' allow the Conv_block to use 1layer FC to generate the balance factor
                '2layer' allow the Conv_block to use 2layer FC to generate the balance factor
            model_initial:
                'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
    '''

    def __init__(self, in_ch=6, AdapNorm=True, AdapNorm_attention_flag='1layer', model_initial='kaiming'):
        super(FeatExtractor, self).__init__()
        self.AdapNorm = AdapNorm
        self.AdapNorm_attention_flag = AdapNorm_attention_flag
        self.model_initial = model_initial

        self.inc = Conv_block(in_ch, 64, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.down1 = Basic_block(64, 128, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.down2 = Basic_block(128, 128, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.down3 = Basic_block(128, 128, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)

    def forward(self, x, params=None):
        if params is not None:
            params_inc = {}
            params_down1 = {}
            params_down2 = {}
            params_down3 = {}
            for k in params:
                if 'inc' in k:
                    params_inc[k] = params[k]
                elif 'down1' in k:
                    params_down1[k] = params[k]
                elif 'down2' in k:
                    params_down2[k] = params[k]
                elif 'down3' in k:
                    params_down3[k] = params[k]
                else:
                    pass
            dx1 = self.inc(x, params_inc)
            dx2 = self.down1(dx1, params_down1)
            dx3 = self.down2(dx2, params_down2)
            dx4 = self.down3(dx3, params_down3)
        else:
            dx1 = self.inc(x)
            dx2 = self.down1(dx1)
            dx3 = self.down2(dx2)
            dx4 = self.down3(dx3)

        re_dx2 = F.adaptive_avg_pool2d(dx2, 32)
        re_dx3 = F.adaptive_avg_pool2d(dx3, 32)
        catfeat = torch.cat([re_dx2, re_dx3, dx4], 1)

        return catfeat, dx4


class FeatEmbedder(nn.Module):
    '''
        Args:
            in_ch (int): the channel numbers of input features
            AdapNorm (bool): 
                'True' allow the Conv_block to combine BN and IN
                'False' allow the Conv_block to use BN
            AdapNorm_attention_flag:
                '1layer' allow the Conv_block to use 1layer FC to generate the balance factor
                '2layer' allow the Conv_block to use 2layer FC to generate the balance factor
            model_initial:
                'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
    '''

    def __init__(self, in_ch=384, AdapNorm=True, AdapNorm_attention_flag='1layer', model_initial='kaiming'):
        super(FeatEmbedder, self).__init__()
        self.AdapNorm = AdapNorm
        self.AdapNorm_attention_flag = AdapNorm_attention_flag
        self.model_initial = model_initial

        self.conv_block1 = Conv_block(in_ch, 128, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.conv_block2 = Conv_block(128, 256, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.conv_block3 = Conv_block(256, 512, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.max_pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 2)

        # model initial
        init_weights(self.fc, init_type=self.model_initial)

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
                else:
                    pass
            x = self.conv_block1(x, params_conv_block1)
            x = self.max_pool(x)
            x = self.conv_block2(x, params_conv_block2)
            x = self.max_pool(x)
            x = self.conv_block3(x, params_conv_block3)
        else:
            x = self.conv_block1(x)
            x = self.max_pool(x)
            x = self.conv_block2(x)
            x = self.max_pool(x)
            x = self.conv_block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


class DepthEstmator(nn.Module):
    '''
        Args:
            in_ch (int): the channel numbers of input features
            AdapNorm (bool): 
                'True' allow the Conv_block to combine BN and IN
                'False' allow the Conv_block to use BN
            AdapNorm_attention_flag:
                '1layer' allow the Conv_block to use 1layer FC to generate the balance factor
                '2layer' allow the Conv_block to use 2layer FC to generate the balance factor
            model_initial:
                'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
    '''

    def __init__(self, in_ch=384, AdapNorm=True, AdapNorm_attention_flag='1layer', model_initial='kaiming'):
        super(DepthEstmator, self).__init__()
        self.AdapNorm = AdapNorm
        self.AdapNorm_attention_flag = AdapNorm_attention_flag
        self.model_initial = model_initial

        self.conv_block1 = Conv_block(in_ch, 128, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.conv_block2 = Conv_block(128, 64, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)
        self.conv_block3 = Conv_block(64, 1, self.AdapNorm, self.AdapNorm_attention_flag, self.model_initial)

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
                else:
                    pass
            x = self.conv_block1(x, params_conv_block1)
            x = self.conv_block2(x, params_conv_block2)
            x = self.conv_block3(x, params_conv_block3)
        else:
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
        return x


class Framework(nn.Module):
    '''
        Framework contains all the modules

        Args:
            in_ch (int): the channel numbers of input features
            mid_ch (int): the channel numbers of output features in FeatExtractor
            AdapNorm (bool): 
                'True' allow the Conv_block to combine BN and IN
                'False' allow the Conv_block to use BN
            AdapNorm_attention_flag:
                '1layer' allow the Conv_block to use 1layer FC to generate the balance factor
                '2layer' allow the Conv_block to use 2layer FC to generate the balance factor
            model_initial:
                'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
    '''

    def __init__(self, in_ch=6, mid_ch=384, AdapNorm=True, AdapNorm_attention_flag='1layer', model_initial='kaiming'):

        super(Framework, self).__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.AdapNorm = AdapNorm
        self.AdapNorm_attention_flag = AdapNorm_attention_flag
        self.model_initial = model_initial

        self.FeatExtractor = FeatExtractor(in_ch=self.in_ch,
                                           AdapNorm=self.AdapNorm,
                                           AdapNorm_attention_flag=self.AdapNorm_attention_flag,
                                           model_initial=self.model_initial)

        self.Classifier = FeatEmbedder(in_ch=self.mid_ch, AdapNorm=False)
        self.DepthEstmator = DepthEstmator(in_ch=self.mid_ch, AdapNorm=False)

    def forward(self, x, param=None):
        x, _ = self.FeatExtractor(x, param)
        y = self.Classifier(x)
        depth = self.DepthEstmator(x)
        return y, depth, x
