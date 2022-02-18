import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np


model_urls = {
    'scnet50_v1d': 'https://backseason.oss-cn-beijing.aliyuncs.com/scnet/scnet50_v1d-4109d1e1.pth',
}


class ISM_Module(nn.Module):
    def __init__(self, k_size=3):
        """The Information Supplement Module (ISM).

        Args:
            k_size (int, optional): Conv1d kernel_size . Defaults to 3.
        """
        super(ISM_Module, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        """
        Args:
            x (torch.tensor): Input tensor of shape (nt, c, h, w)
        """
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class TIM_Module(nn.Module):
    def __init__(self, in_channels, reduction=16, n_segment=8, return_attn=False):
        """The Temporal Inconsistency Module (TIM).

        Args:
            in_channels (int): Input channel number.
            reduction (int, optional): Channel compression ratio r in the split operation.. Defaults to 16.
            n_segment (int, optional): Number of input frames.. Defaults to 8.
            return_attn (bool, optional): Whether to return the attention part. Defaults to False.

        """
        super(TIM_Module, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.n_segment = n_segment
        self.return_attn = return_attn

        self.reduced_channels = self.in_channels // self.reduction

        # first conv to shrink input channels
        self.conv1 = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.reduced_channels)

        self.conv_ht = nn.Conv2d(self.reduced_channels, self.reduced_channels, 
            kernel_size=(3, 1), padding=(1, 0), groups=self.reduced_channels, bias=False)
        self.conv_tw = nn.Conv2d(self.reduced_channels, self.reduced_channels, 
            kernel_size=(1, 3), padding=(0, 1), groups=self.reduced_channels, bias=False)

        self.avg_pool_ht = nn.AvgPool2d((2, 1), (2, 1))
        self.avg_pool_tw = nn.AvgPool2d((1, 2), (1, 2))

        # HTIE in two directions
        self.htie_conv1 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.vtie_conv1 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.htie_conv2 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.vtie_conv2 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.ht_up_conv = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels)
        )
        self.tw_up_conv = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels)
        )

        self.sigmoid = nn.Sigmoid()


    def feat_ht(self, feat):
        """The H-T branch in the TIM module.

        Args:
            feat (torch.tensor): Input feature with shape [n, t, c, h, w] (c is in_channels // reduction)

        """
        n, t, c, h, w = feat.size()
        # [n, t, c, h, w] -> [n, w, c, h, t] -> [nw, c, h, t]
        feat_h = feat.permute(0, 4, 2, 3, 1).contiguous().view(-1, c, h, t)

        # [nw, c, h, t-1]
        feat_h_fwd, _ = feat_h.split([self.n_segment-1, 1], dim=3)
        feat_h_conv = self.conv_ht(feat_h)
        _, feat_h_conv_fwd = feat_h_conv.split([1, self.n_segment-1], dim=3)

        diff_feat_fwd = feat_h_conv_fwd - feat_h_fwd
        diff_feat_fwd = F.pad(diff_feat_fwd, [0, 1], value=0)  # [nw, c, h, t]

        # HTIE, down_up branch
        diff_feat_fwd1 = self.avg_pool_ht(diff_feat_fwd)  # [nw, c, h//2, t]
        diff_feat_fwd1 = self.htie_conv1(diff_feat_fwd1)  # [nw, c, h//2, t]
        diff_feat_fwd1 = F.interpolate(diff_feat_fwd1, diff_feat_fwd.size()[2:]) # [nw, c, h, t]
        # HTIE, direct conv branch
        diff_feat_fwd2 = self.htie_conv2(diff_feat_fwd)  # [nw, c, h, t]
        
        # [nw, C, h, t]
        feat_ht_out = self.ht_up_conv(1/3. * diff_feat_fwd + 1/3. * diff_feat_fwd1 + 1/3. * diff_feat_fwd2)
        feat_ht_out = self.sigmoid(feat_ht_out) - 0.5
        # [nw, C, h, t] -> [n, w, C, h, t] -> [n, t, C, h, w]
        feat_ht_out = feat_ht_out.view(n, w, self.in_channels, h, t).permute(0, 4, 2, 3, 1).contiguous()
        # [n, t, C, h, w] -> [nt, C, h, w]
        feat_ht_out = feat_ht_out.view(-1, self.in_channels, h, w)

        return feat_ht_out


    def feat_tw(self, feat):
        """The T-W branch in the TIM module.

        Args:
            feat (torch.tensor): Input feature with shape [n, t, c, h, w] (c is in_channels // reduction)
        """
        n, t, c, h, w = feat.size()
        # [n, t, c, h, w] -> [n, h, c, t, w] -> [nh, c, t, w]
        feat_w = feat.permute(0, 3, 2, 1, 4).contiguous().view(-1, c, t, w)

        # [nh, c, t-1, w]
        feat_w_fwd, _ = feat_w.split([self.n_segment-1, 1], dim=2)
        feat_w_conv = self.conv_tw(feat_w)
        _, feat_w_conv_fwd = feat_w_conv.split([1, self.n_segment-1], dim=2)

        diff_feat_fwd = feat_w_conv_fwd - feat_w_fwd
        diff_feat_fwd = F.pad(diff_feat_fwd, [0, 0, 0, 1], value=0)  # [nh, c, t, w]

        # VTIE, down_up branch
        diff_feat_fwd1 = self.avg_pool_tw(diff_feat_fwd)  # [nh, c, t, w//2]
        diff_feat_fwd1 = self.vtie_conv1(diff_feat_fwd1)  # [nh, c, t, w//2]
        diff_feat_fwd1 = F.interpolate(diff_feat_fwd1, diff_feat_fwd.size()[2:]) # [nh, c, t, w]
        # VTIE, direct conv branch
        diff_feat_fwd2 = self.vtie_conv2(diff_feat_fwd)  # [nh, c, t, w]

        # [nh, C, t, w]
        feat_tw_out = self.tw_up_conv(1/3. * diff_feat_fwd + 1/3. * diff_feat_fwd1 + 1/3. * diff_feat_fwd2)
        feat_tw_out = self.sigmoid(feat_tw_out) - 0.5
        # [nh, C, t, w] -> [n, h, C, t, w] -> [n, t, C, h, W]
        feat_tw_out = feat_tw_out.view(n, h, self.in_channels, t, w).permute(0, 3, 2, 1, 4).contiguous()
        # [n, t, C, h, w] -> [nt, C, h, w]
        feat_tw_out = feat_tw_out.view(-1, self.in_channels, h, w)

        return feat_tw_out


    def forward(self, x):
        """
        Args:
            x (torch.tensor): Input with shape [nt, c, h, w]
        """
        # [nt, c, h, w] -> [nt, c//r, h, w]
        bottleneck = self.conv1(x)
        bottleneck = self.bn1(bottleneck)
        # [nt, c//r, h, w] -> [n, t, c//r, h, w]
        bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])

        F_h = self.feat_ht(bottleneck)  # [nt, c, h, w]
        F_w = self.feat_tw(bottleneck)  # [nt, c, h, w]

        att = 0.5 * (F_h + F_w)

        if self.return_attn:
            return att

        y2 = x + x * att

        return y2


class ShiftModule(nn.Module):
    def __init__(self, input_channels, n_segment=8, n_div=8, mode='shift'):
        """A depth-wise conv on the segment level.

        Args:
            input_channels (int): Input channel number.
            n_segment (int, optional): Number of input frames.. Defaults to 8.
            n_div (int, optional): How many channels to group as a fold.. Defaults to 8.
            mode (str, optional): One of "shift", "fixed", "norm". Defaults to 'shift'.
        """
        super(ShiftModule, self).__init__()
        self.input_channels = input_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(self.fold_div*self.fold, self.fold_div*self.fold,
                kernel_size=3, padding=1, groups=self.fold_div*self.fold,
                bias=False)

        if mode == 'shift':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            # shift left
            self.conv.weight.data[:self.fold, 0, 2] = 1 
            # shift right
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 
            if 2*self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True


    def forward(self, x):
        """
        Args:
            x (torch.tensor): Input with shape [nt, c, h, w]
        """
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        # (n, h, w, c, t)
        x = x.permute(0, 3, 4, 2, 1) 
        x = x.contiguous().view(n_batch*h*w, c, self.n_segment)
        # (n*h*w, c, t)
        x = self.conv(x) 
        x = x.view(n_batch, h, w, c, self.n_segment)
        # (n, t, c, h, w)
        x = x.permute(0, 4, 3, 1, 2) 
        x = x.contiguous().view(nt, c, h, w)
        return x


class SCConv(nn.Module):
    """
    The spatial conv in SIM. Used in SCBottleneck
    """
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv, self).__init__()
        self.f_w = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv2d(inplanes, planes, kernel_size=(1,3), stride=1,
                                padding=(0,padding), dilation=(1,dilation),
                                groups=groups, bias=False),
                    norm_layer(planes), nn.ReLU(inplace=True))
        self.f_h = nn.Sequential(
                    # nn.AvgPool2d(kernel_size=(pooling_r,1), stride=(pooling_r,1)), 
                    nn.Conv2d(inplanes, planes, kernel_size=(3,1), stride=1,
                                padding=(padding,0), dilation=(dilation,1),
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )


    def forward(self, x):
        identity = x

        # sigmoid(identity + k2)
        out = torch.sigmoid(
            torch.add(
                identity, 
                F.interpolate(self.f_h(self.f_w(x)), identity.size()[2:])
            )
        ) 
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        s2t_info = out
        out = self.k4(out) # k4

        return out, s2t_info


class SCBottleneck(nn.Module):
    """
    SCNet SCBottleneck. Variant for ResNet Bottlenect.
    """
    expansion = 4
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=None):
        super(SCBottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)
        self.tim = TIM_Module(group_width, n_segment=num_segments) 
        self.shift = ShiftModule(group_width, n_segment=num_segments, n_div=8, mode='shift')
        self.inplanes = inplanes
        self.planes = planes
        self.ism = ISM_Module()
        self.shift = ShiftModule(group_width, n_segment=num_segments, n_div=8, mode='shift')

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        groups=cardinality, bias=False),
                    norm_layer(group_width),
                    )

        self.scconv = SCConv(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)

        self.conv3 = nn.Conv2d(
            group_width * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation


    def forward(self, x):
        """Forward func which splits the input into two branchs a and b.
        a: trace features
        b: spatial features
        """
        residual = x

        out_a = self.relu(self.bn1_a(self.conv1_a(x)))
        out_b = self.relu(self.bn1_b(self.conv1_b(x)))

        # spatial representations
        out_b, s2t_info = self.scconv(out_b)
        out_b = self.relu(out_b)

        # trace features
        out_a = self.tim(out_a)
        out_a = self.shift(out_a + self.ism(s2t_info))
        out_a = self.relu(self.k1(out_a))

        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SCNet(nn.Module):
    def __init__(self, num_segments, block, layers, groups=1, bottleneck_width=32,
                 num_classes=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 avd=False, norm_layer=nn.BatchNorm2d):
        """SCNet, a variant based on ResNet.

        Args:
            num_segments (int): 
                Number of input frames.
            block (class): 
                Class for the residual block.
            layers (list): 
                Number of layers in each block.
            num_classes (int, optional): 
                Number of classification class.. Defaults to 1000.
            dilated (bool, optional): 
                Whether to apply dilation conv. Defaults to False.
            dilation (int, optional): 
                The dilation parameter in dilation conv. Defaults to 1.
            deep_stem (bool, optional): 
                Whether to replace 7x7 conv in input stem with 3 3x3 conv. Defaults to False.
            stem_width (int, optional): 
                Stem width in conv1 stem. Defaults to 64.
            avg_down (bool, optional): 
                Whether to use AvgPool instead of stride conv when downsampling in the bottleneck. Defaults to False.
            avd (bool, optional): 
                The avd parameter for the block Defaults to False.
            norm_layer (class, optional): 
                Normalization layer. Defaults to nn.BatchNorm2d.
        """
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.avd = avd
        self.num_segments = num_segments

        super(SCNet, self).__init__()
        conv_layer = nn.Conv2d
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer)
        elif dilation==2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    is_first=True):
        """
        Core function to build layers.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.num_segments, self.inplanes, planes, stride, downsample=downsample,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=1, is_first=is_first, 
                                norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.num_segments, self.inplanes, planes, stride, downsample=downsample,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=2, is_first=is_first, 
                                norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.num_segments, self.inplanes, planes,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=dilation, 
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def scnet50_v1d(num_segments, pretrained=False, **kwargs):
    """
    SCNet backbone, which is based on ResNet-50
    Args:
        num_segments (int):
            Number of input frames.
        pretrained (bool, optional):
            Whether to load pretrained weights.
    """
    model = SCNet(num_segments, SCBottleneck, [3, 4, 6, 3],
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['scnet50_v1d']), strict=False)

    return model
