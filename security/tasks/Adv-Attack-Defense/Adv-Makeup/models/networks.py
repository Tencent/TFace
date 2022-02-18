import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import functools

'''
Encoder structure configuration
'''
config_enc = [
    ('conv2d', [64, 3, 5, 5, 2, 2]),
    ('lrelu', [0.2, True]),
    ('conv2d', [128, 64, 5, 5, 2, 2]),
    ('bn', [128])
]

'''
Decoder structure configuration
'''
config_dec = [
    ('relu', [True]),
    ('conv2d', [128, 128, 3, 3, 1, 1]),
    ('bn', [128]),
    ('cat1', 1),
    ('relu', [True]),
    ('up', [nn.functional.interpolate, 2, 'bilinear', True]),
    ('conv2d', [64, 256, 3, 3, 1, 1]),
    ('bn', [64]),
    ('cat2', 1),
    ('relu', [True]),
    ('up', [nn.functional.interpolate, 2, 'bilinear', True]),
    ('conv2d', [64, 128, 3, 3, 1, 1]),
    ('bn', [64]),
    ('relu', [True]),
    ('conv2d', [3, 64, 3, 3, 1, 1]),
    ('tanh', [])
]


class Encoder(nn.Module):
    def __init__(self, input_dim, ngf=64):
        super(Encoder, self).__init__()
        self.config_enc = config_enc

        # Init param list of kernels
        self.vars_enc = nn.ParameterList()
        # Init param list of batch-norm
        self.vars_bn_enc = nn.ParameterList()

        for i, (name_enc, param_enc) in enumerate(self.config_enc):
            if name_enc == 'conv2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w_enc = nn.Parameter(torch.ones(*param_enc[:4]))
                # Init the params of weights
                init.kaiming_normal_(w_enc)
                self.vars_enc.append(w_enc)
                # [ch_in, ch_out]
                self.vars_enc.append(nn.Parameter(torch.zeros(param_enc[0])))
            elif name_enc == 'bn':
                # [ch_out]
                w_enc = nn.Parameter(torch.ones(param_enc[0]))
                self.vars_enc.append(w_enc)
                # [ch_out]
                self.vars_enc.append(nn.Parameter(torch.zeros(param_enc[0])))

                # Must set requires_grad=False
                # Init running mean and variance
                running_mean_enc = nn.Parameter(torch.zeros(param_enc[0]), requires_grad=False)
                running_var_enc = nn.Parameter(torch.ones(param_enc[0]), requires_grad=False)
                self.vars_bn_enc.extend([running_mean_enc, running_var_enc])

    '''
    Sequential the network blocks
    '''
    def forward(self, x_enc, vars_enc=None, bn_training=True):
        """
        :param x_enc: input data for inference
        :param vars_enc: params list
        :param bn_training: training flag
        :return: output with diff sz
        """
        if vars_enc is None:
            vars_enc = self.vars_enc

        idx_enc = 0
        bn_idx_enc = 0
        for i, (name_enc, param_enc) in enumerate(self.config_enc):
            if name_enc == 'conv2d':
                # Obtain weights and bias
                w_enc, b_enc = vars_enc[idx_enc], vars_enc[idx_enc + 1]
                x_enc = F.conv2d(x_enc, w_enc, b_enc, stride=param_enc[4], padding=param_enc[5])
                idx_enc += 2
            elif name_enc == 'bn':
                w_enc, b_enc = vars_enc[idx_enc], vars_enc[idx_enc + 1]
                running_mean_enc, running_var_enc = self.vars_bn_enc[bn_idx_enc], self.vars_bn_enc[bn_idx_enc + 1]
                x_enc = F.batch_norm(x_enc, running_mean_enc, running_var_enc,
                                     weight=w_enc, bias=b_enc, training=bn_training)
                idx_enc += 2
                bn_idx_enc += 2
            elif name_enc == 'leakyrelu':
                x_enc = F.leaky_relu(x_enc, negative_slope=param_enc[0], inplace=param_enc[1])
            if i == 0:
                out_1_enc = x_enc
            if i == 3:
                out_2_enc = x_enc

        return out_1_enc, out_2_enc

    def parameters(self):
        return self.vars_enc


class Decoder(nn.Module):
    def __init__(self, output_dim, ngf=64):
        super(Decoder, self).__init__()
        self.config_dec = config_dec

        # Init param list of kernels
        self.vars_dec = nn.ParameterList()
        # Init param list of batch-norm
        self.vars_bn_dec = nn.ParameterList()

        for i, (name_dec, param_dec) in enumerate(self.config_dec):
            if name_dec == 'conv2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w_dec = nn.Parameter(torch.ones(*param_dec[:4]))
                # Init the params of weights
                init.kaiming_normal_(w_dec)
                self.vars_dec.append(w_dec)
                # [ch_in, ch_out]
                self.vars_dec.append(nn.Parameter(torch.zeros(param_dec[0])))
            elif name_dec == 'bn':
                # [ch_out]
                w_dec = nn.Parameter(torch.ones(param_dec[0]))
                self.vars_dec.append(w_dec)
                # [ch_out]
                self.vars_dec.append(nn.Parameter(torch.zeros(param_dec[0])))

                # Must set requires_grad=False
                # Init running mean and variance
                running_mean_dec = nn.Parameter(torch.zeros(param_dec[0]), requires_grad=False)
                running_var_dec = nn.Parameter(torch.ones(param_dec[0]), requires_grad=False)
                self.vars_bn_dec.extend([running_mean_dec, running_var_dec])

    '''
    Sequential the network blocks
    '''
    def forward(self, x1_dec, x2_dec, vars_dec=None, bn_training=True):
        """
        :param x1_dec: input data from encoder output
        :param x2_dec: input data from encoder output
        :param vars_dec: params list
        :param bn_training: training flag
        :return: decoded image
        """
        if vars_dec is None:
            vars_dec = self.vars_dec

        idx_dec = 0
        bn_idx_dec = 0
        x_dec = x2_dec
        for i, (name_dec, param_dec) in enumerate(self.config_dec):
            if name_dec == 'conv2d':
                # Obtain weights and bias
                w_dec, b_dec = vars_dec[idx_dec], vars_dec[idx_dec + 1]
                x_dec = F.conv2d(x_dec, w_dec, b_dec, stride=param_dec[4], padding=param_dec[5])
                idx_dec += 2
            elif name_dec == 'bn':
                w_dec, b_dec = vars_dec[idx_dec], vars_dec[idx_dec + 1]
                running_mean_dec, running_var_dec = self.vars_bn_dec[bn_idx_dec], self.vars_bn_dec[bn_idx_dec + 1]
                x_dec = F.batch_norm(x_dec, running_mean_dec, running_var_dec,
                                     weight=w_dec, bias=b_dec, training=bn_training)
                idx_dec += 2
                bn_idx_dec += 2
            elif name_dec == 'relu':
                x_dec = F.relu(x_dec, inplace=param_dec[0])
            elif name_dec == 'cat1':
                x_dec = torch.cat([d_1_dec, x2_dec], 1)
            elif name_dec == 'cat2':
                x_dec = torch.cat([d_2_dec, x1_dec], 1)
            elif name_dec == 'up':
                # Get the upsampling function
                up_dec = functools.partial(param_dec[0],
                                           scale_factor=param_dec[1], mode=param_dec[2], align_corners=param_dec[3])
                # Upsample the x_dec
                x_dec = up_dec(x_dec)
            elif name_dec == 'tanh':
                x_dec = F.tanh(x_dec)

            if i == 2:
                d_1_dec = x_dec
            if i == 7:
                d_2_dec = x_dec

        out_dec = x_dec

        return out_dec

    def parameters(self):
        return self.vars_dec


class Discriminator(nn.Module):
    def __init__(self, input_dim, ndf=64):
        super(Discriminator, self).__init__()

        model = []
        model += [LeakyReLUConv2d(input_dim, ndf * 2, kernel_size=3, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(ndf * 2, ndf * 2, kernel_size=3, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(ndf * 2, ndf * 2, kernel_size=3, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, padding=0)]
        model += [nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        out = x.view(-1)

        return out


class LeakyReLUConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(inplanes, outplanes,
                                                       kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(inplanes, outplanes,
                                kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(outplanes, affine=False)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def init_weights(net, init_type, gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fainplanes')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, device, init_type='normal', gain=0.02):
    net.to(device)
    init_weights(net, init_type, gain)
    return net

