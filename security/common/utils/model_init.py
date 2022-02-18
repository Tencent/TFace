import torch
from torch.nn import init


def weights_init_xavier(m):
    ''' Xavier initialization '''
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchInstanceNorm2d') != -1:
        init.uniform(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm2d') != -1:
        init.uniform(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)
    else:
        pass


def weights_init_normal(m):
    ''' Normal initialization '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    ''' Kaiming initialization '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    ''' Orthogonal initialization '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal_rnn(m):
    ''' Orthogonal_RNN initialization '''
    classname = m.__class__.__name__
    if classname.find('LSTM') != -1:
        init.orthogonal_(m.all_weights[0][0], gain=1)
        init.orthogonal_(m.all_weights[0][1], gain=1)
        init.constant_(m.all_weights[0][2], 1)
        init.constant_(m.all_weights[0][3], 1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    elif init_type == 'orthogonal_rnn':
        net.apply(weights_init_orthogonal_rnn)
    elif init_type == 'const':
        net.apply(weights_init_const)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def init_model(net, restore, init_type, init=True):
    """Init models with cuda and weights."""
    # init weights of model
    if init:
        init_weights(net, init_type)

    # restore model weights
    if restore is not None:
        if os.path.exists(restore):

            # original saved file with DataParallel
            state_dict = torch.load(restore)
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module' in k:
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            # load params
            net.load_state_dict(new_state_dict)

            net.restored = True
            print("*************Restore model from: {}".format(os.path.abspath(restore)))
        else:
            # raise ValueError('the path ' + restore +' does not exist')
            print('the path ' + restore + ' does not exist')
    print('init model')

    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net
