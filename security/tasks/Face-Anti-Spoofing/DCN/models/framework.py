import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

from .base_block import Conv2dBlock

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..'))
from common.utils.model_init import init_weights


class Encoder(nn.Module):
    '''
        Encoder extract the features map of the inputs
        Args:
            in_ch (int): the channel numbers of input features
    '''
    def __init__(self, in_ch):
        super(Encoder, self).__init__()
        self.net = []
        self.cnn = nn.Conv2d(in_ch,64,3,1,1,bias=True)
        self.net += [self.cnn]       
        self.net += [Conv2dBlock(First_block=1)]
        self.net += [Conv2dBlock(First_block=0)]
        self.net += [Conv2dBlock(First_block=0)]

        self.net = nn.Sequential(*self.net)
        init_weights(self.cnn,init_type='kaiming')
    
    def forward(self, x):
        code = []
        for idx, layer in enumerate(self.net):
            x = layer(x)
            if idx in [1,2,3]:
                code.append(x)
        code = self.resize_concate(code,32)
        return x,code

    def resize_concate(self,feature,size):
        new_feature = torch.zeros((feature[0].shape[0],0,size,size))
        new_feature = new_feature.to(feature[0].device)

        for idx,i in enumerate(feature):
            origin_size = i.shape[-1]
            i = F.interpolate(i, scale_factor=(size/origin_size))
            new_feature = torch.cat((new_feature,i),1)
        return new_feature

class Auxilary_Deep(nn.Module):
    '''
        Auxilary_Deep calcuate the depth maps of the inputs
        Args:
            in_ch (int): the channel numbers of input features
    '''
    def __init__(self, in_ch):
        super(Auxilary_Deep,self).__init__()
        self.in_ch = in_ch
        self.model=[]
        self.model+=[nn.Conv2d(self.in_ch, 256, kernel_size=3, stride=1, padding=1)]
        self.model+=[nn.BatchNorm2d(256)]
        self.model+=[nn.ReLU()]

        self.model+=[nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)]
        self.model+=[nn.BatchNorm2d(128)]
        self.model+=[nn.ReLU()]

        self.model+=[nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)]
        self.model+=[nn.BatchNorm2d(64)]
        self.model+=[nn.ReLU()]

        self.model+=[nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)]
        self.model+=[nn.BatchNorm2d(1)]
        self.model+=[nn.ReLU()]

        self.model = nn.Sequential(*self.model)
        init_weights(self.model,init_type='kaiming')

    def forward(self,x):
        return self.model(x)

class Similarity(nn.Module):
    '''
        Similarity calcuate the similarity between the inputs
        Args:
            in_ch (int): the channel numbers of input features
    '''
    def __init__(self,in_ch):
        super(Similarity, self).__init__()
        self.in_ch = in_ch
        self.model=[]
        self.model+=[nn.Conv2d(self.in_ch, 128,kernel_size=3, stride=1, padding=1)]
        self.model+=[nn.BatchNorm2d(128)]
        self.model+=[nn.ReLU()]  

        self.model+=[nn.AvgPool2d(kernel_size=2, stride=2)]

        self.model+=[nn.Conv2d(128,64,kernel_size=3, stride=1, padding=1)]
        self.model+=[nn.BatchNorm2d(64)]
        self.model+=[nn.ReLU()]

        self.model+=[nn.AvgPool2d(kernel_size=2, stride=2)]

        self.model+=[nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1)]
        self.model+=[nn.BatchNorm2d(64)]
        self.model+=[nn.ReLU()]

        self.model+=[nn.AvgPool2d(kernel_size=2, stride=2)]

        self.model = nn.Sequential(*self.model)
        init_weights(self.model,init_type='kaiming')

    def forward(self,x):
        x = F.interpolate(x, scale_factor=(24/32))
        x = self.model(x)
        return x

class DCN(nn.Module):
    '''
        DCN contains all the modules
        Args:
            in_ch (int): the channel numbers of input features
    '''
    def __init__(self, in_ch):
        super(DCN, self).__init__()

        self.encoder = Encoder(in_ch)
        self.depth = Auxilary_Deep(in_ch=384)
        self.similarity = Similarity(in_ch=384)

    def forward(self, images):
        x_code, x_skip_code = self.encoder(images)

        dep = self.depth(x_skip_code)
        x_sim = self.similarity(x_skip_code)
        w = F.unfold(x_sim, kernel_size=1, stride=1, padding=0).permute(0, 2, 1)
        w_normed = w / (w * w + 1e-7).sum(dim=2, keepdim=True).sqrt()
        B, K = w.shape[:2]
        sim = torch.einsum('bij,bjk->bik', w_normed, w_normed.permute(0, 2, 1))
        
        return dep,sim
        
if __name__ == '__main__':
    in_channels=6

    model = DCN(in_ch=in_channels)
    x = torch.randn((2,6,256,256))
    depth,sim = model(x)
