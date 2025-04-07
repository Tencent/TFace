# based on:
# https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/backbone/model_irse.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (Linear, Conv2d, BatchNorm1d, BatchNorm2d, 
                      PReLU, Dropout, MaxPool2d, Sequential, Module
)

from torchkit.backbone.common import initialize_weights, Flatten, SEModule
from torchkit.backbone.model_irse import (
    BasicBlockIR, BottleneckIR, BasicBlockIRSE, BottleneckIRSE, get_blocks
)

def normal_init(m, mean, std):
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class XCosAttention(nn.Module):
    def __init__(self, group_size, use_softmax=True, softmax_t=1, chw2hwc=True):
        super(XCosAttention, self).__init__()
        self.embedding_net = nn.Sequential(
            nn.Conv2d(group_size, group_size, 3, padding=1),
            nn.BatchNorm2d(group_size),
            nn.PReLU())
        self.attention = nn.Sequential(
            nn.Conv2d(group_size, group_size, 3, padding=1),
            nn.BatchNorm2d(group_size),
            nn.PReLU(),
            nn.Conv2d(group_size, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )
        self.USE_SOFTMAX = use_softmax
        self.SOFTMAX_T = softmax_t
        self.chw2hwc = chw2hwc

    def softmax(self, x, T=1):
        x /= T
        return F.softmax(x.reshape(x.size(0), x.size(1), -1), 2).view_as(x)

    def divByNorm(self, x):
        x -= x.view(x.size(0), x.size(1), -1).min(dim=2)[0].repeat(1, 1, x.size(2) * x.size(3)).view(x.size(0), x.size(1), x.size(2), x.size(3))
        x /= x.view(x.size(0), x.size(1), -1).sum(dim=2).repeat(1, 1, x.size(2) * x.size(3)).view(x.size(0), x.size(1), x.size(2), x.size(3))
        return x

    def forward(self, feat_grid):
        proj_embed = self.embedding_net(feat_grid)
        attention_weights = self.attention(proj_embed)
        if self.USE_SOFTMAX:
            attention_weights = self.softmax(attention_weights, self.SOFTMAX_T)
        else:
            attention_weights = self.divByNorm(attention_weights)
        if self.chw2hwc:
            attention_weights = attention_weights.permute(0, 2, 3, 1)
        return attention_weights

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class SlerpFace(nn.Module):
    def __init__(self, input_size, num_layers, mode='ir', group_size=16):
        super(SlerpFace, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [18, 34, 50, 100, 152, 200], "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"

        # 初始化输入层
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        # get blocks
        blocks = get_blocks(num_layers)
        if num_layers <= 100:
            unit_module = BasicBlockIRSE if mode == 'ir_se' else BasicBlockIR
            output_channel = 512
        else:
            unit_module = BottleneckIRSE if mode == 'ir_se' else BottleneckIR
            output_channel = 2048

        # construct body
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride))
        self.body = nn.Sequential(*modules)

        # output layer
        if input_size[0] == 112:
            self.output_layer = nn.Sequential(
                nn.BatchNorm2d(output_channel),
                nn.Dropout(0.4),
                Flatten(),
                nn.Linear(output_channel * 7 * 7, 512),
                nn.BatchNorm1d(512, affine=False)
            )
        else:
            self.output_layer = nn.Sequential(
                nn.BatchNorm2d(output_channel),
                nn.Dropout(0.4),
                Flatten(),
                nn.Linear(output_channel * 14 * 14, 512),
                nn.BatchNorm1d(512, affine=False)
            )

        # Xcos module
        self.group_cov = nn.Sequential(
            nn.Conv2d(512, group_size, (1, 1), 1, 0),
            nn.BatchNorm2d(group_size),
            nn.PReLU(group_size)
        )
        self.group_attention = XCosAttention(group_size=group_size, use_softmax=True, softmax_t=1, chw2hwc=True)

        # initialize weights
        initialize_weights(self.modules())

    def get_body_output(self, x):
        x = self.input_layer(x)
        return self.body(x)

    def forward(self, x, flip=True):
        x_body = self.get_body_output(x)
        x_body_cov = self.group_cov(x_body)
        x_vector = self.output_layer(x_body)
        
        if flip:
            x_flip = torch.flip(x, dims=[3])
            flip_body = self.get_body_output(x_flip)
            flip_body_cov = self.group_cov(flip_body)
            return x_vector, x_body_cov + flip_body_cov
        return x_vector, x_body_cov

    def gen_weight_map(self, x):
        return self.group_attention(x)

    def gen_vector_feature(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        return self.output_layer(x)

    def gen_group_feature(self, x, flip=True):
        x_body = self.get_body_output(x)
        x_cov = self.group_cov(x_body)
        if flip:
            x_flip = torch.flip(x, dims=[3])
            flip_body = self.get_body_output(x_flip)
            flip_cov = self.group_cov(flip_body)
            return x_cov + flip_cov
        return x_cov

if __name__ == "__main__":
    backbone = SlerpFace(
        input_size=[112, 112],
        num_layers=50,
        group_size=16
    ).cuda()
    x = torch.randn(1, 3, 112, 112).cuda()
    group_feature = backbone.gen_group_feature(x)
    print(group_feature.shape)