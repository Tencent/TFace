import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from einops import rearrange

from utils import *


class M3DFEL(nn.Module):
    """The proposed M3DFEL framework

    Args:
        args
    """

    def __init__(self, args):
        super(M3DFEL, self).__init__()

        self.args = args
        self.device = torch.device(
            'cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')
        self.bag_size = self.args.num_frames // self.args.instance_length
        self.instance_length = self.args.instance_length

        # backbone networks
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.features = nn.Sequential(
            *list(model.children())[:-1])  # after avgpool 512x1
        self.lstm = nn.LSTM(input_size=512, hidden_size=512,
                            num_layers=2, batch_first=True, bidirectional=True)

        # multi head self attention
        self.heads = 8
        self.dim_head = 1024 // self.heads
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(
            1024, (self.dim_head * self.heads) * 3, bias=False)

        self.norm = DMIN(num_features=1024)
        self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)

        # classifier
        self.fc = nn.Linear(1024, self.args.num_classes)
        self.Softmax = nn.Softmax(dim=-1)

    def MIL(self, x):
        """The Multi Instance Learning Agregation of instances

        Inputs:
            x: [batch, bag_size, 512]
        """
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        # [batch, bag_size, 1024]
        ori_x = x

        # MHSA
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.norm(x)
        x = torch.sigmoid(x)

        x = ori_x * x

        return x

    def forward(self, x):

        # [batch, 16, 3, 112, 112]
        x = rearrange(x, 'b (t1 t2) c h w -> (b t1) c t2 h w',
                      t1=self.bag_size, t2=self.instance_length)
        # [batch*bag_size, 3, il, 112, 112]

        x = self.features(x).squeeze()
        # [batch*bag_size, 512]
        x = rearrange(x, '(b t) c -> b t c', t=self.bag_size)

        # [batch, bag_size, 512]
        x = self.MIL(x)
        # [batch, bag_size, 1024]

        x = self.pwconv(x).squeeze()
        # [batch, 1024]
        out = self.fc(x)
        # [batch, 7]

        return out
