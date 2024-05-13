import torch
import torch.nn as nn
import torch.nn.functional as F
from torchjpeg import dct

def images_to_batch(x):
    x = (x + 1) / 2 * 255
    x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
    if x.shape[1] != 3:
        print("Wrong input, Channel should equals to 3")
        return
    x = dct.to_ycbcr(x)  # comvert RGB to YCBCR
    x -= 128
    bs, ch, h, w = x.shape
    block_num = h // 8
    x = x.view(bs * ch, 1, h, w)
    x = F.unfold(x, kernel_size=(8, 8), dilation=1, padding=0,
                 stride=(8, 8))
    x = x.transpose(1, 2)
    x = x.view(bs, ch, -1, 8, 8)
    dct_block = dct.block_dct(x)
    dct_block = dct_block.view(bs, ch, block_num, block_num, 64).permute(0, 1, 4, 2, 3)
    dct_block = dct_block[:, :, 1:, :, :]  # remove DC
    dct_block = dct_block.reshape(bs, -1, block_num, block_num)
    return dct_block


class NoisyActivation(nn.Module):
    def __init__(self, input_shape=112, budget_mean=4, sensitivity=None):
        super(NoisyActivation, self).__init__()
        self.h, self.w = input_shape, input_shape
        if sensitivity is None:
            sensitivity = torch.ones([189, self.h, self.w]).cuda()
        self.sensitivity = sensitivity.reshape(189 * self.h * self.w)
        self.given_locs = torch.zeros((189, self.h, self.w))
        size = self.given_locs.shape
        self.budget = budget_mean * 189 * self.h * self.w
        self.locs = nn.Parameter(torch.Tensor(size).copy_(self.given_locs))
        self.rhos = nn.Parameter(torch.zeros(size))
        self.laplace = torch.distributions.laplace.Laplace(0, 1)
        self.rhos.requires_grad = True
        self.locs.requires_grad = True

    def scales(self):
        softmax = nn.Softmax()
        return (self.sensitivity / (softmax(self.rhos.reshape(189 * self.h * self.w))
                * self.budget)).reshape(189, self.h, self.w)

    def sample_noise(self):
        epsilon = self.laplace.sample(self.rhos.shape).cuda()
        return self.locs + self.scales() * epsilon

    def forward(self, input):
        noise = self.sample_noise()
        output = input + noise
        return output

    def aux_loss(self):
        scale = self.scales()
        loss = -1.0 * torch.log(scale.mean())
        return loss
