import torch
import torch.nn as nn
from torchkit.backbone import get_model
from torchkit.loss import get_loss
from tasks.minusface.utils import UNet
from tasks.partialface.utils import dct_transform, idct_transform


class MinusGenerativeModel(nn.Module):
    def __init__(self, mode='stage1', backbone=UNet):
        super().__init__()
        self.mode = mode
        # by default, we use a U-Net as generator
        # it may be replaced with other network architectures
        if self.mode == 'toy':
            self.backbone = backbone(3, 3)
        else:
            self.backbone = backbone(192, 192)

    def forward(self, x):
        if self.mode == 'toy':
            x = self.backbone(x)
            return x
        else:
            x_features = self.backbone.encode(x)
            x = self.backbone.decode(*x_features)
            x_latent = x_features[-1]
            return x, x_latent


class MinusBackbone(nn.Module):
    def __init__(self, mode='stage1', n_duplicate=1, generator=None, recognizer=None):
        super().__init__()
        self.mode = mode

        # by default, we produce X_p as (3, 112, 112)
        # X_p can be enlarged by setting the following n_duplicate to other than 1
        # it equals sample X_p n_duplicate times independently, then concatenate them by channel dimension
        # we experimentally find this practice marginally enhances accuracy (as recognizable features increases)
        # at the cost of some increased transmission overhead
        self.n_duplicate = n_duplicate

        self.generator = MinusGenerativeModel(mode=self.mode, backbone=UNet) if generator is None else generator
        self.recognizer = get_model('IR_18')([112, 112]) if recognizer is None else recognizer

        # the recognition model is modified of only its input channels
        if mode == 'stage1':
            self.recognizer.input_layer = nn.Sequential(nn.Conv2d(192, 64, (3, 3), 1, 1, bias=False),
                                                        nn.BatchNorm2d(64), nn.PReLU(64))
        elif mode == 'stage2':
            self.recognizer.input_layer = nn.Sequential(nn.Conv2d(3 * self.n_duplicate, 64, (3, 3), 1, 1, bias=False),
                                                        nn.BatchNorm2d(64), nn.PReLU(64))

        # the encoder and decoder mappings can be replaced with other differentiable mappings
        # that are together invertible and encoder alone homomorphic
        self.encoder = dct_transform
        self.decoder = idct_transform

    def shuffle(self, x_residue_up):
        """ shuffle high-dimensional residues' channel order
        """
        b, c, h, w = x_residue_up.shape

        x_residue_perms = []
        for i in range(self.n_duplicate):
            permuted_indices = [torch.randperm(c) for _ in range(b)]

            permuted_indices = torch.stack(permuted_indices).to(x_residue_up.device)
            permuted_indices = permuted_indices.view(b, c, 1, 1).expand(-1, -1, h, w)
            perm = self.decoder(torch.gather(x_residue_up, 1, permuted_indices))
            x_residue_perms.append(perm)
        x_residue_shift = torch.cat(x_residue_perms, dim=1)

        return x_residue_shift

    def obtain_residue(self, x):
        x_up = self.encoder(x)
        x_encode_up, x_latent = self.generator(x_up)
        x_residue_up = x_up - x_encode_up

        x_encode = self.decoder(x_encode_up)
        x_residue = self.decoder(x_residue_up)

        # for DCT: image-form inputs have a range of [-1, 1], and inverse DCT produces [0, 1]
        # they must be aligned to calculate L1 loss
        x_encode = x_encode * 2 - 1
        x_residue = x_residue * 2 - 1
        return x_up, x_encode, x_residue, x_latent, x_encode_up, x_residue_up

    def forward(self, x):
        if self.mode == 'toy':
            x_encode = self.generator(x)
            x_residue = x - x_encode
            x_feature = self.recognizer(x_residue)
            return x_encode, x_residue, x_feature

        elif self.mode == 'stage1':
            _, x_encode, x_residue, x_latent, _, x_residue_up = self.obtain_residue(x)
            x_feature = self.recognizer(x_residue_up)

        else:  # stage2
            # freeze generator and optimize recognition model only
            with torch.no_grad():
                _, x_encode, x_residue, x_latent, _, x_residue_up = self.obtain_residue(x)
                x_residue_shuffle = self.shuffle(x_residue_up)
            x_feature = self.recognizer(x_residue_shuffle)

        return x_encode, x_residue, x_feature, x_latent


class KLSparseLoss(nn.Module):
    def __init__(self, sparsity_target=0.05):
        super().__init__()
        self.sparsity_target = sparsity_target

    @staticmethod
    def kl_divergence(p, q):
        q = torch.clamp(q, 1e-8, 1 - 1e-8)
        s1 = torch.sum(p * torch.log(p / q))
        s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
        return s1 + s2

    def forward(self, x):
        x = torch.mean(x, dim=0, keepdim=True)
        x = x / torch.sum(x)
        p = torch.ones_like(x).to(x.device) * self.sparsity_target
        kl = self.kl_divergence(p, x)
        return kl


class MinusLoss(nn.Module):
    def __init__(self, mode='stage1', weights=None):
        super().__init__()
        self.mode = mode
        self.generative_loss = torch.nn.L1Loss()
        self.recognition_loss = get_loss('DistCrossEntropy')

        # by default, the sparsity loss is not used (weight set to 0)
        # it was a KL sparse loss designed to encourage the sparsity of generator's latents
        # we find it could occasionally benefit the quality of residues by setting to a small value (e.g., 1e-3)
        # it is however not used for our paper's results
        self.latent_sparsity_loss = KLSparseLoss(sparsity_target=0.01)

        self.weights = {'gen': 5.0, 'fr': 1.0, 'spar': 0.} if weights is None else weights

    def forward(self, x, x_encode, x_latent, outputs, labels):
        if self.mode == 'toy':
            loss_gen = self.generative_loss(x, x_encode) * self.weights['gen']
            loss_fr = self.recognition_loss(outputs, labels) * self.weights['fr']
            loss = loss_gen + loss_fr
            return loss, loss_gen, loss_fr

        else:
            if self.mode == 'stage2':
                # to train recognition model using learned residue,
                # freeze generator and optimize recognition model only
                self.weights['gen'] = self.weights['spar'] = 0.

            loss_gen = self.generative_loss(x, x_encode) * self.weights['gen']
            loss_fr = self.recognition_loss(outputs, labels) * self.weights['fr']
            loss_ls = self.latent_sparsity_loss(x_latent) * self.weights['spar']

            loss = loss_gen + loss_fr + loss_ls
            return loss, loss_gen, loss_fr, loss_ls


if __name__ == '__main__':
    x = torch.rand(16, 3, 112, 112)
    x = x * 2 - 1
    model = MinusBackbone(mode='stage1')
    y = model(x)
    print([item.shape for item in y])
