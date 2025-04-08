import torch
from models.autoencoder.modules import Decoder, Encoder
from models.autoencoder.quantization import VectorQuantizer2 as VectorQuantizer
from omegaconf import OmegaConf


class VQEncoderInterface(torch.nn.Module):

    def __init__(self, first_stage_config_path: str, encoder_state_dict_path: str):
        super().__init__()

        embed_dim = 3

        config = OmegaConf.load(first_stage_config_path)
        dd_config = config.params.ddconfig

        self.encoder = Encoder(**dd_config)
        self.quant_conv = torch.nn.Conv2d(dd_config["z_channels"], embed_dim, 1)

        state_dict = torch.load(encoder_state_dict_path)
        self.load_state_dict(state_dict)

    def forward(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h


class VQDecoderInterface(torch.nn.Module):

    def __init__(self, first_stage_config_path: str, decoder_state_dict_path: str):
        super().__init__()

        embed_dim = 3
        n_embed = 8192

        config = OmegaConf.load(first_stage_config_path)
        dd_config = config.params.ddconfig

        self.decoder = Decoder(**dd_config)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, dd_config["z_channels"], 1)

        state_dict = torch.load(decoder_state_dict_path)
        self.load_state_dict(state_dict)

    def forward(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
