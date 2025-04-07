# ======================================================================================================================
# Based on: https://nn.labml.ai/diffusion/ddpm/unet.html
# @misc{labml,
# author = {Varuna Jayasiri, Nipun Wijerathne},
# title = {labml.ai Annotated Paper Implementations},
# year = {2020},
# url = {https://nn.labml.ai/},
# }
# ======================================================================================================================

import math
from functools import partial
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from models.diffusion.nn import (SpatialEmbeddingCrossAttentionBlock,
                                 SpatialSelfAttentionBlock)
from models.diffusion.util import TimestepEmbedSequential, conv_nd, zero_module
from utils.helpers import zero_module


class SinusoidalTimeEmbedding(torch.nn.Module):

    def __init__(self, n_channels: int, max_period: int = 10000):
        super().__init__()
        self.n_channels = n_channels
        self.max_period = max_period

        input_channels = n_channels // 4

        half = input_channels // 2
        self.frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )

        self.lin1 = torch.nn.Linear(input_channels, self.n_channels)
        self.act = torch.nn.SiLU()
        self.lin2 = torch.nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        frequencies = self.frequencies.to(t.device)
        args = t[:, None].float() * frequencies[None]

        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        emb = self.lin1(emb)
        emb = self.act(emb)
        emb = self.lin2(emb)

        return emb


class AdaptiveGroupNormalization(torch.nn.Module):

    def __init__(self, n_groups: int, n_channels: int):
        super().__init__()
        self.norm = torch.nn.GroupNorm(n_groups, n_channels)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
        return (1 + scale) * self.norm(x) + shift


class ResidualBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        context_channels: int,
        n_groups: int = 32,
        p_dropout: float = 0.0,
        condition_type: str = "AddPlusGN",
        is_context_conditional: bool = False,
    ):
        super().__init__()

        self.condition_type = condition_type
        self.is_context_conditional = is_context_conditional

        self.norm1 = torch.nn.GroupNorm(n_groups, in_channels)
        self.act1 = torch.nn.SiLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        if condition_type.lower() in ["adagn", "diffae", "ca"]:
            self.time_emb = torch.nn.Sequential(
                torch.nn.SiLU(), torch.nn.Linear(time_channels, out_channels)
            )

            if self.is_context_conditional and condition_type.lower() != "ca":
                self.context_emb = torch.nn.Sequential(
                    torch.nn.SiLU(), torch.nn.Linear(context_channels, out_channels)
                )

            self.scale_emb = torch.nn.Linear(out_channels, out_channels)
            self.shift_emb = torch.nn.Linear(out_channels, out_channels)

            self.norm2 = AdaptiveGroupNormalization(n_groups, out_channels)
        else:
            self.time_emb = torch.nn.Sequential(
                torch.nn.SiLU(), torch.nn.Linear(time_channels, out_channels)
            )

            if self.is_context_conditional:
                self.context_emb = torch.nn.Sequential(
                    torch.nn.SiLU(), torch.nn.Linear(context_channels, out_channels)
                )

            self.norm2 = torch.nn.GroupNorm(n_groups, out_channels)

        self.act2 = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.conv2 = zero_module(
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
            )
        )

        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 1)
            )
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None):
        h = self.conv1(self.act1(self.norm1(x)))

        if self.condition_type.lower() == "addplusgn":
            h += self.time_emb(t)[:, :, None, None]

            if self.is_context_conditional:
                h += self.context_emb(c)[:, :, None, None]

            h = self.norm2(h)

        elif self.condition_type.lower() in ["adagn", "diffae", "ca"]:

            emb = self.time_emb(t)

            if self.is_context_conditional and self.condition_type.lower() == "adagn":
                emb += self.context_emb(c)

            scale = self.scale_emb(emb)[:, :, None, None]
            shift = self.shift_emb(emb)[:, :, None, None]

            h = self.norm2(h, scale=scale, shift=shift)

            if self.is_context_conditional and self.condition_type.lower() == "diffae":
                h *= self.context_emb(c)[:, :, None, None]

        else:
            raise NotImplementedError

        h = self.conv2(self.dropout(self.act2(h)))

        return h + self.shortcut(x)


class DownBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        context_channels: int,
        condition_type: str,
        is_context_conditional: bool,
        has_attention: bool,
        attention_heads: int,
        attention_head_channels: int,
    ):
        super().__init__()

        self.has_attention = has_attention
        self.res = ResidualBlock(
            in_channels,
            out_channels,
            time_channels=time_channels,
            context_channels=context_channels,
            condition_type=condition_type,
            is_context_conditional=is_context_conditional,
        )

        if has_attention:
            if condition_type.lower() == "ca" and is_context_conditional:
                self.attn = SpatialEmbeddingCrossAttentionBlock(
                    in_channels=out_channels,
                    context_dim=context_channels,
                    n_heads=attention_heads,
                    head_channels=attention_head_channels,
                )
            else:
                self.attn = SpatialSelfAttentionBlock(
                    in_channels=out_channels,
                    n_heads=attention_heads,
                    head_channels=attention_head_channels,
                )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor = None,
        src_self_attn_map=None,
        src_cross_attn_map=None,
    ):
        x = self.res(x, t, c)
        if self.has_attention:
            x, self_attn_map, cross_attn_map = self.attn(
                x, c, src_self_attn_map, src_cross_attn_map
            )
            return x, self_attn_map, cross_attn_map

        return x, None, None


class UpBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        context_channels: int,
        condition_type: str,
        is_context_conditional: bool,
        has_attention: bool,
        attention_heads: int,
        attention_head_channels: int,
    ):
        super().__init__()

        self.has_attention = has_attention
        self.res = ResidualBlock(
            in_channels + out_channels,
            out_channels,
            time_channels=time_channels,
            context_channels=context_channels,
            condition_type=condition_type,
            is_context_conditional=is_context_conditional,
        )

        if has_attention:
            if condition_type.lower() == "ca" and is_context_conditional:
                self.attn = SpatialEmbeddingCrossAttentionBlock(
                    in_channels=out_channels,
                    context_dim=context_channels,
                    n_heads=attention_heads,
                    head_channels=attention_head_channels,
                )
            else:
                self.attn = SpatialSelfAttentionBlock(
                    in_channels=out_channels,
                    n_heads=attention_heads,
                    head_channels=attention_head_channels,
                )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor = None,
        src_self_attn_map=None,
        src_cross_attn_map=None,
    ):
        x = self.res(x, t, c)
        if self.has_attention:
            x, self_attn_map, cross_attn_map = self.attn(
                x, c, src_self_attn_map, src_cross_attn_map
            )
            return x, self_attn_map, cross_attn_map

        return x, None, None


class MiddleBlock(torch.nn.Module):

    def __init__(
        self,
        n_channels: int,
        time_channels: int,
        context_channels: int,
        condition_type: str,
        is_context_conditional: bool,
        attention_heads: int,
        attention_head_channels: int,
    ):
        super().__init__()
        self.res1 = ResidualBlock(
            n_channels,
            n_channels,
            time_channels=time_channels,
            context_channels=context_channels,
            condition_type=condition_type,
            is_context_conditional=is_context_conditional,
        )

        if condition_type.lower() == "ca" and is_context_conditional:
            self.attn = SpatialEmbeddingCrossAttentionBlock(
                in_channels=n_channels,
                context_dim=context_channels,
                n_heads=attention_heads,
                head_channels=attention_head_channels,
            )
        else:
            self.attn = SpatialSelfAttentionBlock(
                in_channels=n_channels,
                n_heads=attention_heads,
                head_channels=attention_head_channels,
            )

        self.res2 = ResidualBlock(
            n_channels,
            n_channels,
            time_channels=time_channels,
            context_channels=context_channels,
            condition_type=condition_type,
            is_context_conditional=is_context_conditional,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        src_self_attn_map=None,
        src_cross_attn_map=None,
    ):
        x = self.res1(x, t, c)
        x, self_attn_map, cross_attn_map = self.attn(
            x, c, src_self_attn_map, src_cross_attn_map
        )
        x = self.res2(x, t, c)
        return x, self_attn_map, cross_attn_map


class Upsample(torch.nn.Module):

    def __init__(self, n_channels):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(
            n_channels, n_channels, (4, 4), (2, 2), (1, 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        _, _ = t, c
        return self.conv(x)


class Downsample(torch.nn.Module):

    def __init__(self, n_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        _, _ = t, c
        return self.conv(x)


class ConditionalUNet(torch.nn.Module):

    def __init__(
        self,
        input_channels: int = 3,
        initial_channels: int = 64,
        channel_multipliers: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attention: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
        attention_heads: int = 4,
        attention_head_channels: int = 32,
        n_blocks_per_resolution: int = 2,
        condition_type: str = "AddPlusGN",
        context_input_channels: int = 512,
        context_channels: int = 512,
        is_context_conditional: bool = False,
        n_context_classes: int = 0,
        learn_empty_context: bool = False,
        context_dropout_probability: float = 0.0,
        unconditioned_probability: float = 0.0,
    ):

        super().__init__()

        self.unconditioned_probability = unconditioned_probability

        n_resolutions = len(channel_multipliers)
        self.is_context_conditional = is_context_conditional

        # project image into feature map
        self.image_proj = torch.nn.Conv2d(
            input_channels, initial_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # time embedding layer. Time embedding has `n_channels * 4` channels
        time_channels = initial_channels * 4
        self.time_emb = SinusoidalTimeEmbedding(time_channels)

        if self.is_context_conditional:

            if context_dropout_probability > 0:
                self.context_dropout = torch.nn.Dropout(p=context_dropout_probability)
            else:
                self.context_dropout = torch.nn.Identity()

            if n_context_classes > 0:
                self.context_emb = torch.nn.Embedding(
                    n_context_classes, embedding_dim=context_channels
                )
            else:
                self.context_emb = torch.nn.Linear(
                    context_input_channels, context_channels
                )

            if learn_empty_context:
                # create learnable constant embedding for dropped contexts
                self.empty_context_embedding = torch.nn.Parameter(
                    torch.empty(context_channels)
                )
                torch.nn.init.normal_(self.empty_context_embedding)
                print(
                    f"use learnable empty_context_embedding: {self.empty_context_embedding.shape}"
                )
            else:
                self.empty_context_embedding = torch.zeros(context_channels)

        down_sample_block = partial(
            DownBlock,
            context_channels=context_channels,
            time_channels=time_channels,
            attention_heads=attention_heads,
            attention_head_channels=attention_head_channels,
            condition_type=condition_type,
            is_context_conditional=is_context_conditional,
        )

        middle_block = partial(
            MiddleBlock,
            context_channels=context_channels,
            time_channels=time_channels,
            attention_heads=attention_heads,
            attention_head_channels=attention_head_channels,
            condition_type=condition_type,
            is_context_conditional=is_context_conditional,
        )

        up_sample_block = partial(
            UpBlock,
            context_channels=context_channels,
            time_channels=time_channels,
            attention_heads=attention_heads,
            attention_head_channels=attention_head_channels,
            condition_type=condition_type,
            is_context_conditional=is_context_conditional,
        )

        # ======================================= DOWN SAMPLER ================
        down = []
        # number of channels
        out_channels = in_channels = initial_channels
        # for each resolution
        count = 0
        for i in range(n_resolutions):
            out_channels = in_channels * channel_multipliers[i]
            for _ in range(n_blocks_per_resolution):
                down.append(
                    down_sample_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        has_attention=is_attention[i],
                    )
                )
                in_channels = out_channels
                count += 1

            # down sample at all resolutions except the last
            if i < n_resolutions - 1:

                down.append(Downsample(in_channels))
        # Combine the set of modules
        self.down = torch.nn.ModuleList(down)
        # ========================================== MIDDLE ===================
        self.middle = middle_block(n_channels=out_channels)

        # ======================================== UP SAMPLER =================
        up = []
        # number of channels
        in_channels = out_channels
        # for each resolution
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels

            for _ in range(n_blocks_per_resolution):
                up.append(
                    up_sample_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        has_attention=is_attention[i],
                    )
                )

            # final block to reduce the number of channels
            out_channels = in_channels // channel_multipliers[i]
            up.append(
                up_sample_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    has_attention=is_attention[i],
                )
            )

            in_channels = out_channels

            # up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # combine the set of modules
        self.up = torch.nn.ModuleList(up)

        # final normalization and convolution layer
        self.norm = torch.nn.GroupNorm(32, initial_channels)
        self.act = torch.nn.SiLU()
        self.final = zero_module(
            torch.nn.Conv2d(
                in_channels, input_channels, kernel_size=(3, 3), padding=(1, 1)
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor = None,
        dropout_mask: torch.Tensor = None,
        condition_flag=None,
        src_self_attn_map_list=None,
        src_cross_attn_map_list=None,
    ):

        t = self.time_emb(t)
        x = self.image_proj(x)

        # use context only if the model is context_conditional
        if self.is_context_conditional:
            if context is None:
                c = (
                    self.empty_context_embedding.unsqueeze(0)
                    .repeat(len(x), 1)
                    .to(x.device)
                )
            else:
                c = self.context_emb(context)

                if dropout_mask is not None:
                    c[dropout_mask] = self.empty_context_embedding.type(c.dtype).to(
                        c.device
                    )

                c = self.context_dropout(c)

                if (self.unconditioned_probability > 0) and (self.training):
                    unconditioned_mask = (
                        torch.rand(c.shape[0]) < self.unconditioned_probability
                    )
                    unconditioned_mask[0] = True
                    c[unconditioned_mask] = self.empty_context_embedding.type(
                        c.dtype
                    ).to(c.device)
                if condition_flag is not None:
                    c[~condition_flag] = self.empty_context_embedding.to(x.device)

        h = [x]
        self_attn_map_list = []
        cross_attn_map_list = []
        attn_cnt = 0
        for m in self.down:
            if isinstance(m, DownBlock):

                src_self_attn_map = (
                    src_self_attn_map_list[attn_cnt]
                    if src_self_attn_map_list is not None
                    else None
                )
                src_cross_attn_map = (
                    src_cross_attn_map_list[attn_cnt]
                    if src_cross_attn_map_list is not None
                    else None
                )
                attn_cnt += 1
                x, self_attn_map, cross_attn_map = m(
                    x,
                    t,
                    c,
                    src_self_attn_map=src_self_attn_map,
                    src_cross_attn_map=src_cross_attn_map,
                )

                self_attn_map_list.append(self_attn_map)
                cross_attn_map_list.append(cross_attn_map)
            else:
                x = m(x, t, c)
            h.append(x)

        src_self_attn_map = (
            src_self_attn_map_list[attn_cnt]
            if src_self_attn_map_list is not None
            else None
        )
        src_cross_attn_map = (
            src_cross_attn_map_list[attn_cnt]
            if src_cross_attn_map_list is not None
            else None
        )
        attn_cnt += 1
        x, self_attn_map, cross_attn_map = self.middle(
            x,
            t,
            c,
            src_self_attn_map=src_self_attn_map,
            src_cross_attn_map=src_cross_attn_map,
        )
        self_attn_map_list.append(self_attn_map)
        cross_attn_map_list.append(cross_attn_map)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t, c)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)

                src_self_attn_map = (
                    src_self_attn_map_list[attn_cnt]
                    if src_self_attn_map_list is not None
                    else None
                )
                src_cross_attn_map = (
                    src_cross_attn_map_list[attn_cnt]
                    if src_cross_attn_map_list is not None
                    else None
                )
                attn_cnt += 1
                x, self_attn_map, cross_attn_map = m(
                    x,
                    t,
                    c,
                    src_self_attn_map=src_self_attn_map,
                    src_cross_attn_map=src_cross_attn_map,
                )

                self_attn_map_list.append(self_attn_map)
                cross_attn_map_list.append(cross_attn_map)

        noise_pred = self.final(self.act(self.norm(x)))

        return noise_pred, self_attn_map_list, cross_attn_map_list
