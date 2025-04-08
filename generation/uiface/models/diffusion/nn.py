import math

import torch
from einops import rearrange
from utils.checkpoint import checkpoint
from utils.helpers import zero_module


class SpatialEmbeddingCrossAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        context_dim,
        inner_channels=None,
        n_context_tokens=None,
        n_heads=4,
        head_channels=32,
        use_checkpoint=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.context_dim = context_dim

        if n_heads is None or n_heads <= 0:
            n_heads = in_channels // head_channels

        self.n_heads = n_heads
        # print(n_heads)
        self.head_channels = head_channels
        self.inner_channels = (
            n_heads * head_channels if inner_channels is None else inner_channels
        )

        self.x_proj_in = torch.nn.Conv2d(in_channels, self.inner_channels, 1)
        self.x_proj_out = zero_module(
            torch.nn.Conv2d(self.inner_channels, in_channels, 1)
        )

        self.self_attention = MultiHeadAttention(
            in_channels=self.inner_channels,
            n_heads=n_heads,
            head_channels=head_channels,
            use_checkpoint=use_checkpoint,
        )

        d = head_channels if n_context_tokens is None else n_context_tokens
        self.c_proj_in = torch.nn.Linear(context_dim, self.context_dim * d)

        self.cross_attention = MultiHeadAttention(
            in_channels=self.inner_channels,
            key_value_channels=context_dim,
            n_heads=n_heads,
            head_channels=head_channels,
            use_checkpoint=use_checkpoint,
        )

        self.x_norm = torch.nn.GroupNorm(32, self.inner_channels)

        # TODO: c_norm might not be necessary
        self.c_norm = torch.nn.GroupNorm(32, self.context_dim)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(self.inner_channels, self.inner_channels * 4),
            torch.nn.GELU(),
            torch.nn.Linear(self.inner_channels * 4, self.inner_channels),
        )

    def forward(self, x, c, src_self_attn_map=None, src_cross_attn_map=None):
        # x is batch of image tensors or feature maps
        # c is batch of context embedding vectors

        b, n, h, w = x.shape
        x_in = x

        x = self.x_proj_in(x)
        x = rearrange(x, "b n h w -> b n (h w)")

        c = self.c_proj_in(c)
        c = rearrange(c, "b (n d) -> b n d", n=self.context_dim)
        c = self.c_norm(c)
        x_pre = x
        x, self_attn_map = self.self_attention(
            self.x_norm(x), src_attn_map=src_self_attn_map, negative=False
        )
        x = x + x_pre

        x_pre = x
        x, cross_attn_map = self.cross_attention(
            self.x_norm(x), c, src_attn_map=src_cross_attn_map, negative=True
        )
        x = x + x_pre

        x = (
            rearrange(
                self.ff(rearrange(self.x_norm(x), "b n d -> b d n").contiguous()),
                "b d n -> b n d",
            ).contiguous()
            + x
        )

        x = rearrange(x, "b n (h w) -> b n h w", h=h, w=w)
        x = self.x_proj_out(x)

        return x + x_in, self_attn_map, cross_attn_map


class SpatialSelfAttentionBlock(torch.nn.Module):

    def __init__(self, in_channels, n_heads=4, head_channels=32, use_checkpoint=False):
        super().__init__()
        self.in_channels = in_channels

        if n_heads is None or n_heads <= 0:
            n_heads = in_channels // head_channels

        self.head_channels = head_channels

        self.attention = MultiHeadAttention(
            in_channels=in_channels,
            n_heads=n_heads,
            head_channels=head_channels,
            use_checkpoint=use_checkpoint,
        )

        self.norm = torch.nn.GroupNorm(32, in_channels)

    def forward(self, x, c):
        b, c, h, w = x.shape
        x_in = x

        x = self.norm(x)
        x = rearrange(x, "b c h w -> b c (h w)")

        x = self.attention(x)

        x = rearrange(x, "b c (h w) -> b c h w", h=h)

        return x + x_in


class MultiHeadAttention(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        key_value_channels=None,
        n_heads=4,
        head_channels=32,
        use_checkpoint=False,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.n_heads = n_heads
        self.head_channels = head_channels

        self.inner_dim = n_heads * head_channels

        if key_value_channels is None:
            key_value_channels = in_channels

        self.to_q = torch.nn.Conv1d(in_channels, self.inner_dim, 1)
        self.to_k = torch.nn.Conv1d(key_value_channels, self.inner_dim, 1)
        self.to_v = torch.nn.Conv1d(key_value_channels, self.inner_dim, 1)

        self.proj_out = torch.nn.Conv1d(self.inner_dim, in_channels, 1)

    def forward(self, x, c: torch.Tensor = None, src_attn_map=None, negative=False):
        c = x if c is None else c
        if self.use_checkpoint:
            if x.shape[0] != c.shape[0]:
                print(x.shape, c.shape)
            return checkpoint(
                self._forward, (x, c, src_attn_map, negative), self.parameters(), True
            )
        else:
            return self._forward(x, c, src_attn_map, negative)

    def _forward(self, x, c, src_attn_map, negative=False):
        q, k, v = self.to_q(x), self.to_k(c), self.to_v(c)
        x, attn_map = self.qkv_attention(
            q, k, v, n_heads=self.n_heads, src_attn_map=src_attn_map, negative=negative
        )
        return self.proj_out(x), attn_map

    @staticmethod
    def qkv_attention(q, k, v, n_heads, src_attn_map, negative=False):
        bq, wq, lq = q.shape
        bk, wk, lk = k.shape
        bv, wv, lv = v.shape

        width = wq
        bs = bq

        assert width % n_heads == 0

        q = q.reshape(bs * n_heads, width // n_heads, lq)
        k = k.reshape(bs * n_heads, width // n_heads, lk)
        v = v.reshape(bs * n_heads, width // n_heads, lv)

        scale = 1 / math.sqrt(math.sqrt(width // n_heads))

        if src_attn_map is not None:
            if negative:
                attn_map = torch.einsum(
                    "b c t , b c s -> b t s", q * scale, k * scale
                )  # (300, 256, 32)

                mean, std = attn_map.mean(dim=[2], keepdim=True), attn_map.std(
                    dim=[2], keepdim=True
                )
                mean_src, std_src = src_attn_map.mean(
                    dim=[2], keepdim=True
                ), src_attn_map.std(dim=[2], keepdim=True)

                scale = 0.5
                t_mean = scale * mean_src + (1 - scale) * mean
                t_std = scale * std_src + (1 - scale) * std

                attn_map_norm = (attn_map - mean) / std
                attn_map = attn_map_norm * t_std + t_mean

            else:
                attn_map = src_attn_map

        else:
            attn_map = torch.einsum("b c t , b c s -> b t s", q * scale, k * scale)
        weight = torch.softmax(attn_map.float(), dim=-1).type(attn_map.dtype)

        a = torch.einsum("b t s, b c s -> b c t", weight, v)

        return rearrange(a, "(b n) c t -> b (n c) t", b=bs).contiguous(), torch.einsum(
            "b c t , b c s -> b t s", q * scale, k * scale
        )


if __name__ == "__main__":

    x = torch.ones((3, 128, 16, 16))
    c = torch.ones((3, 256))

    block = SpatialEmbeddingCrossAttentionBlock(
        head_channels=32, n_heads=4, in_channels=128, context_dim=256
    )

    trainable_params, _, total_params = count_model_parameters(block)
    print(f"#Params Model: {trainable_params} (Total: {total_params})")

    block = SpatialTransformer(in_channels=128, n_heads=4, d_head=32, context_dim=256)

    trainable_params, _, total_params = count_model_parameters(block)
    print(f"#Params Model: {trainable_params} (Total: {total_params})")
