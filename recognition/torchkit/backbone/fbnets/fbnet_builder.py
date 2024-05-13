from __future__ import absolute_import, division, print_function, unicode_literals

import os
import copy
import json
import logging
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d
from torchkit.backbone.common import initialize_weights
from .layers import (
    FrozenBatchNorm2d,
    interpolate,
    _NewEmptyTensorOp
)


logger = logging.getLogger(__name__)
json_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'fbnet_modeldef.json')
with open(json_file) as f:
    MODEL_ARCH = json.load(f)


def make_divisible(x, divisible_by=8, min_val=8):
    """make channels divisible
    """
    return max(int(np.ceil(int(x) * 1. / divisible_by) * divisible_by), min_val)


PRIMITIVES = {
    "skip": lambda C_in, C_out, expansion, stride, **kwargs: Identity(
        C_in, C_out, stride
    ),
    "ir_k3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, **kwargs
    ),
    "ir_k5": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, kernel=5, **kwargs
    ),
    "ir_k7": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, kernel=7, **kwargs
    ),
    "ir_k1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, kernel=1, **kwargs
    ),
    "shuffle": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, shuffle_type="mid", pw_group=4, **kwargs
    ),
    "basic_block": lambda C_in, C_out, expansion, stride, **kwargs: CascadeConv3x3(
        C_in, C_out, stride
    ),
    "shift_5x5": lambda C_in, C_out, expansion, stride, **kwargs: ShiftBlock5x5(
        C_in, C_out, expansion, stride
    ),
    # layer search 2
    "ir_k3_e1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=3, **kwargs
    ),
    "ir_k3_e3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=3, **kwargs
    ),
    "ir_k3_e6": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=3, **kwargs
    ),
    "ir_k3_s4": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 4, stride, kernel=3, shuffle_type="mid", pw_group=4, **kwargs
    ),
    "ir_k5_e1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=5, **kwargs
    ),
    "ir_k5_e3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=5, **kwargs
    ),
    "ir_k5_e6": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=5, **kwargs
    ),
    "ir_k5_s4": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 4, stride, kernel=5, shuffle_type="mid", pw_group=4, **kwargs
    ),
    # layer search se
    "ir_k3_e1_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=3, se=True, **kwargs
    ),
    "ir_k3_e3_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=3, se=True, **kwargs
    ),
    "ir_k3_e6_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=3, se=True, **kwargs
    ),
    "ir_k3_s4_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in,
        C_out,
        4,
        stride,
        kernel=3,
        shuffle_type="mid",
        pw_group=4,
        se=True,
        **kwargs
    ),
    "ir_k5_e1_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=5, se=True, **kwargs
    ),
    "ir_k5_e3_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=5, se=True, **kwargs
    ),
    "ir_k5_e6_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=5, se=True, **kwargs
    ),
    "ir_k5_s4_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in,
        C_out,
        4,
        stride,
        kernel=5,
        shuffle_type="mid",
        pw_group=4,
        se=True,
        **kwargs
    ),
    # layer search 3 (in addition to layer search 2)
    "ir_k3_s2": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=3, shuffle_type="mid", pw_group=2, **kwargs
    ),
    "ir_k5_s2": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=5, shuffle_type="mid", pw_group=2, **kwargs
    ),
    "ir_k3_s2_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in,
        C_out,
        1,
        stride,
        kernel=3,
        shuffle_type="mid",
        pw_group=2,
        se=True,
        **kwargs
    ),
    "ir_k5_s2_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in,
        C_out,
        1,
        stride,
        kernel=5,
        shuffle_type="mid",
        pw_group=2,
        se=True,
        **kwargs
    ),
    # layer search 4 (in addition to layer search 3)
    "ir_k3_sep": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, kernel=3, cdw=True, **kwargs
    ),
    "ir_k33_e1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=3, cdw=True, **kwargs
    ),
    "ir_k33_e3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=3, cdw=True, **kwargs
    ),
    "ir_k33_e6": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=3, cdw=True, **kwargs
    ),
    # layer search 5 (in addition to layer search 4)
    "ir_k7_e1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=7, **kwargs
    ),
    "ir_k7_e3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=7, **kwargs
    ),
    "ir_k7_e6": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=7, **kwargs
    ),
    "ir_k7_sep": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, kernel=7, cdw=True, **kwargs
    ),
    "ir_k7_sep_e1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=7, cdw=True, **kwargs
    ),
    "ir_k7_sep_e3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=7, cdw=True, **kwargs
    ),
    "ir_k7_sep_e6": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=7, cdw=True, **kwargs
    ),
}


class Flatten(nn.Module):
    """flatten tensor to 1D
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class Identity(nn.Module):
    """identity module
    """
    def __init__(self, C_in, C_out, stride):
        super(Identity, self).__init__()
        self.output_depth = C_out
        self.conv = (
            ConvBNRelu(
                C_in,
                C_out,
                kernel=1,
                stride=stride,
                pad=0,
                no_bias=1,
                use_relu="relu",
                bn_type="bn",
            )
            if C_in != C_out or stride != 1
            else None
        )

    def forward(self, x):
        if self.conv:
            out = self.conv(x)
        else:
            out = x
        return out


class CascadeConv3x3(nn.Sequential):
    """cascade conv block
    """
    def __init__(self, C_in, C_out, stride):
        assert stride in [1, 2]
        ops = [
            Conv2d(C_in, C_in, 3, stride, 1, bias=False),
            BatchNorm2d(C_in),
            nn.ReLU(inplace=True),
            Conv2d(C_in, C_out, 3, 1, 1, bias=False),
            BatchNorm2d(C_out),
        ]
        super(CascadeConv3x3, self).__init__(*ops)
        self.res_connect = (stride == 1) and (C_in == C_out)

    def forward(self, x):
        y = super(CascadeConv3x3, self).forward(x)
        if self.res_connect:
            y += x
        return y


class Shift(nn.Module):
    """shift blcok
    """
    def __init__(self, C, kernel_size, stride, padding):
        super(Shift, self).__init__()
        self.C = C
        kernel = torch.zeros((C, 1, kernel_size, kernel_size), dtype=torch.float32)
        ch_idx = 0

        assert stride in [1, 2]
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = 1

        hks = kernel_size // 2
        ksq = kernel_size ** 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == hks and j == hks:
                    num_ch = C // ksq + C % ksq
                else:
                    num_ch = C // ksq
                kernel[ch_idx : ch_idx + num_ch, 0, i, j] = 1
                ch_idx += num_ch

        self.register_parameter("bias", None)
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        if x.numel() > 0:
            return nn.functional.conv2d(
                x,
                self.kernel,
                self.bias,
                (self.stride, self.stride),
                (self.padding, self.padding),
                self.dilation,
                self.C,  # groups
            )

        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:],
                (self.padding, self.dilation),
                (self.dilation, self.dilation),
                (self.kernel_size, self.kernel_size),
                (self.stride, self.stride),
            )
        ]
        output_shape = [x.shape[0], self.C] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class ShiftBlock5x5(nn.Sequential):
    """shift blcok with 5*5 kernel
    """
    def __init__(self, C_in, C_out, expansion, stride):
        assert stride in [1, 2]
        self.res_connect = (stride == 1) and (C_in == C_out)

        C_mid = make_divisible(C_in * expansion, 8, 8)

        ops = [
            # pw
            Conv2d(C_in, C_mid, 1, 1, 0, bias=False),
            BatchNorm2d(C_mid),
            nn.ReLU(inplace=True),
            # shift
            Shift(C_mid, 5, stride, 2),
            # pw-linear
            Conv2d(C_mid, C_out, 1, 1, 0, bias=False),
            BatchNorm2d(C_out),
        ]
        super(ShiftBlock5x5, self).__init__(*ops)

    def forward(self, x):
        y = super(ShiftBlock5x5, self).forward(x)
        if self.res_connect:
            y += x
        return y


class ChannelShuffle(nn.Module):
    """Channel shuffle block
    """
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, "Incompatible group size {} for input channel {}".format(
            g, C
        )
        return (
            x.view(N, g, int(C / g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )


class ConvBNRelu(nn.Sequential):
    """basic conv-bn-relu block
    """
    def __init__(
        self,
        input_depth,
        output_depth,
        kernel,
        stride,
        pad,
        no_bias,
        use_relu,
        bn_type,
        group=1,
        *args,
        **kwargs
    ):
        super(ConvBNRelu, self).__init__()

        assert use_relu in ["relu", None]
        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]
        assert bn_type in ["bn", "af", "gn", None]
        assert stride in [1, 2, 4]

        op = Conv2d(
            input_depth,
            output_depth,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            bias=not no_bias,
            groups=group,
            *args,
            **kwargs
        )
        nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
        if op.bias is not None:
            nn.init.constant_(op.bias, 0.0)
        self.add_module("conv", op)

        if bn_type == "bn":
            bn_op = BatchNorm2d(output_depth)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=output_depth)
        elif bn_type == "af":
            bn_op = FrozenBatchNorm2d(output_depth)
        if bn_type is not None:
            self.add_module("bn", bn_op)

        if use_relu == "relu":
            self.add_module("relu", nn.ReLU(inplace=True))


class SEModule(nn.Module):
    """se module with reduction 4
    """
    reduction = 4

    def __init__(self, C):
        super(SEModule, self).__init__()
        mid = max(make_divisible(C // self.reduction), 16)
        conv1 = Conv2d(C, mid, 1, 1, 0)
        conv2 = Conv2d(mid, C, 1, 1, 0)

        self.op = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), conv1, nn.ReLU(inplace=True), conv2, nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.op(x)


class Upsample(nn.Module):
    """upsample module
    """
    def __init__(self, scale_factor, mode, align_corners=None):
        super(Upsample, self).__init__()
        self.scale = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return interpolate(
            x, scale_factor=self.scale, mode=self.mode,
            align_corners=self.align_corners
        )


def _get_upsample_op(stride):
    assert (
        stride in [1, 2, 4]
        or stride in [-1, -2, -4]
        or (isinstance(stride, tuple) and all(x in [-1, -2, -4] for x in stride))
    )

    scales = stride
    ret = None
    if isinstance(stride, tuple) or stride < 0:
        scales = [-x for x in stride] if isinstance(stride, tuple) else -stride
        stride = 1
        ret = Upsample(scale_factor=scales, mode="nearest", align_corners=None)

    return ret, stride


class IRFBlock(nn.Module):
    """inverted residual block 
    """
    def __init__(
        self,
        input_depth,
        output_depth,
        expansion,
        stride,
        bn_type="bn",
        kernel=3,
        width_divisor=8,
        shuffle_type=None,
        pw_group=1,
        se=False,
        cdw=False,
        dw_skip_bn=False,
        dw_skip_relu=False,
    ):
        super(IRFBlock, self).__init__()

        assert kernel in [1, 3, 5, 7], kernel

        self.use_res_connect = stride == 1 and input_depth == output_depth
        self.output_depth = output_depth

        mid_depth = int(input_depth * expansion)
        mid_depth = make_divisible(mid_depth, width_divisor, width_divisor)

        # pw
        self.pw = ConvBNRelu(
            input_depth,
            mid_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu="relu",
            bn_type=bn_type,
            group=pw_group,
        ) if mid_depth != input_depth else nn.Sequential()

        # negative stride to do upsampling
        self.upscale, stride = _get_upsample_op(stride)

        # dw
        if kernel == 1:
            self.dw = nn.Sequential()
        elif cdw:
            dw1 = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=stride,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                use_relu="relu",
                bn_type=bn_type,
            )
            dw2 = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=1,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                use_relu="relu" if not dw_skip_relu else None,
                bn_type=bn_type if not dw_skip_bn else None,
            )
            self.dw = nn.Sequential(OrderedDict([("dw1", dw1), ("dw2", dw2)]))
        else:
            self.dw = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=stride,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                use_relu="relu" if not dw_skip_relu else None,
                bn_type=bn_type if not dw_skip_bn else None,
            )

        self.se4 = SEModule(mid_depth) if se else nn.Sequential()

        # pw-linear
        self.pwl = ConvBNRelu(
            mid_depth,
            output_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu=None,
            bn_type=bn_type,
            group=pw_group,
        )

        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = ChannelShuffle(pw_group)

        self.output_depth = output_depth

    def forward(self, x):
        y = self.pw(x)
        if self.shuffle_type == "mid":
            y = self.shuffle(y)
        if self.upscale is not None:
            y = self.upscale(y)
        y = self.dw(y)
        y = self.se4(y)
        y = self.pwl(y)
        if self.use_res_connect:
            y += x
        return y


def _expand_block_cfg(block_cfg):
    assert isinstance(block_cfg, list)
    ret = []
    for idx in range(block_cfg[2]):
        cur = copy.deepcopy(block_cfg)
        cur[2] = 1
        cur[3] = 1 if idx >= 1 else cur[3]
        ret.append(cur)
    return ret


def expand_stage_cfg(stage_cfg):
    """ For a single stage
    """
    assert isinstance(stage_cfg, list)
    ret = []
    for x in stage_cfg:
        ret += _expand_block_cfg(x)
    return ret


def expand_stages_cfg(stage_cfgs):
    """ For a list of stages
    """
    assert isinstance(stage_cfgs, list)
    ret = []
    for x in stage_cfgs:
        ret.append(expand_stage_cfg(x))
    return ret


def _block_cfgs_to_list(block_cfgs):
    assert isinstance(block_cfgs, list)
    ret = []
    for stage_idx, stage in enumerate(block_cfgs):
        stage = expand_stage_cfg(stage)
        for block_idx, block in enumerate(stage):
            cur = {"stage_idx": stage_idx, "block_idx": block_idx, "block": block}
            ret.append(cur)
    return ret


def _add_to_arch(arch, info, name):
    """ arch = [{block_0}, {block_1}, ...]
        info = [
            # stage 0
            [
                block0_info,
                block1_info,
                ...
            ], ...
        ]
        convert to:
        arch = [
            {
                block_0,
                name: block0_info,
            },
            {
                block_1,
                name: block1_info,
            }, ...
        ]
    """
    assert isinstance(arch, list) and all(isinstance(x, dict) for x in arch)
    assert isinstance(info, list) and all(isinstance(x, list) for x in info)
    idx = 0
    for stage_idx, stage in enumerate(info):
        for block_idx, block in enumerate(stage):
            assert (
                arch[idx]["stage_idx"] == stage_idx
                and arch[idx]["block_idx"] == block_idx
            ), "Index ({}, {}) does not match for block {}".format(
                stage_idx, block_idx, arch[idx]
            )
            assert name not in arch[idx]
            arch[idx][name] = block
            idx += 1


def unify_arch_def(arch_def):
    """ unify the arch_def to:
        {
            ...,
            "arch": [
                {
                    "stage_idx": idx,
                    "block_idx": idx,
                    ...
                },
                {}, ...
            ]
        }
    """
    ret = copy.deepcopy(arch_def)

    assert "block_cfg" in arch_def and "stages" in arch_def["block_cfg"]
    assert "stages" not in ret
    # copy 'first', 'last' etc. inside arch_def['block_cfg'] to ret
    ret.update({x: arch_def["block_cfg"][x] for x in arch_def["block_cfg"]})
    ret["stages"] = _block_cfgs_to_list(arch_def["block_cfg"]["stages"])
    del ret["block_cfg"]

    assert "block_op_type" in arch_def
    _add_to_arch(ret["stages"], arch_def["block_op_type"], "block_op_type")
    del ret["block_op_type"]

    return ret


def get_num_stages(arch_def):
    ret = 0
    for x in arch_def["stages"]:
        ret = max(x["stage_idx"], ret)
    ret = ret + 1
    return ret


def get_blocks(arch_def, stage_indices=None, block_indices=None):
    ret = copy.deepcopy(arch_def)
    ret["stages"] = []
    for block in arch_def["stages"]:
        keep = True
        if stage_indices not in (None, []) and block["stage_idx"] not in stage_indices:
            keep = False
        if block_indices not in (None, []) and block["block_idx"] not in block_indices:
            keep = False
        if keep:
            ret["stages"].append(block)
    return ret


class FBNetBuilder(object):
    """FBNet builder
    """
    def __init__(
        self,
        width_ratio,
        bn_type="bn",
        width_divisor=1,
        dw_skip_bn=False,
        dw_skip_relu=False,
    ):
        self.width_ratio = width_ratio
        self.last_depth = -1
        self.bn_type = bn_type
        self.width_divisor = width_divisor
        self.dw_skip_bn = dw_skip_bn
        self.dw_skip_relu = dw_skip_relu

    def add_first(self, stage_info, input_size=None, dim_in=3, pad=True):
        # stage_info: [c, s, kernel]
        assert len(stage_info) >= 2
        if input_size is None:
            input_size = [112, 112]
        channel = stage_info[0]
        stride = stage_info[1]
        out_depth = self._get_divisible_width(int(channel * self.width_ratio))
        kernel = 3
        if len(stage_info) > 2:
            kernel = stage_info[2]

        out = ConvBNRelu(
            dim_in,
            out_depth,
            kernel=kernel,
            stride=1 if input_size[0] == 112 else stride, # fbnet's orignal input size is 224
            pad=kernel // 2 if pad else 0,
            no_bias=1,
            use_relu="relu",
            bn_type=self.bn_type,
        )
        self.last_depth = out_depth
        return out

    def add_blocks(self, blocks):
        """ blocks: [{}, {}, ...]
        """
        assert isinstance(blocks, list) and all(
            isinstance(x, dict) for x in blocks
        ), blocks

        modules = OrderedDict()
        for block in blocks:
            stage_idx = block["stage_idx"]
            #print("stage_idx =", stage_idx)
            
            block_idx = block["block_idx"]
            block_op_type = block["block_op_type"]
            tcns = block["block"]
            n = tcns[2]
            assert n == 1
            nnblock = self.add_ir_block(tcns, [block_op_type])
            nn_name = "xif{}_{}".format(stage_idx, block_idx)
            assert nn_name not in modules
            modules[nn_name] = nnblock
        ret = nn.Sequential(modules)
        return ret

    # def add_final_pool(self, model, blob_in, kernel_size):
    #     ret = model.AveragePool(blob_in, "final_avg", kernel=kernel_size, stride=1)
    #     return ret

    def _add_ir_block(
        self, dim_in, dim_out, stride, expand_ratio, block_op_type, **kwargs
    ):
        ret = PRIMITIVES[block_op_type](
            dim_in,
            dim_out,
            expansion=expand_ratio,
            stride=stride,
            bn_type=self.bn_type,
            width_divisor=self.width_divisor,
            dw_skip_bn=self.dw_skip_bn,
            dw_skip_relu=self.dw_skip_relu,
            **kwargs
        )
        return ret, ret.output_depth

    def add_ir_block(self, tcns, block_op_types, **kwargs):
        t, c, n, s = tcns
        assert n == 1
        out_depth = self._get_divisible_width(int(c * self.width_ratio))
        dim_in = self.last_depth
        op, ret_depth = self._add_ir_block(
            dim_in,
            out_depth,
            stride=s,
            expand_ratio=t,
            block_op_type=block_op_types[0],
            **kwargs
        )
        self.last_depth = ret_depth
        return op

    def _get_divisible_width(self, width):
        ret = make_divisible(int(width), self.width_divisor, self.width_divisor)
        return ret

    def add_last_states(self, dropout_ratio=0.4):
        output_layer = nn.Sequential(nn.BatchNorm2d(self.last_depth),
                                        nn.Dropout(dropout_ratio), Flatten(),
                                        nn.Linear(self.last_depth * 7 * 7, 512),
                                        nn.BatchNorm1d(512, affine=False))
        self.last_depth = 512
        return output_layer

def _get_trunk_cfg(arch_def):
    num_stages = get_num_stages(arch_def)
    trunk_stages = arch_def.get("backbone", range(num_stages - 1))
    ret = get_blocks(arch_def, stage_indices=trunk_stages)
    return ret

class FBNet(nn.Module):
    """FBNet
    """
    def __init__(
        self, builder, arch_def, dim_in, input_size=None
    ):
        super(FBNet, self).__init__()
        if input_size is None:
            input_size = [112, 112]
        assert input_size[0] in [112, 224], \
            "input_size should be [112, 112] or [224, 224]"
        self.first = builder.add_first(arch_def["first"], input_size=input_size, dim_in=dim_in)
        trunk_cfg = _get_trunk_cfg(arch_def)
        self.stages = builder.add_blocks(trunk_cfg["stages"])
        self.last_stages = builder.add_last_states()

        initialize_weights(self.modules())
    
    def forward(self, x):
        y = self.first(x)
        y = self.stages(y)
        y = self.last_stages(y)
        return y

def get_fbnet_model(arch, input_size):
    assert arch in MODEL_ARCH
    arch_def = MODEL_ARCH[arch]
    arch_def = unify_arch_def(arch_def)
    builder = FBNetBuilder(width_ratio=1.0, bn_type="bn", width_divisor=8, dw_skip_bn=True, dw_skip_relu=True)
    model = FBNet(builder, arch_def, dim_in=3, input_size=input_size)
    return model
