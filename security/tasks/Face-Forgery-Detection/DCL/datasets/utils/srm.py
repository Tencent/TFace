"""Steganalysis Rich Model (SRM)
    - Note: values taken from Zhou et al., "Learning Rich Features for Image Manipulation Detection", CVPR2018
    - Reference: Rich models for steganalysis ofdigital images, TIFS 2012
    - Code adapted from: https://github.com/selimsef/dfdc_deepfake_challenge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def setup_srm_weights(input_channels: int = 3, output_channel=1) -> torch.Tensor:
    """Creates the SRM kernels for noise analysis.
    note: values taken from Zhou et al., "Learning Rich Features for Image Manipulation Detection", CVPR2018
    
    Args:
        input_channels (int, optional):  Defaults to 3.
        output_channel (int, optional): Defaults to 1.

    Returns:
        torch.Tensor
    """
    srm_kernel = torch.from_numpy(
        np.array([
            [  # srm 1/2 horiz
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 1., -2., 1., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
            ],
            [  # srm 1/4
                [0., 0., 0., 0., 0.],
                [0., -1., 2., -1., 0.],
                [0., 2., -4., 2., 0.],
                [0., -1., 2., -1., 0.],
                [0., 0., 0., 0., 0.],
            ],
            [  # srm 1/12
                [-1., 2., -2., 2., -1.],
                [2., -6., 8., -6., 2.],
                [-2., 8., -12., 8., -2.],
                [2., -6., 8., -6., 2.],
                [-1., 2., -2., 2., -1.],
            ]
        ])).float()
    srm_kernel[0] /= 2
    srm_kernel[1] /= 4
    srm_kernel[2] /= 12
    return srm_kernel.view(3, 1, 5, 5).repeat(output_channel, input_channels, 1, 1)


def setup_srm_layer(input_channels: int = 3, output_channel=None) -> torch.nn.Module:
    """Creates a SRM convolution layer for noise analysis.

    Args:
        input_channels (int, optional): [description]. Defaults to 3.
        output_channel ([type], optional): [description]. Defaults to None.

    Returns:
        torch.nn.Module: [description]
    """
    if output_channel == None:
        weights = setup_srm_weights(input_channels)
        conv = torch.nn.Conv2d(input_channels, out_channels=3, kernel_size=5, stride=1, padding=2, bias=False)
    else:
        weights = setup_srm_weights(input_channels, output_channel)
        conv = torch.nn.Conv2d(input_channels,
                               out_channels=output_channel,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               bias=False)
    with torch.no_grad():
        conv.weight = torch.nn.Parameter(weights, requires_grad=False)
    return conv
