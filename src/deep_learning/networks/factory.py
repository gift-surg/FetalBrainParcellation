# Copyright 2021 Lucas Fidon and Suprosanna Shit
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../../..'))
from src.deep_learning.networks.dynunet_compatibility import DynUNet  # DynUNet from MONAI 0.4.0+85.gaf1ffd6


SUPPORTED_NETWORKS = [
    'DynUNet',
]

def get_network(config, in_channels, n_class, device):
    """
    Return a 3D CNN.
    :param config: config training parameters.
    :param in_channels: int. Number of input channels.
    :param n_class: int. Number of output classes.
    :param device: Device to use (cpu or gpu).
    :return:
    """
    net_name = config['network']['model_name']

    # Check the config parameters for the model
    if not net_name in SUPPORTED_NETWORKS:
        raise ArgumentError(
            'Model name %s is not supported. Please choose a model name in %s' % \
            (net_name, SUPPORTED_NETWORKS)
        )

    if net_name == 'DynUNet':
        net = get_DynUNet(config, in_channels, n_class, device)
    else:
        net = None

    return net


def get_DynUNet(config, in_channels, n_class, device):
    """
    Return a 3D U-Net.
    :param config: config training parameters.
    :param in_channels: int. Number of input channels.
    :param n_class: int. Number of output classes.
    :param device: Device to use (cpu or gpu).
    :return:
    """
    strides, kernels = [], []

    sizes = config['data']['patch_size']
    spacings = config['data']['spacing']
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    net = DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=n_class,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supr_num=1,
        res_block=False,
    ).to(device)
    return net
