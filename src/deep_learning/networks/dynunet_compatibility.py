"""
MONAI dynUNet from MONAI 0.4.0+85.gaf1ffd6
There are breaking changes in the API from at least version 0.5.0
As a result, we use this older version to allow to use previously trained networks.
"""


from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetOutBlock, UnetResBlock, UnetUpBlock

__all__ = ["DynUNet", "DynUnet", "Dynunet"]


class DynUNetSkipLayer(nn.Module):
    """
    Defines a layer in the UNet topology which combines the downsample and upsample pathways with the skip connection.
    The member `next_layer` may refer to instances of this class or the final bottleneck layer at the bottom the UNet
    structure. The purpose of using a recursive class like this is to get around the Torchscript restrictions on
    looping over lists of layers and accumulating lists of output tensors which much be indexed. The `heads` list is
    shared amongst all the instances of this class and is used to store the output from the supervision heads during
    forward passes of the network.
    """

    heads: List[torch.Tensor]

    def __init__(self, index, heads, downsample, upsample, super_head, next_layer):
        super().__init__()
        self.downsample = downsample
        self.upsample = upsample
        self.next_layer = next_layer
        self.super_head = super_head
        self.heads = heads
        self.index = index

    def forward(self, x):
        downout = self.downsample(x)
        nextout = self.next_layer(downout)
        upout = self.upsample(nextout, downout)

        self.heads[self.index] = self.super_head(upout)

        return upout


class DynUNet(nn.Module):
    """
    This reimplementation of a dynamic UNet (DynUNet) is based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    This model is more flexible compared with ``monai.networks.nets.UNet`` in three
    places:

        - Residual connection is supported in conv blocks.
        - Anisotropic kernel sizes and strides can be used in each layers.
        - Deep supervision heads can be added.

    The model supports 2D or 3D inputs and is consisted with four kinds of blocks:
    one input block, `n` downsample blocks, one bottleneck and `n+1` upsample blocks. Where, `n>0`.
    The first and last kernel and stride values of the input sequences are used for input block and
    bottleneck respectively, and the rest value(s) are used for downsample and upsample blocks.
    Therefore, pleasure ensure that the length of input sequences (``kernel_size`` and ``strides``)
    is no less than 3 in order to have at least one downsample upsample blocks.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        strides: convolution strides for each blocks.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: [``"batch"``, ``"instance"``, ``"group"``]
            feature normalization type and arguments.
        deep_supr_num: number of feature maps that will output during deep supervision head. The
            value should be less than the number of up sample layers. Defaults to 1.
        res_block: whether to use residual connection based convolution blocks during the network.
            Defaults to ``True``.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        norm_name: str = "instance",
        deep_supr_num: int = 1,
        res_block: bool = False,
    ):
        super(DynUNet, self).__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.conv_block = UnetResBlock if res_block else UnetBasicBlock
        self.filters = [min(2 ** (5 + i), 320 if spatial_dims == 3 else 512) for i in range(len(strides))]
        self.input_block = self.get_input_block()
        self.downsamples = self.get_downsamples()
        self.bottleneck = self.get_bottleneck()
        self.upsamples = self.get_upsamples()
        self.output_block = self.get_output_block(0)
        self.deep_supervision_heads = self.get_deep_supervision_heads()
        self.deep_supr_num = deep_supr_num
        self.apply(self.initialize_weights)
        self.check_kernel_stride()
        self.check_deep_supr_num()

        # initialize the typed list of supervision head outputs so that Torchscript can recognize what's going on
        self.heads: List[torch.Tensor] = [torch.rand(1)] * (len(self.deep_supervision_heads) + 1)

        def create_skips(index, downsamples, upsamples, superheads, bottleneck):
            """
            Construct the UNet topology as a sequence of skip layers terminating with the bottleneck layer. This is
            done recursively from the top down since a recursive nn.Module subclass is being used to be compatible
            with Torchscript. Initially the length of `downsamples` will be one more than that of `superheads`
            since the `input_block` is passed to this function as the first item in `downsamples`, however this
            shouldn't be associated with a supervision head.
            """

            if len(downsamples) != len(upsamples):
                raise AssertionError(f"{len(downsamples)} != {len(upsamples)}")
            if (len(downsamples) - len(superheads)) not in (1, 0):
                raise AssertionError(f"{len(downsamples)}-(0,1) != {len(superheads)}")

            if len(downsamples) == 0:  # bottom of the network, pass the bottleneck block
                return bottleneck
            if index == 0:  # don't associate a supervision head with self.input_block
                current_head, rest_heads = nn.Identity(), superheads
            else:
                current_head, rest_heads = superheads[0], superheads[1:]

            # create the next layer down, this will stop at the bottleneck layer
            next_layer = create_skips(1 + index, downsamples[1:], upsamples[1:], rest_heads, bottleneck)

            return DynUNetSkipLayer(index, self.heads, downsamples[0], upsamples[0], current_head, next_layer)

        self.skip_layers = create_skips(
            0,
            [self.input_block] + list(self.downsamples),
            self.upsamples[::-1],
            self.deep_supervision_heads,
            self.bottleneck,
        )

    def check_kernel_stride(self):
        kernels, strides = self.kernel_size, self.strides
        error_msg = "length of kernel_size and strides should be the same, and no less than 3."
        if not (len(kernels) == len(strides) and len(kernels) >= 3):
            raise AssertionError(error_msg)

        for idx in range(len(kernels)):
            kernel, stride = kernels[idx], strides[idx]
            if not isinstance(kernel, int):
                error_msg = "length of kernel_size in block {} should be the same as spatial_dims.".format(idx)
                if len(kernel) != self.spatial_dims:
                    raise AssertionError(error_msg)
            if not isinstance(stride, int):
                error_msg = "length of stride in block {} should be the same as spatial_dims.".format(idx)
                if len(stride) != self.spatial_dims:
                    raise AssertionError(error_msg)

    def check_deep_supr_num(self):
        deep_supr_num, strides = self.deep_supr_num, self.strides
        num_up_layers = len(strides) - 1
        if deep_supr_num < 1 or deep_supr_num >= num_up_layers:
            raise AssertionError("deep_supr_num should be less than the number of up sample layers.")

    def forward(self, x):
        out = self.skip_layers(x)
        return self.output_block(out)

    def get_feature_maps(self):
        """
        Return the feature maps.

        """
        return self.heads[1 : self.deep_supr_num + 1]

    def get_input_block(self):
        return self.conv_block(
            self.spatial_dims,
            self.in_channels,
            self.filters[0],
            self.kernel_size[0],
            self.strides[0],
            self.norm_name,
        )

    def get_bottleneck(self):
        return self.conv_block(
            self.spatial_dims,
            self.filters[-2],
            self.filters[-1],
            self.kernel_size[-1],
            self.strides[-1],
            self.norm_name,
        )

    def get_output_block(self, idx: int):
        return UnetOutBlock(
            self.spatial_dims,
            self.filters[idx],
            self.out_channels,
        )

    def get_downsamples(self):
        inp, out = self.filters[:-2], self.filters[1:-1]
        strides, kernel_size = self.strides[1:-1], self.kernel_size[1:-1]
        return self.get_module_list(inp, out, kernel_size, strides, self.conv_block)

    def get_upsamples(self):
        inp, out = self.filters[1:][::-1], self.filters[:-1][::-1]
        strides, kernel_size = self.strides[1:][::-1], self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        return self.get_module_list(inp, out, kernel_size, strides, UnetUpBlock, upsample_kernel_size)

    def get_module_list(
        self,
        in_channels: List[int],
        out_channels: List[int],
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        conv_block: nn.Module,
        upsample_kernel_size: Optional[Sequence[Union[Sequence[int], int]]] = None,
    ):
        layers = []
        if upsample_kernel_size is not None:
            for in_c, out_c, kernel, stride, up_kernel in zip(
                in_channels, out_channels, kernel_size, strides, upsample_kernel_size
            ):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "upsample_kernel_size": up_kernel,
                }
                layer = conv_block(**params)
                layers.append(layer)
        else:
            for in_c, out_c, kernel, stride in zip(in_channels, out_channels, kernel_size, strides):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                }
                layer = conv_block(**params)
                layers.append(layer)
        return nn.ModuleList(layers)

    def get_deep_supervision_heads(self):
        return nn.ModuleList([self.get_output_block(i + 1) for i in range(len(self.upsamples) - 1)])

    @staticmethod
    def initialize_weights(module):
        name = module.__class__.__name__.lower()
        if "conv3d" in name or "conv2d" in name:
            nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif "norm" in name:
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.zeros_(module.bias)


DynUnet = Dynunet = DynUNet
