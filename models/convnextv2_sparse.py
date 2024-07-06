# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from MinkowskiOps import (
    to_sparse,
)
from timm.models.layers import trunc_normal_
from torch import Tensor

from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
    MinkowskiGELU,
)
from .sparse_norm_layers import MinkowskiLayerNorm, MinkowskiGRN, MinkowskiDropPath


class Block(nn.Module):
    """Sparse ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim: int, drop_path: float = 0.0, D: int = 3):
        super().__init__()
        self.dwconv = MinkowskiDepthwiseConvolution(
            dim, kernel_size=7, bias=True, dimension=D
        )
        self.norm = MinkowskiLayerNorm(dim, 1e-6)
        self.pwconv1 = MinkowskiLinear(dim, 4 * dim)
        self.act = MinkowskiGELU()
        self.pwconv2 = MinkowskiLinear(4 * dim, dim)
        self.grn = MinkowskiGRN(4 * dim)
        self.drop_path = MinkowskiDropPath(drop_path)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x


class SparseConvNeXtV2(nn.Module):
    """Sparse ConvNeXtV2.

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        patch_size: int = 32,
        img_size: int = 128,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: list[int] = None,
        dims: list[int] = None,
        drop_path_rate: float = 0.0,
        D: int = 3,
        use_orig_stem: bool = False,
    ):
        super().__init__()
        print("using the new sparse convnextv2 model")
        self.depths = depths
        if self.depths is None:  # set default value
            self.depths = [3, 3, 9, 3]
        self.num_classes = num_classes
        self.img_size = img_size
        self.use_orig_stem = use_orig_stem
        self.num_stage = len(depths)
        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        if dims is None:
            dims = [96, 192, 384, 768]

        # the original ConvNeXtV2 stem layer (big downsampling - kernel size 4, stride 4 -- assuming patch size 32)
        if self.use_orig_stem:
            self.stem_orig = nn.Sequential(
                MinkowskiConvolution(
                    in_chans,
                    dims[0],
                    kernel_size=patch_size // (2 ** (self.num_stage - 1)),
                    stride=patch_size // (2 ** (self.num_stage - 1)),
                    bias=True,
                    dimension=D,
                ),  # 8 is a factor of the num_stages -> 2**(num_stages - 1
                MinkowskiLayerNorm(dims[0], eps=1e-6),
            )
        else:
            # an initial conv layer to get dense feature maps
            self.initial_conv = nn.Sequential(
                MinkowskiConvolution(
                    in_chans, dims[0], kernel_size=3, stride=1, bias=True, dimension=D
                ),
                MinkowskiLayerNorm(dims[0], eps=1e-6),
                MinkowskiGELU(),
            )
            # our version of the stem downsampling: includes depthwise_conv + layer norm
            self.stem = nn.Sequential(
                MinkowskiDepthwiseConvolution(
                    dims[0],
                    kernel_size=patch_size // (2 ** (self.num_stage - 1)),
                    stride=patch_size // (2 ** (self.num_stage - 1)),
                    bias=True,
                    dimension=D,
                ),
                MinkowskiLayerNorm(dims[0], eps=1e-6),
            )

        """
        after the first downsampling layer. 
        patch 32, after downsampling -> 8x8 -> 6 visible patches (mask ratio 0.6, 16 total patches)
        patch 16, after downsampling -> 8x8 -> 25 visible patches (mask ratio 0.6, 64 total patches)

        this explains why patch 16 results in slower run times. 
        """

        for i in range(3):
            downsample_layer = nn.Sequential(
                MinkowskiLayerNorm(dims[i], eps=1e-6),
                MinkowskiConvolution(
                    dims[i],
                    dims[i + 1],
                    kernel_size=2,
                    stride=2,
                    bias=True,
                    dimension=D,
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stage):
            stage = nn.Sequential(
                *[
                    Block(dim=dims[i], drop_path=dp_rates[cur + j], D=D)
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=0.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiDepthwiseConvolution):
            trunc_normal_(m.kernel, std=0.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight, std=0.02)
            nn.init.constant_(m.linear.bias, 0)

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** 0.5)
        return (
            mask.reshape(-1, p, p)
            .repeat_interleave(scale, axis=1)
            .repeat_interleave(scale, axis=2)
        )

    def forward(self, x:Tensor, mask:Tensor)->Tensor:
        num_patches = mask.shape[1]
        scale = int(self.img_size // (num_patches**0.5))
        mask = self.upsample_mask(mask, scale)

        mask = mask.unsqueeze(1).type_as(x)
        x *= 1.0 - mask

        x = to_sparse(x)

        # patch embedding
        if self.use_orig_stem:
            x = self.stem_orig(x)
        else:
            x = self.initial_conv(x)
            x = self.stem(x)

        x = self.stages[0](x)

        # sparse encoding
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i + 1](
                x
            )  # the stages var in the code has 4 stages, so the first one is to be used immediately after the stem

        # densify
        x = x.dense()[0]

        return x
