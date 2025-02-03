# Copyright (c) Meta Platforms, Inc. and affiliates.
from argparse import Namespace
from typing import Tuple, Dict, AnyStr

import torch
import torch.nn as nn
from kornia.augmentation import RandomCrop
from timm.models.layers import trunc_normal_
from torch import Tensor

from MODALITIES import PIXEL_WISE_MODALITIES
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
)
from .convnextv2 import Block, ConvNeXtV2
from .convnextv2_sparse import SparseConvNeXtV2
from .norm_layers import LayerNorm


# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class FCMAE(nn.Module):
    """Fully Convolutional Masked Autoencoder with ConvNeXtV2 backbone"""

    def __init__(
        self,
        img_size: int = 112,
        depths: list[int] = None,
        dims: list[int] = None,
        decoder_depth: int = 1,
        decoder_embed_dim: int = 512,
        patch_size: float = 16,
        mask_ratio: float = 0.6,
        norm_pix_loss: bool = False,
        args: Namespace = None,
        loss_fn=None,
        sparse: bool = True,
    ):
        super().__init__()

        print("using the multi-modal fcmae model")
        # configs
        self.args = args
        self.img_size = img_size
        if depths is None:  # set default value
            depths = [3, 3, 9, 3]
        self.depths = depths
        if dims is None:
            dims = [96, 192, 384, 768]
        self.dims = dims
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss
        self.loss_fn = loss_fn
        self.sparse = sparse

        self.in_chans = (
            len(args.modalities["sentinel2"])
            if args.modalities["sentinel2"] != "all"
            else len(args.modalities_full["sentinel2"])
        )
        self.out_chans = {}
        for modality in self.args.modalities.keys():

            if modality in ["sentinel2", "sentinel1", "aster", "canopy_height_eth"]:
                # all the conituous pixel level modalities
                if self.args.modalities[modality] == "all":
                    self.out_chans[modality] = len(self.args.modalities_full[modality])
                else:
                    self.out_chans[modality] = len(self.args.modalities[modality])
            elif modality == "biome":
                self.out_chans[modality] = 14  # 14 biomes
            elif modality == "eco_region":
                self.out_chans[modality] = 846  # 846 eco regions
            elif modality in ["lat", "lon", "month", "era5"]:
                if self.args.modalities[modality] == "all":
                    self.out_chans[modality] = len(self.args.modalities_full[modality])
                else:
                    self.out_chans[modality] = len(self.args.modalities[modality])
            elif modality == "esa_worldcover":
                self.out_chans[modality] = 11  # 11 classes for esa worldcover
            elif modality == "dynamic_world":
                self.out_chans[modality] = 9  # 9 classes for dynamic world

        # encoder
        if sparse:
            self.encoder = SparseConvNeXtV2(
                in_chans=self.in_chans,
                depths=depths,
                dims=dims,
                D=2,
                patch_size=patch_size,
                img_size=img_size,
                use_orig_stem=args.use_orig_stem,
            )
        else:
            self.encoder = ConvNeXtV2(
                in_chans=self.in_chans,
                depths=depths,
                dims=dims,
                patch_size=patch_size,
                img_size=img_size,
                use_orig_stem=args.use_orig_stem,
            )
        self.proj = nn.Conv2d(
            in_channels=dims[-1], out_channels=decoder_embed_dim, kernel_size=1
        )

        # mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        decoder = [
            Block(dim=decoder_embed_dim, drop_path=0.0) for _ in range(decoder_depth)
        ]

        # creating a decoder for each modality
        self.decoder_dict = nn.ModuleDict()
        self.pred_dict = nn.ModuleDict()
        for modality in self.args.out_modalities.keys():
            if modality in [
                "sentinel2",
                "sentinel1",
                "aster",
                "canopy_height_eth",
                "dynamic_world",
                "esa_worldcover",
                "IMNET",
            ]:
                # all the pixel level modalities
                self.decoder_dict[modality] = nn.Sequential(*decoder)
                self.pred_dict[modality] = nn.Conv2d(
                    in_channels=decoder_embed_dim,
                    out_channels=patch_size**2 * self.out_chans[modality],
                    kernel_size=1,
                )
            elif modality in ["biome", "eco_region", "lat", "lon", "month", "era5"]:
                # all the non-pixel level modalities along with a global average pooling
                self.decoder_dict[modality] = nn.Sequential(*decoder)
                self.layer_norm_tmp = LayerNorm(
                    decoder_embed_dim, eps=1e-6, data_format="channels_first"
                )
                self.pred_dict[modality] = nn.Linear(
                    in_features=decoder_embed_dim, out_features=self.out_chans[modality]
                )

        self.apply(self._init_weights)

        self.random_crop = RandomCrop((img_size, img_size))

    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=0.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiDepthwiseConvolution):
            trunc_normal_(m.kernel)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight)
            nn.init.constant_(m.linear.bias, 0)
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        if hasattr(self, "mask_token"):
            torch.nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, imgs: Tensor, modality: str) -> Tensor:
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        if modality in ["dynamic_world", "esa_worldcover"]:
            # for these modalities, we only have one channel
            channels = 1
        else:
            channels = self.out_chans[modality]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], channels, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * channels))
        return x

    def unpatchify(self, x: Tensor) -> Tensor:
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        print("shape of x:", x.shape)
        h = w = self.img_size // p
        # assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def gen_random_mask(self, x: Tensor, mask_ratio: float) -> Tensor:
        N = x.shape[0]  # number of samples
        L = (x.shape[2] // self.patch_size) ** 2  # number of patches
        len_keep = int(L * (1 - mask_ratio))  # number of patches to keep

        # the following lines generate a mask with 0s and 1s at random locations
        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask  # (batch_size, no_patches**2)

    def upsample_mask(self, mask: Tensor, scale: float):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** 0.5)
        return (
            mask.reshape(-1, p, p)
            .repeat_interleave(scale, dim=1)
            .repeat_interleave(scale, dim=2)
        )

    def forward_encoder(self, imgs: Tensor, mask_ratio: float) -> Tuple[Tensor, Tensor]:
        # generate random masks
        mask = self.gen_random_mask(imgs, mask_ratio)
        # encoding
        x = self.encoder(imgs, mask)
        return x, mask

    def forward_decoder(self, x: Tensor, mask: Tensor) -> Dict[AnyStr, Tensor]:
        pred = {}
        x = self.proj(x)
        n, c, h, w = x.shape
        mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1.0 - mask) + mask_token * mask
        for modalities in self.args.out_modalities.keys():
            # decoding
            x_ = self.decoder_dict[modalities](x)
            if modalities in ["biome", "eco_region", "lat", "lon", "month", "era5"]:
                x_ = self.layer_norm_tmp(x_)
                # for the image level modalities we use global average pooling followed by the linear layer in pred_dict
                x_ = x_.mean(dim=[-2, -1])
            # pred
            pred[modalities] = self.pred_dict[modalities](x_)
        return pred

    def forward_loss(
        self, imgs_dict: Dict[AnyStr, Tensor], preds: Dict[AnyStr, Tensor], mask: Tensor
    ) -> Tuple[Tensor, Dict, Tensor, Tensor]:
        """
        imgs_dict: A dict of different modalities, each with shape of [N, C, H, W], C is the number of channels/bands
        preds: A dict of predictions for different modalities each of shape [N, L, p*p*C]
        mask: [N, L], 0 is keep, 1 is remove
        """

        loss_dict = {}
        for modality in self.args.out_modalities.keys():
            if modality in ["biome", "eco_region", "lat", "lon", "month", "era5"]:
                # all the image level modalities
                # we still further divide this into categorical and continuous modalities
                if modality in ["biome", "eco_region"]:
                    # categorical modalities
                    imgs = imgs_dict[modality]
                    pred = preds[modality]
                    imgs_classes = torch.argmax(imgs, dim=-1)
                    # we don't need to patchify the image for these modalities
                    # compute the loss
                    loss = nn.CrossEntropyLoss()(pred, imgs_classes)
                    loss_dict[modality] = loss
                elif modality in ["lat", "lon", "month", "era5"]:

                    # continuous modalities
                    imgs = imgs_dict[modality]
                    pred = preds[modality]
                    # we don't need to patchify the image for these modalities but we can still ignore any nan values
                    nan_mask = torch.isnan(imgs)
                    pred = pred[~nan_mask]
                    imgs = imgs[~nan_mask]
                    # compute the loss
                    loss = nn.MSELoss()(pred, imgs)
                    loss_dict[modality] = loss
            elif modality in ["dynamic_world", "esa_worldcover"]:
                # pixel level modalities but categorical
                imgs = imgs_dict[modality]
                pred = preds[modality]

                if len(pred.shape) == 4:
                    n, c, _, _ = pred.shape
                    pred = pred.reshape(n, c, -1)
                    pred = torch.einsum("ncl->nlc", pred)

                # pred is of the shape [N, L, C] where C is patch_size**2 * num_classes. we need to first convert this to [N, L, patch_size**2, num_classes]
                # L is the number of patches
                pred = pred.reshape(
                    pred.shape[0], pred.shape[1], self.patch_size**2, -1
                )

                target = self.patchify(imgs, modality)

                # we only compute the loss on the patches where the mask is 1
                # mask is of the shape [N, L]
                # target is of the shape [N, L, patch_size**2 * num_classes]
                # pred is of the shape [N, L, patch_size**2, num_classes]
                # we need to apply the mask on target and pred for every channel

                target = target.reshape(
                    target.shape[0], target.shape[1], self.patch_size**2, -1
                )
                mask_tmp = (
                    mask.unsqueeze(-1).repeat(1, 1, self.patch_size**2).unsqueeze(-1)
                )

                target = target.reshape(target.shape[0], -1)
                pred = pred.reshape(pred.shape[0], -1, self.out_chans[modality])
                mask_tmp = mask_tmp.reshape(mask.shape[0], -1)

                # we only compute the loss on the patches where the mask is 1
                target = target[mask_tmp == 1]
                pred = pred[mask_tmp == 1]

                # we also apply a nan mask on the target and pred, since sometimes the target can be nan
                nan_mask = target == -1
                target = target[~nan_mask]
                pred = pred[~nan_mask]
                loss = nn.CrossEntropyLoss()(pred, target)
                loss_dict[modality] = loss

            elif modality == "IMNET":
                imgs = imgs_dict[modality]
                pred = preds[modality]
                if len(pred.shape) == 4:
                    n, c, _, _ = pred.shape
                    pred = pred.reshape(n, c, -1)
                    pred = torch.einsum("ncl->nlc", pred)

                target = self.patchify(imgs, modality)
                if self.norm_pix_loss:
                    mean = target.mean(dim=-1, keepdim=True)
                    var = target.var(dim=-1, keepdim=True)
                    target = (target - mean) / (var + 1.0e-6) ** 0.5
                loss = (pred - target) ** 2
                loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

                loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
                loss_dict[modality] = loss
            else:
                # pixel level modalities but continuous
                imgs = imgs_dict[modality]
                pred = preds[modality]

                if len(pred.shape) == 4:
                    n, c, _, _ = pred.shape  # [N, C, H, W]
                    pred = pred.reshape(n, c, -1)
                    pred = torch.einsum("ncl->nlc", pred)
                target = self.patchify(imgs, modality)

                if (
                    self.norm_pix_loss and modality == "sentinel2"
                ):  # we only compute the per-patch norm on sentinel2
                    mean = target.mean(dim=-1, keepdim=True)
                    var = target.var(dim=-1, keepdim=True)
                    target = (target - mean) / (var + 1.0e-6) ** 0.5

                loss = (pred - target) ** 2  # using mean squared error
                nan_mask = torch.isnan(loss)
                count = torch.count_nonzero(~nan_mask, dim=-1)
                loss[nan_mask] = 0
                loss = loss.sum(dim=-1) / count

                # uncomment the below line to compute the loss on the whole image - this results in better reconstructions, but
                # not better representations for downstream tasks
                # mask = torch.ones_like(mask)

                # counting the number of pixels where mask is 1 and loss is not nan. since we only compute the loss on these.
                # we create the nan mask again, since sometimes count can be 0.
                nan_mask = torch.isnan(loss * mask)
                tmp = loss * mask
                tmp[nan_mask] = 0
                sum_ = tmp.sum()

                count = torch.count_nonzero(tmp)
                loss = sum_ / count  # mean loss on removed patches
                loss_dict[modality] = loss

        loss_list = [loss_dict[modality] for modality in loss_dict.keys()]
        if self.args.loss_aggr == "uncertainty":
            uncertainty_loss_, log_vars = self.loss_fn(loss_list)
            loss_combined = sum(uncertainty_loss_)
            return loss_combined, loss_dict, log_vars, uncertainty_loss_
        elif self.args.loss_aggr == "unweighted":
            loss_combined = sum(loss_list)
            return loss_combined, loss_dict, None, None

    def forward(
        self, imgs_dict: Dict[AnyStr, Tensor], labels=None, mask_ratio: float = 0.6
    ):

        # apply random crop to all pixel-wise modalities
        params = self.random_crop.generate_parameters(imgs_dict["sentinel2"].shape)

        # Apply the same transform to all images in the batch
        for modality in imgs_dict:
            type_changed = False
            if modality in PIXEL_WISE_MODALITIES:
                # the interpolate function in random_crop does not work for long dtype. hence for int64, we convert to float
                if imgs_dict[modality].dtype == torch.int64:
                    imgs_dict[modality] = imgs_dict[modality].float()
                    type_changed = True
                    
                imgs_dict[modality] = self.random_crop.apply_transform(
                    imgs_dict[modality], params, None
                )
                if type_changed:
                    imgs_dict[modality] = imgs_dict[modality].long() # convert back to long if the type was changed


        # here imgs_dict is a dictionary with every modality, we set imgs to be the input which in this case
        # is always sentinel2.
        imgs = imgs_dict["sentinel2"]

        # convert nan to 0 for "sentinel2", "sentinel1", "aster", "canopy_height_eth".
        # This is done since the data is normalized to have a mean of 0 and std of 1. hence
        # effectively we are setting the nan values to the mean. In the case of the input,
        # setting to 0 also ensures that these values become sparse.
        for modality in imgs_dict.keys():
            if modality in ["sentinel2", "sentinel1", "aster", "canopy_height_eth"]:
                imgs_dict[modality] = torch.nan_to_num(
                    imgs_dict[modality], nan=0.0, posinf=0.0, neginf=0.0
                )

        x, mask = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(x, mask)
        loss, loss_dict, log_vars, normalized_loss_list = self.forward_loss(
            imgs_dict, pred, mask
        )
        return loss, pred, mask, loss_dict, log_vars, normalized_loss_list


def convnextv2_atto(**kwargs):
    model = FCMAE(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


def convnextv2_femto(**kwargs):
    model = FCMAE(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


def convnextv2_pico(**kwargs):
    model = FCMAE(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


def convnextv2_nano(**kwargs):
    model = FCMAE(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


def convnextv2_tiny(**kwargs):
    model = FCMAE(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnextv2_base(**kwargs):
    model = FCMAE(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


def convnextv2_large(**kwargs):
    model = FCMAE(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


def convnextv2_huge(**kwargs):
    model = FCMAE(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model
