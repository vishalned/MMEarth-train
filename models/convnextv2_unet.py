
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from .utils import LayerNorm, GRN

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)

        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size=3, padding=1)
        self.norm = LayerNorm(out_dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ConvNeXtV2_unet(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, patch_size = 32, img_size = 128, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1., use_orig_stem = False, args=None
                 ):
        super().__init__()
        self.depths = depths
        self.img_size = img_size
        self.patch_size = patch_size
        self.use_orig_stem = use_orig_stem
        self.args = args
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.num_stage = len(depths)
        if self.use_orig_stem:
            self.stem_orig = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size = patch_size//(2**(self.num_stage-1)), stride=patch_size//(2**(self.num_stage-1))),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
        else:
            self.initial_conv = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1, padding=1),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                nn.GELU(),
            )
            # depthwise conv for stem
            self.stem = nn.Sequential(
                nn.Conv2d(dims[0], dims[0], kernel_size=patch_size//(2**(self.num_stage-1)), stride=patch_size//(2**(self.num_stage-1)), groups=dims[0]),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )


        for i in range(3): 
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(self.num_stage):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
            

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Conv2d(int(dims[0]/2), num_classes, kernel_size=1, stride=1)

        self.upsample_layers = nn.ModuleList()
        self.layerNorm = nn.ModuleList()

        # creating the upsampling with nn.upsample + conv + layernorm + gelu activation. 
        for i in reversed(range(self.num_stage)):
            if i == 3:
                # for the first upsampling block, we dont need to concatenate with any feature map
                self.upsample_layers.append(UpsampleBlock(dims[i], int(dims[i]/2), scale_factor=2))
            elif i == 0:
                # for the last upsampling block, we use our special big stem upsampling followed by the initial conv (upsampled version).
                self.upsample_layers.append(UpsampleBlock(dims[i]*2, int(dims[i]), scale_factor=patch_size//(2**(self.num_stage-1))))

                if self.use_orig_stem:
                    # if we use the original stem, we dont concatenate with the feature map from the encoder since the original stem
                    # doesnt make use of the special initial conv layer when downsampling. we only add the initial conv layer here
                    # just to add additional non-linearity, conv and layernorm.
                    self.initial_conv_upsample = nn.Sequential(
                        nn.Conv2d(dims[i], int(dims[i]/2), kernel_size=3, stride=1, padding=1),
                        LayerNorm(int(dims[i]/2), eps=1e-6, data_format="channels_first"),
                        nn.GELU(),
                    )
                else:
                    self.initial_conv_upsample = nn.Sequential(
                        nn.Conv2d(dims[i]*2, int(dims[i]/2), kernel_size=3, stride=1, padding=1),
                        LayerNorm(int(dims[i]/2), eps=1e-6, data_format="channels_first"),
                        nn.GELU(),
                    )
            else:
                # for the rest of the upsampling blocks, we need to concatenate with the feature map from the encoder hence 
                # the input dimension is doubled.
                self.upsample_layers.append(UpsampleBlock(dims[i]*2, int(dims[i]/2), scale_factor=2))


            


        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
    
    def encoder(self, x):
        enc_features = []
        if self.use_orig_stem:
            x = self.stem_orig(x)
            enc_features.append(x)
        else:
            x = self.initial_conv(x)
            enc_features.append(x)
            x = self.stem(x)
            # self.tmp_var = x
            enc_features.append(x)

        x = self.stages[0](x)

        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i+1](x)
            enc_features.append(x) if i < 2 else None 

        # in total we only save 3 feature maps
        return x, enc_features
    
    def decoder(self, x, enc_features):


        for i in range(3):
            x = self.upsample_layers[i](x)
            tmp = enc_features.pop()
            x = torch.cat([x, tmp], dim=1)
        x = self.upsample_layers[3](x)
        if not self.args.use_orig_stem:
            tmp = enc_features.pop()
            x = torch.cat([x, tmp], dim=1)
        x = self.initial_conv_upsample(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x, enc_features = self.encoder(x)
        x = self.decoder(x, enc_features)
        # x = self.head(x)
        return x

    def forward(self, x):
        x = x.float()
        x = self.forward_features(x)
        x = self.head(x)

        return x

def convnextv2_unet_atto(**kwargs):
    model = ConvNeXtV2_unet(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_unet_femto(**kwargs):
    model = ConvNeXtV2_unet(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnextv2_unet_pico(**kwargs):
    model = ConvNeXtV2_unet(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_unet_nano(**kwargs):
    model = ConvNeXtV2_unet(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_unet_tiny(**kwargs):
    model = ConvNeXtV2_unet(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_unet_base(**kwargs):
    model = ConvNeXtV2_unet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_unet_large(**kwargs):
    model = ConvNeXtV2_unet(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_unet_huge(**kwargs):
    model = ConvNeXtV2_unet(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model