# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import numbers
from einops import rearrange

from functools import partial
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from torch import nn, Tensor
from .utils import merge_pre_bn

from einops import rearrange
from einops.layers.torch import Rearrange
from taming.modules.MambaIR_best import VSSBlock


NORM_EPS = 1e-5


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def lower_bound_fwd(x: Tensor, bound: Tensor) -> Tensor:
    return torch.max(x, bound)


def lower_bound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = (x >= bound) | (grad_output < 0)
    return pass_through_if * grad_output, None

def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )

class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from "Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N: int):
        super(AttentionBlock, self).__init__()

        # 内部类，用于定义一个简单的残差单元
        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super(ResidualUnit, self).__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(N, N // 2, kernel_size=1, stride=1, padding=0, bias=False),  # 1x1卷积
                    nn.ReLU(inplace=True),
                    nn.Conv2d(N // 2, N // 2, kernel_size=3, stride=1, padding=1, bias=False),  # 3x3卷积
                    nn.ReLU(inplace=True),
                    nn.Conv2d(N // 2, N, kernel_size=1, stride=1, padding=0, bias=False),  # 1x1卷积
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        # 定义第一个卷积路径
        self.conv_a = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit()
        )

        # 定义第二个卷积路径，包括3个残差单元和一个1x1卷积
        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            nn.Conv2d(N, N, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        # 使用sigmoid激活函数进行特征权重化
        out = a * torch.sigmoid(b)
        out += identity  # 残差连接
        return out


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False)
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out

# 设计两个不同的卷积路径 (conv_a 和 conv_b) 来处理输入特征图
class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from "Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N: int):
        super(AttentionBlock, self).__init__()

        # 内部类，用于定义一个简单的残差单元
        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super(ResidualUnit, self).__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(N, N // 2, kernel_size=1, stride=1, padding=0, bias=False),  # 1x1卷积
                    nn.ReLU(inplace=True),
                    nn.Conv2d(N // 2, N // 2, kernel_size=3, stride=1, padding=1, bias=False),  # 3x3卷积
                    nn.ReLU(inplace=True),
                    nn.Conv2d(N // 2, N, kernel_size=1, stride=1, padding=0, bias=False),  # 1x1卷积
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        # 定义第一个卷积路径
        self.conv_a = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit()
        )

        # 定义第二个卷积路径，包括3个残差单元和一个1x1卷积
        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            nn.Conv2d(N, N, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        # 使用sigmoid激活函数进行特征权重化
        out = a * torch.sigmoid(b)
        out += identity  # 残差连接
        return out


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))


class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """

    def __init__(self, out_channels, head_dim):
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                       padding=1, groups=out_channels // head_dim, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)

        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None, mlp_ratio=None, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def merge_bn(self, pre_norm):
        merge_pre_bn(self.conv1, pre_norm)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class SGBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False):
        super(SGBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = inp // expand_ratio

        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # nn.ReLU6(inplace=True),
            # pw
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(oup, oup, 3, 1, 1, groups=oup, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, activation=nn.ReLU6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = in_channels * expand_ratio
        self.is_residual = self.stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            # pw Point-wise
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation(inplace=True),
            # dw  Depth-wise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation(inplace=True),
            # pw-linear, Point-wise linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.is_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class NCB(nn.Module):
    """
    Next Convolution Block
    """
    def __init__(self, in_channels, out_channels, stride=1, path_dropout=0,
                 drop=0, head_dim=32, mlp_ratio=3):
        super(NCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        assert out_channels % head_dim == 0

        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        #self.mhca = MHCA(out_channels, head_dim)
        self.mnxet = SGBlock(inp=out_channels, oup=out_channels, stride=1, expand_ratio=2)
        
        # self.mnxet = InvertedResidual(in_channels=out_channels, out_channels=out_channels, stride=1, expand_ratio=2)
        
        self.attention_path_dropout = DropPath(path_dropout)

        self.norm = norm_layer(out_channels)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop, bias=True)
        self.mlp_path_dropout = DropPath(path_dropout)
        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.mlp.merge_bn(self.norm)
            self.is_bn_merged = True

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.attention_path_dropout(self.mnxet(x))
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm(x)
        else:
            out = x
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 13, padding=6, groups=dim)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)

        return x

# 这传入的x是b c h w
class NTB(nn.Module):
    """
    Next Transformer Block
    """

    def __init__(
            self, in_channels, out_channels, path_dropout, stride=1, sr_ratio=1,
            mlp_ratio=2, head_dim=32, mix_block_ratio=0.75, attn_drop=0, drop=0,
    ):
        super(NTB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio
        norm_func = partial(nn.BatchNorm2d, eps=NORM_EPS)

        self.patch_embed = PatchEmbed(in_channels, in_channels // 2, stride)
        self.norm1 = norm_func(in_channels // 2)

        # self.attn = ConvMod(in_channels // 2)
        self.vss = VSSBlock(in_channels // 2)

        self.mhsa_path_dropout = DropPath(path_dropout)

        self.projection = PatchEmbed(in_channels, in_channels // 2, stride=1)

        self.mnxet = SGBlock(inp=in_channels // 2, oup=in_channels // 2, stride=1, expand_ratio=2)
        self.mhca_path_dropout = DropPath(path_dropout)

        self.norm2 = norm_func(out_channels)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop)
        self.mlp_path_dropout = DropPath(path_dropout)

        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.e_mhsa.merge_bn(self.norm1)
            self.mlp.merge_bn(self.norm2)
            self.is_bn_merged = True

    def forward(self, x):

        x_sa = self.patch_embed(x)
        B, C, H, W = x_sa.shape
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm1(x_sa)
        else:
            out = x_sa
            
        out = Rearrange('b c h w -> b h w c')(out)
        out = self.vss(out)
        out = Rearrange('b h w c -> b c h w')(out)
        
        # out = self.mhsa_path_dropout(out)
        x_sa = x_sa + out

        x_pro = self.projection(x)
        x_mn = x_pro + self.mnxet(x_pro)

        x = torch.cat([x_sa, x_mn], dim=1)

        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm2(x)
        else:
            out = x
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x

class Encoder(nn.Module):
    def __init__(self, *, ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 config=[2, 2, 2, 2, 2, 2], drop_path_rate=0.1, num_slices=5, max_support_slices=5, in_channels,
                 resolution, z_channels, double_z=True, resamp_with_conv=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.config = config
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        self.drop_path_rate = drop_path_rate
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0

        self.one = nn.Module()

        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # (B, C, H, W)
        self.one.block_1 = NCB(ch, ch, stride=1, path_dropout=0.1, drop=0., head_dim=32)

        self.one.block_2 = NCB(ch, ch, stride=1, path_dropout=0.1, drop=0., head_dim=32)

        # (B, ch * 2, H/2, W/2)
        self.one.reduction1 = Downsample(ch, ch * 2, resamp_with_conv)

        self.one.block_3 = VSSBlock(hidden_dim=ch * 2, drop_path=self.drop_path_rate)

        self.one.block_4 = VSSBlock(hidden_dim=ch * 2, drop_path=self.drop_path_rate)

        self.one.reduction2 = Downsample(ch * 2, ch * 4, resamp_with_conv)

        self.one.block_5 = VSSBlock(hidden_dim=ch * 4, drop_path=self.drop_path_rate)

        self.one.block_6 = VSSBlock(hidden_dim=ch * 4, drop_path=self.drop_path_rate)

        self.one.reduction3 = Downsample(ch * 4, ch * 8, resamp_with_conv)

        self.one.block_7 = VSSBlock(hidden_dim=ch * 8, drop_path=self.drop_path_rate)

        self.one.block_8 = VSSBlock(hidden_dim=ch * 8, drop_path=self.drop_path_rate)

        self.one.reduction4 = Downsample(ch * 8, ch * 8, resamp_with_conv)

        self.one.block_9 = VSSBlock(hidden_dim=ch * 8, drop_path=self.drop_path_rate)

        self.one.block_10 = VSSBlock(hidden_dim=ch * 8, drop_path=self.drop_path_rate)

        self.norm_out = Normalize(ch * 8)

        self.conv_out = torch.nn.Conv2d(ch * 8,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None
        mask_matrix = None

        # downsampling
        # print(f"x: {x.shape}")
        hs = self.conv_in(x)
        # print(f"After conv_in: {hs.shape}")
        h = self.one.block_1(hs)
        # print(f"After block_1: {h.shape}")
        h = self.one.block_2(h)
        # print(f"After block_2: {h.shape}")
        h = self.one.reduction1(h)
        # print(f"After reduction1: {h.shape}")

        h = self.one.block_3(h)
        # print(f"After block_3: {h.shape}")

        h = self.one.block_4(h)
        # print(f"After block_4: {h.shape}")

        h = self.one.reduction2(h)
        # print(f"After reduction2: {h.shape}")

        h = self.one.block_5(h)
        h = self.one.block_6(h)
        h = self.one.reduction3(h)
        # print(f"After reduction3: {h.shape}")
        h = self.one.block_7(h)
        h = self.one.block_8(h)
        # print(f"After block_8: {h.shape}")

        h = self.one.reduction4(h)
        # print(f"After reduction4: {h.shape}")
        h = self.one.block_9(h)
        h = self.one.block_10(h)
        # print(f"After block_10: {h.shape}")

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)  # 激活函数
        h = self.conv_out(h)
        # print("It is MambaIR VSSBlock")

        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 config=[2, 2, 2, 2, 2, 2], drop_path_rate=0.1, num_slices=5, max_support_slices=5,
                 resamp_with_conv=True, in_channels, resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.config = config
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        self.drop_path_rate = drop_path_rate
        # self.up = up
        # self.up = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       ch * 8,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # upsampling

        self.one = nn.Module()

        self.one.block_10 = VSSBlock(hidden_dim=ch * 8, drop_path=self.drop_path_rate)

        self.one.block_9 = VSSBlock(hidden_dim=ch * 8, drop_path=self.drop_path_rate)

        self.one.reduction4 = Upsample(ch * 8, ch * 8, resamp_with_conv)

        self.one.block_8 = VSSBlock(hidden_dim=ch * 8, drop_path=self.drop_path_rate)

        self.one.block_7 = VSSBlock(hidden_dim=ch * 8, drop_path=self.drop_path_rate)

        self.one.reduction3 = Upsample(ch * 8, ch * 4, resamp_with_conv)

        self.one.block_6 = VSSBlock(hidden_dim=ch * 4, drop_path=self.drop_path_rate)

        self.one.block_5 = VSSBlock(hidden_dim=ch * 4, drop_path=self.drop_path_rate)

        self.one.reduction2 = Upsample(ch * 4, ch * 2, resamp_with_conv)

        self.one.block_4 = VSSBlock(hidden_dim=ch * 2, drop_path=self.drop_path_rate)

        self.one.block_3 = VSSBlock(hidden_dim=ch * 2, drop_path=self.drop_path_rate)

        self.one.reduction1 = Upsample(ch * 2, ch, resamp_with_conv)

        self.one.block_2 = NCB(ch, ch, stride=1, path_dropout=0.1, drop=0., head_dim=32)

        self.one.block_1 = NCB(ch, ch, stride=1, path_dropout=0.1, drop=0., head_dim=32)

        # end
        self.norm_out = Normalize(ch)
        self.conv_out = torch.nn.Conv2d(ch,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # upsampling

        h = self.one.block_10(h)
        h = self.one.block_9(h)
        h = self.one.reduction4(h)
        h = self.one.block_8(h)
        h = self.one.block_7(h)
        h = self.one.reduction3(h)
        h = self.one.block_6(h)
        h = self.one.block_5(h)
        h = self.one.reduction2(h)
        h = self.one.block_4(h)
        h = self.one.block_3(h)
        h = self.one.reduction1(h)
        h = self.one.block_2(h)
        h = self.one.block_1(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # print(f"After decoder: {h.shape}")

        return h