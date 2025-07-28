import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings

warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch, numbers
import torch.nn as nn
from einops import rearrange

from engine.extre_module.ultralytics_nn.conv import Conv


class Partial_Conv(nn.Module):
    def __init__(self, inc, ouc, n_div=4, forward='split_cat'):
        super().__init__()
        assert inc == ouc, "Input and output channels must be the same for this integration."
        self.dim_conv3 = inc // n_div
        self.dim_untouched = inc - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        self.conv1x1 = nn.Identity()

        if forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return self.conv1x1(x)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FiGCA_MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x


class FiGCA(nn.Module):
    """
    Efficient local global context aggregation module
    dim: number of channels of input
    heads: number of heads utilized in computing attention
    """

    def __init__(self, dim, heads=4, pconv_n_div=4):
        super().__init__()
        self.heads = heads

        # 核心改进：使用 PConv 替换 DWConv
        # 输入和输出通道数均为 dim//2
        local_channels = dim // 2
        self.pconv = Partial_Conv(local_channels, local_channels, n_div=pconv_n_div, forward='split_cat')

        self.qkvl = nn.Conv2d(dim // 2, (dim // 4) * self.heads, 1, padding=0)
        self.pool_q = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.pool_k = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x1, x2 = torch.split(x, [C // 2, C // 2], dim=1)

        x1 = self.act(self.pconv(x1))

        x2 = self.act(self.qkvl(x2))

        x2 = x2.reshape(B, self.heads, C // 4, H, W)

        # This part of the code seems to have a bug or a very specific design choice.
        # q = torch.sum(x2[:, :-3, :, :, :], dim=1) is unusual.
        # Assuming heads=4, x2 has shape [B, 4, C//4, H, W].
        # Then x2[:, :-3, :, :, :] would have shape [B, 1, C//4, H, W], so the sum is just a squeeze.
        # Let's stick to the original implementation logic.
        q = torch.sum(x2[:, :-3, :, :, :], dim=1)
        k = x2[:, -3, :, :, :]

        q = self.pool_q(q)
        k = self.pool_k(k)

        v = x2[:, -2, :, :, :].flatten(2)
        lfeat = x2[:, -1, :, :, :]

        qk = torch.matmul(q.flatten(2), k.flatten(2).transpose(1, 2))
        qk = torch.softmax(qk, dim=1).transpose(1, 2)

        x2 = torch.matmul(qk, v).reshape(B, C // 4, H, W)

        x = torch.cat([x1, lfeat, x2], dim=1)

        return x


class FiGCA_EncoderBlock(nn.Module):
    """
    dim: number of channels of input features
    """

    def __init__(self, inc, dim, mlp_ratio=4, heads=4):
        super().__init__()

        self.layer_norm1 = LayerNorm(dim, 'BiasFree')
        self.layer_norm2 = LayerNorm(dim, 'BiasFree')
        self.mlp = FiGCA_MLP(dim=dim, mlp_ratio=mlp_ratio)
        self.attn = FiGCA(dim, heads=heads)

        self.conv1x1 = Conv(inc, dim, 1) if inc != dim else nn.Identity()

    def forward(self, x):
        x = self.conv1x1(x)
        inp_copy = x

        x = self.layer_norm1(inp_copy)
        x = self.attn(x)
        out = x + inp_copy

        x = self.layer_norm2(out)
        x = self.mlp(x)
        out = out + x

        return out
