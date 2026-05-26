"""
Multi-Channel Calibrated Swin Transformer (MC-SwinT) for 1D signal classification.

Reference:
    Multi-channel Calibrated Transformer with Shifted Windows for few-shot fault diagnosis
    under sharp speed variation. ISA Transactions.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    """MLP block used in Transformer."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer() if act_layer else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def window_partition(input, windows_size):
    """
    Partition input into non-overlapping windows.

    Args:
        input: [B, C, N]
        windows_size: Window size.

    Returns:
        Windows tensor [B * num_windows, window_size, C].
    """
    B, C, N = input.shape
    windows = input.view(B, C, N // windows_size, windows_size)
    windows = windows.permute(0, 2, 3, 1).contiguous().view(-1, windows_size, C)
    return windows


def window_reverse(windows, original_size, window_size):
    """
    Reverse window partition.

    Args:
        windows: [B * num_windows, window_size, C]
        original_size: Original sequence length.
        window_size: Window size.

    Returns:
        Reconstructed tensor [B, C, original_size].
    """
    N = original_size
    B = int(windows.shape[0] / (N / window_size))
    output = windows.view(B, N // window_size, window_size, -1)
    output = output.permute(0, 3, 1, 2).contiguous().view(B, -1, N)
    return output


class ConvDownsampler(nn.Module):
    """Convolutional downsampling layer (stride=2)."""

    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Conv1d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.norm(x)
        return x


class ConvolutionalEmbedding(nn.Module):
    """Multi-layer convolutional embedding for raw signal."""

    def __init__(self, in_c, kernel_sizes, strides, out_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_c)
        layers = []
        for idx, (ks, s, oc) in enumerate(zip(kernel_sizes, strides, out_channels)):
            in_ch = in_c if idx == 0 else out_channels[idx - 1]
            layers.extend([
                nn.Conv1d(in_ch, oc, kernel_size=ks, stride=s, padding=ks // 2),
                nn.BatchNorm1d(oc),
                nn.ReLU(True),
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.norm(x)
        x = self.layers(x)
        return x


class PatchEmerging(nn.Module):
    """Patch embedding via convolution."""

    def __init__(self, in_channels, out_channels, patch_size, stride):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        return self.proj(x)


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with learned position encoding.

    Args:
        dim: Number of input channels.
        window_size: Window size.
        num_heads: Number of attention heads.
        qkv_bias: Whether to add bias to QKV.
        attn_drop: Attention dropout rate.
        proj_drop: Output projection dropout rate.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, proj_drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.pos_embedding = nn.Conv1d(dim, dim, 3, 1, 1)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(B_ // nw, nw, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Learned position encoding (LePE)
        lepe = self.pos_embedding(v.transpose(2, 3).reshape(B_, -1, N))
        value = (attn @ v).transpose(1, 2).reshape(-1, N, C) + lepe.transpose(1, 2)
        value = self.proj(value)
        value = self.proj_drop(value)
        return value


class MCSwinTransformerBlock(nn.Module):
    """
    MC-Swin Transformer block with shifted window attention.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size.
        shift_size: Shift size for SW-MSA.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        drop_path: Stochastic depth rate.
        layer_scale: Layer scale initial value (None to disable).
    """

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.5,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attention = WindowAttention(
            dim=dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x, attn_mask):
        L = self.L
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # Padding
        pad_r = pad_l = 0
        if N % self.window_size != 0:
            pad_r = pad_l = (self.window_size - N % self.window_size) % self.window_size
            x = F.pad(x, (0, 0, pad_r, pad_l))
        _, Np, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size,), dims=(1,))
        else:
            shifted_x = x
            attn_mask = None

        # Window partition -> attention -> reverse
        x_windows = window_partition(shifted_x.transpose(-1, -2), windows_size=self.window_size)
        attn_windows = self.attention(x_windows, attn_mask)
        shifted_x = window_reverse(attn_windows, original_size=Np, window_size=self.window_size)
        shifted_x = shifted_x.transpose(-1, -2)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size,), dims=1)
        else:
            x = shifted_x

        # Remove padding
        if pad_r > 0 or pad_l > 0:
            x = x[:, :N, :].contiguous()

        # Residual connection
        if not self.layer_scale:
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = shortcut + self.drop_path(self.gamma1 * x)
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))

        return x


class MCSwinLayer(nn.Module):
    """A stage of MC-Swin Transformer blocks with optional downsampling."""

    def __init__(self, dim, depth, num_heads, window_size=8, downsample=False,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0.5, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = window_size // 2

        self.blocks = nn.Sequential(*[
            MCSwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer, norm_layer=norm_layer, layer_scale=layer_scale
            )
            for i in range(depth)
        ])
        self.downsample = downsample
        self.downsample_layer = ConvDownsampler(dim=dim) if downsample else nn.Identity()

    def create_mask(self, x, N):
        if N % self.window_size != 0:
            Np = int(np.ceil(N / self.window_size)) * self.window_size
        else:
            Np = N
        img_mask = torch.zeros((1, Np, 1), device=x.device)
        n_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for n in n_slices:
            img_mask[:, n, :] = cnt
            cnt += 1

        mask_windows = window_partition(img_mask.transpose(-1, -2), windows_size=self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, N):
        attn_mask = self.create_mask(x, N)
        for block in self.blocks:
            block.L = N
            x = block(x, attn_mask)
        x = self.downsample_layer(x)
        if self.downsample:
            N = (N + 1) // 2
        return x, N


class MCSwinT(nn.Module):
    """
    Multi-Channel Calibrated Swin Transformer for 1D signal classification.

    Args:
        in_c: Number of input channels.
        num_classes: Number of output classes.
        kernel_sizes: Kernel sizes for convolutional embedding.
        strides: Strides for convolutional embedding.
        out_channels: Output channels for convolutional embedding.
        dim: Transformer embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        window_size: Window size for shifted window attention.
        h_args: Hidden layer sizes for classifier head.
        downscale: Whether to apply downsampling.
        mlp_ratio: MLP expansion ratio.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        drop_path: Stochastic depth rate.
        layer_scale: Layer scale initial value.
    """

    def __init__(self, in_c=1, num_classes=10, kernel_sizes=None, strides=None,
                 out_channels=None, dim=128, depth=3, num_heads=8, window_size=16,
                 h_args=None, downscale=False, mlp_ratio=4, qkv_bias=True,
                 qk_scale=None, drop=0.5, attn_drop=0.5, drop_path=0.5,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=0.5):
        super().__init__()

        # Default configurations
        if kernel_sizes is None:
            kernel_sizes = [15, 9, 5, 3]
        if strides is None:
            strides = [2, 1, 1, 1]
        if out_channels is None:
            out_channels = [64, 128, 128, 192]
        if h_args is None:
            h_args = [100, 64, 32]

        self.conv_embedding = ConvolutionalEmbedding(
            in_c=in_c, kernel_sizes=kernel_sizes,
            strides=strides, out_channels=out_channels
        )
        self.patch_embedding = PatchEmerging(
            in_channels=out_channels[-1], out_channels=dim,
            patch_size=8, stride=8
        )
        self.swin_layer = MCSwinLayer(
            dim=dim, depth=depth, num_heads=num_heads,
            window_size=window_size, downsample=downscale,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            act_layer=act_layer, norm_layer=norm_layer, layer_scale=layer_scale
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        self.classifier = nn.ModuleList()
        in_dim = dim
        for h in h_args:
            self.classifier.append(nn.Linear(in_dim, h))
            in_dim = h
        self.classifier.append(nn.Linear(in_dim, num_classes))

    def forward(self, x):
        x = self.conv_embedding(x)
        x = self.patch_embedding(x).transpose(-1, -2)
        _, N, _ = x.shape
        x, N = self.swin_layer(x, N)
        x = self.avg_pool(x.transpose(-1, -2))
        x = x.squeeze(-1)
        for module in self.classifier:
            x = module(x)
        return x
