"""Vision Transformer (ViT) adapted for 1D vibration signal classification."""

import torch
import torch.nn as nn
from functools import partial


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


class PatchEmbed(nn.Module):
    """
    1D Patch Embedding with optional overlap.

    Args:
        data_size: Length of input signal.
        in_c: Number of input channels.
        patch_size: Size of each patch.
        overlap: Overlap between adjacent patches.
        norm_layer: Normalization layer (optional).
    """

    def __init__(self, data_size, in_c, patch_size, overlap=0, norm_layer=None):
        super().__init__()
        self.data_size = data_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.grid_size = (data_size - patch_size) // self.stride + 1
        self.num_patches = self.grid_size
        self.embed_dim = in_c * patch_size

        self.projection = nn.Conv1d(in_c, self.embed_dim, kernel_size=patch_size, stride=self.stride)
        self.batch_norm = nn.BatchNorm1d(self.grid_size)
        self.relu = nn.ReLU(inplace=True)
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, N = x.shape
        assert N == self.data_size, \
            f"Input size {N} does not match expected size {self.data_size}"

        x = self.projection(x)       # [B, embed_dim, grid_size]
        x = x.transpose(1, 2)        # [B, grid_size, embed_dim]
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with optional LayerNorm."""

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None,
                 attn_drop_ratio=0., proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.norm(x)
        return x


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


class Block(nn.Module):
    """Transformer encoder block."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim) if norm_layer else nn.Identity()
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio,
                              proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim) if norm_layer else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer1D(nn.Module):
    """
    Vision Transformer adapted for 1D vibration signal classification.

    Args:
        data_size: Length of input signal.
        in_c: Number of input channels.
        num_classes: Number of output classes.
        patch_size: Size of each patch.
        overlap: Overlap between adjacent patches.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        h_args: List of hidden layer sizes for the classifier head.
        qkv_bias: Whether to use bias in QKV projection.
        drop_ratio: Dropout rate.
        attn_drop_ratio: Attention dropout rate.
        drop_path_ratio: Stochastic depth rate.
    """

    def __init__(self, data_size=1024, in_c=1, num_classes=10,
                 patch_size=32, overlap=30, depth=3, num_heads=4,
                 mlp_ratio=4.0, h_args=None, qkv_bias=True, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 norm_layer=None, act_layer=None):
        super().__init__()
        self.num_classes = num_classes
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = patch_size * in_c

        self.patch_embed = PatchEmbed(
            data_size=data_size, patch_size=patch_size,
            in_c=in_c, overlap=overlap, norm_layer=norm_layer
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                  attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()

        # Classifier head
        self.classifier = self._build_classifier(h_args)
        self._initialize_weights()

    def _build_classifier(self, h_args):
        layers = nn.ModuleList()
        if not h_args:
            layers.append(nn.Linear(self.embed_dim, self.num_classes))
        else:
            in_dim = self.embed_dim
            for h in h_args:
                layers.append(nn.Linear(in_dim, h))
                in_dim = h
            layers.append(nn.Linear(in_dim, self.num_classes))
        return layers

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        for module in self.classifier:
            x = module(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
