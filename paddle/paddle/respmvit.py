import math
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (len(x.shape) - 1)  # 支持任意维度

    # 生成随机张量并 binarize
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)

    # scale & apply mask
    output = x / keep_prob * random_tensor
    return output


class DropPath(paddle.nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(paddle.nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=paddle.nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = paddle.nn.Linear(
            in_features=in_features, out_features=hidden_features
        )
        self.act = act_layer()
        self.fc2 = paddle.nn.Linear(
            in_features=hidden_features, out_features=out_features
        )
        self.drop = paddle.nn.Dropout(p=drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=paddle.nn.GELU,
        norm_layer=paddle.nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, res_prompt=None):
        y, attn = self.attn(self.norm1(x), res_prompt)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class PatchEmbed(paddle.nn.Layer):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size )* (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = paddle.nn.Conv2D(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        B, C, H, W = tuple(x.shape)
        x = self.proj(x)                       # [B, embed_dim, H', W']
        x = x.flatten(start_axis=2)           # [B, embed_dim, H'*W']
        x = x.transpose([0, 2, 1])            # [B, num_patches, embed_dim]
        return x


import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale if qk_scale is not None else head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, res_prompt=None):
        # x: [B, N, C]
        B, N, C = x.shape

        # 计算qkv
        qkv = self.qkv(x)  # [B, N, 3*C]
        qkv = qkv.reshape([B, N, 3, self.num_heads, C // self.num_heads])
        qkv = qkv.transpose([2, 0, 3, 1, 4])  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]     # 各自形状 [B, num_heads, N, head_dim]

        if res_prompt is not None:
            # 假设res_prompt原始形状是 [B, H, W, C]
            if len(res_prompt.shape) == 4:
                Bp, Hp, Wp, Cp = res_prompt.shape
                # 先reshape成序列形式 [B, Np, C]
                res_prompt = res_prompt.reshape([Bp, Hp * Wp, Cp])
            
            # 确保形状匹配：B, Np, C
            # 计算res_prompt的qkv
            qkv_res = self.qkv(res_prompt)
            Np = res_prompt.shape[1]
            qkv_res = qkv_res.reshape([B, Np, 3, self.num_heads, C // self.num_heads])
            qkv_res = qkv_res.transpose([2, 0, 3, 1, 4])  # [3, B, num_heads, Np, head_dim]

            if Np != N:
                # 简单示例：如果不相等，截断或pad到最小长度（可根据需求调整）
                min_len = min(N, Np)
                q = q[:, :, :min_len, :]
                k = k[:, :, :min_len, :]
                v = v[:, :, :min_len, :]
                qkv_res = qkv_res[:, :, :, :min_len, :]
            
            q = q + qkv_res[0]
            k = k + qkv_res[1]
            v = v + qkv_res[2]

        attn = paddle.matmul(q, k.transpose([0,1,3,2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x_out = paddle.matmul(attn, v)  # [B, num_heads, N, head_dim]
        x_out = x_out.transpose([0, 2, 1, 3])  # [B, N, num_heads, head_dim]
        x_out = x_out.reshape([B, N, C])        # [B, N, C]

        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        if res_prompt is not None:
            res_out = paddle.matmul(attn, v).transpose([0, 2, 1, 3]).reshape([B, N, C])
            res_out = self.proj(res_out)
            res_out = self.proj_drop(res_out)
            x_out = x_out + res_out

        return x_out, attn



import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class VisionTransformer(nn.Layer):
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = self.create_parameter(shape=[1, 1, embed_dim], default_initializer=nn.initializer.TruncatedNormal(std=.02))
        self.pos_embed = self.create_parameter(shape=[1, num_patches + 1, embed_dim], default_initializer=nn.initializer.TruncatedNormal(std=.02))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [float(x) for x in paddle.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.LayerList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.depth = depth
        self.decoders_depth = 3

        # self.spatial_prompt = []
        # self.channel_prompt = []
        # for _ in range(3):
        #     self.spatial_prompt.append(self.create_parameter([embed_dim], default_initializer=nn.initializer.Normal()))
        #     self.channel_prompt.append(self.create_parameter([embed_dim, embed_dim], default_initializer=nn.initializer.Identity()))

        # self.conv = nn.Conv2D(in_channels=768, out_channels=768, kernel_size=3, padding=1)
        if num_classes > 0:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = nn.Layer()  # 等价于 Identity，不改变输入


    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:1, :]
        patch_pos_embed = self.pos_embed[:, 1:, :]
        dim = x.shape[-1]
        w0 = w / self.patch_embed.patch_size + 0.1
        h0 = h / self.patch_embed.patch_size + 0.1
        patch_pos_embed = patch_pos_embed.reshape([1, int(math.sqrt(N)), int(math.sqrt(N)), dim])
        patch_pos_embed = patch_pos_embed.transpose([0, 3, 1, 2])
        patch_pos_embed = F.interpolate(patch_pos_embed, scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.transpose([0, 2, 3, 1]).reshape([1, -1, dim])
        return paddle.concat([class_pos_embed, patch_pos_embed], axis=1)

    def prepare_tokens(self, x, ada_token=None):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand([B, 1, -1])
        x = paddle.concat([cls_tokens, x], axis=1)
        x = x + self.interpolate_pos_encoding(x, H, W)
        if ada_token is not None:
            ada_tokens = ada_token.expand([B, ada_token.shape[1], -1])
            x = paddle.concat([x, ada_tokens], axis=1)
        return self.pos_drop(x)

    def forward_features(self, x, res_prompt=None):
        k = 0
        if res_prompt is None:
            res_x = []
            for i, blk in enumerate(self.blocks):
                x, attn = blk(x)
                res_x.append(self.norm(x))
                if i == self.depth - self.decoders_depth:
                    input_token = x
            return self.norm(x), res_x, input_token
        else:
            for i, blk in enumerate(self.blocks):
                if i < self.depth - self.decoders_depth:
                    continue
                x, attn = blk(x, res_prompt[k])
                k += 1
            return self.norm(x), attn

    def forward(self, x, prompt_model):

        x = self.prepare_tokens(x, None)
        x, res_x, input = self.forward_features(x)
        res_prompt = []

        for i in range(0, self.decoders_depth):
            tokens = res_x[12 - self.decoders_depth + i][:, 1:, :]  # [B, N-1, C]

            B, N, C = tokens.shape
            H = W = int(N ** 0.5)
            prompt_input = tokens.transpose([0, 2, 1]).reshape([B, C, H, W])
            prompt = prompt_model(prompt_input)
            prompt = prompt.transpose([0, 2, 3, 1])
            if prompt.shape[0] == 1:
                prompt = prompt.squeeze(0)

            res_prompt = [prompt] + res_prompt

        # second forward
        x, attn = self.forward_features(input, res_prompt)

        return x[:, 0], attn



    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # 返回最后一个 block 的 attention
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output



def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(paddle.nn.LayerNorm),
        # norm_layer=partial(paddle.nn.LayerNorm, eps=1e-06),
        **kwargs
    )
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        # norm_layer=partial(paddle.nn.LayerNorm, eps=1e-06),
        norm_layer=partial(paddle.nn.LayerNorm),
        **kwargs
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        # norm_layer=partial(paddle.nn.LayerNorm, eps=1e-06),
        norm_layer=partial(paddle.nn.LayerNorm),
        **kwargs
    )
    return model