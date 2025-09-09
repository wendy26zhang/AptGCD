import paddle

"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

############################## 相关utils函数，如下 ##############################

def div(self, *args, **kwargs):
    if "other" in kwargs:
        y = kwargs["other"]
    elif "y" in kwargs:
        y = kwargs["y"]
    else:
        y = args[0]

    if not isinstance(y, paddle.Tensor):
        y = paddle.to_tensor(y)

    res = paddle.divide(self, y)

    if "rounding_mode" in kwargs:
        rounding_mode = kwargs["rounding_mode"]
        if rounding_mode=="trunc":
            res = paddle.trunc(res)
        elif rounding_mode=="floor":
            res = paddle.floor(res)

    return res

setattr(paddle.Tensor, "div", div)
setattr(paddle.Tensor, "divide", div)

def reshape(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        assert "shape" in kwargs
        return paddle.reshape(self, shape=kwargs["shape"])

setattr(paddle.Tensor, "reshape", reshape)

def dim2perm(ndim, dim0, dim1):
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm

def view(self, *args, **kwargs):
    if args:
        if len(args)==1 and isinstance(args[0], (tuple, list, str)):
            return paddle.view(self, args[0])
        else:
            return paddle.view(self, list(args))
    elif kwargs:
        return paddle.view(self, shape_or_dtype = list(kwargs.values())[0])

setattr(paddle.Tensor, 'view', view)
############################## 相关utils函数，如上 ##############################



def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (tuple(x.shape)[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape=shape, dtype=x.dtype)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
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


class Attention(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = paddle.nn.Linear(
            in_features=dim, out_features=dim * 3, bias_attr=qkv_bias
        )
        self.attn_drop = paddle.nn.Dropout(p=attn_drop)
        self.proj = paddle.nn.Linear(in_features=dim, out_features=dim)
        self.proj_drop = paddle.nn.Dropout(p=proj_drop)

    def forward(self, x):
        B, N, C = tuple(x.shape)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .transpose(perm=[2, 0, 3, 1, 4])
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(perm=dim2perm(k.ndim, -2, -1)) * self.scale
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(perm=dim2perm((attn @ v).ndim, 1, 2)).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


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

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        else:
            return x


class PatchEmbed(paddle.nn.Layer):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = img_size // patch_size * (img_size // patch_size)
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
        x = (
            self.proj(x)
            .flatten(start_axis=2)
            .transpose(perm=dim2perm(self.proj(x).flatten(start_axis=2).ndim, 1, 2))
        )
        return x


class VisionTransformer(paddle.nn.Layer):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=paddle.nn.LayerNorm,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=[1, 1, embed_dim])
        )
        self.pos_embed = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=[1, num_patches + 1, embed_dim])
        )
        self.pos_drop = paddle.nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in paddle.linspace(start=0, stop=drop_path_rate, num=depth)
        ]
        self.blocks = paddle.nn.LayerList(
            sublayers=[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            paddle.nn.Linear(in_features=embed_dim, out_features=num_classes)
            if num_classes > 0
            else paddle.nn.Identity()
        )
        init_TruncatedNormal = paddle.nn.initializer.TruncatedNormal(std=0.02)
        init_TruncatedNormal(self.pos_embed)
        init_TruncatedNormal = paddle.nn.initializer.TruncatedNormal(std=0.02)
        init_TruncatedNormal(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            init_TruncatedNormal = paddle.nn.initializer.TruncatedNormal(std=0.02)
            init_TruncatedNormal(m.weight)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        elif isinstance(m, paddle.nn.LayerNorm):
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(m.bias)
            init_Constant = paddle.nn.initializer.Constant(value=1.0)
            init_Constant(m.weight)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = tuple(x.shape)[1] - 1
        N = tuple(self.pos_embed.shape)[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = tuple(x.shape)[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = paddle.nn.functional.interpolate(
            x=patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).transpose(perm=[0, 3, 1, 2]),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == tuple(patch_pos_embed.shape)[-2]
            and int(h0) == tuple(patch_pos_embed.shape)[-1]
        )
        patch_pos_embed = patch_pos_embed.transpose(perm=[0, 2, 3, 1]).view(1, -1, dim)
        return paddle.concat(
            x=(class_pos_embed.unsqueeze(axis=0), patch_pos_embed), axis=1
        )

    def prepare_tokens(self, x):
        B, nc, w, h = tuple(x.shape)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(shape=[B, -1, -1])
        x = paddle.concat(x=(cls_tokens, x), axis=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x, return_all_patches=False):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if return_all_patches:
            return x
        else:
            return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                x, attn = blk(x, return_attention=True)
                x = self.norm(x)
                return x, attn

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
        norm_layer=partial(paddle.nn.LayerNorm, eps=1e-06),
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
        norm_layer=partial(paddle.nn.LayerNorm, eps=1e-06),
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
        norm_layer=partial(paddle.nn.LayerNorm, eps=1e-06),
        **kwargs
    )
    return model


class DINOHead(paddle.nn.Layer):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = paddle.nn.Linear(in_features=in_dim, out_features=bottleneck_dim)
        else:
            layers = [paddle.nn.Linear(in_features=in_dim, out_features=hidden_dim)]
            if use_bn:
                layers.append(paddle.nn.BatchNorm1D(num_features=hidden_dim))
            layers.append(paddle.nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(
                    paddle.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
                )
                if use_bn:
                    layers.append(paddle.nn.BatchNorm1D(num_features=hidden_dim))
                layers.append(paddle.nn.GELU())
            layers.append(
                paddle.nn.Linear(in_features=hidden_dim, out_features=bottleneck_dim)
            )
            self.mlp = paddle.nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = paddle.nn.utils.weight_norm(
            layer=paddle.nn.Linear(
                in_features=bottleneck_dim, out_features=out_dim, bias_attr=False
            )
        )
        self.last_layer.weight_g.data.fill_(value=1)
        if norm_last_layer:
            self.last_layer.weight_g.stop_gradient = not False

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            init_TruncatedNormal = paddle.nn.initializer.TruncatedNormal(std=0.02)
            init_TruncatedNormal(m.weight)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)

    def forward(self, x):
        x = self.mlp(x)
        x = paddle.nn.functional.normalize(x=x, axis=-1, p=2)
        x = self.last_layer(x)
        return x


class VisionTransformerWithLinear(paddle.nn.Layer):
    def __init__(self, base_vit, num_classes=200):
        super().__init__()
        self.base_vit = base_vit
        self.fc = paddle.nn.Linear(in_features=768, out_features=num_classes)

    def forward(self, x, return_features=False):
        features = self.base_vit(x)
        features = paddle.nn.functional.normalize(x=features, axis=-1)
        logits = self.fc(features)
        if return_features:
            return logits, features
        else:
            return logits

    @paddle.no_grad()
    def normalize_prototypes(self):
        w = self.fc.weight.data.clone()
        w = paddle.nn.functional.normalize(x=w, axis=1, p=2)
        paddle.assign(w, output=self.fc.weight)