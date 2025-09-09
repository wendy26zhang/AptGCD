import paddle

############################## 相关utils函数，如下 ##############################

def dim2perm(ndim, dim0, dim1):
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm

# def view(self, *args, **kwargs):
#     if args:
#         if len(args)==1 and isinstance(args[0], (tuple, list, str)):
#             return paddle.view(self, args[0])
#         else:
#             return paddle.view(self, list(args))
#     elif kwargs:
#         return paddle.view(self, shape_or_dtype = list(kwargs.values())[0])

# setattr(paddle.Tensor, 'view', view)

def view(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list, str)):
            return self.reshape(args[0])
        else:
            return self.reshape(list(args))
    elif kwargs:
        return self.reshape(list(kwargs.values())[0])

setattr(paddle.Tensor, 'view', view)

############################## 相关utils函数，如上 ##############################



# class DWConv(paddle.nn.Layer):
#     def __init__(self, dim=768):
#         super(DWConv, self).__init__()
#         self.dwconv = paddle.nn.Conv2D(
#             in_channels=dim,
#             out_channels=dim,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias_attr=True,
#             groups=dim,
#         )

#     def forward(self, x, H, W):
#         B, N, C = tuple(x.shape)
#         x = x.transpose(perm=dim2perm(x.ndim, 1, 2)).view(B, C, H, W).contiguous()
#         x = self.dwconv(x)
#         x = x.flatten(start_axis=2).transpose(
#             perm=dim2perm(x.flatten(start_axis=2).ndim, 1, 2)
#         )
#         return x


class DWConv(paddle.nn.Layer):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = paddle.nn.Conv2D(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,  # 深度可分离卷积
            bias_attr=True  # 对应 PyTorch 的 bias=True
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose([0, 2, 1]).reshape([B, C, H, W])
        x = self.dwconv(x)
        x = x.flatten(2).transpose([0, 2, 1])
        return x


class ConvolutionalGLU(paddle.nn.Layer):
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
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = paddle.nn.Linear(
            in_features=in_features, out_features=hidden_features * 2
        )
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = paddle.nn.Linear(
            in_features=hidden_features, out_features=out_features
        )
        self.drop = paddle.nn.Dropout(p=drop)

    def forward(self, x):
        B, C, H, W = tuple(x.shape)
        x = x.reshape([B, C, -1]).transpose([0, 2, 1])
        #x = x.view(B, C, -1).transpose(perm=dim2perm(x.view(B, C, -1).ndim, 1, 2))
        # x = x.view(B, C, -1).transpose(perm=dim2perm(x.view(B, C, -1).ndim, 1, 2))
       # x, v = self.fc1(x).chunk(chunks=2, axis=-1)
        x, v = paddle.chunk(self.fc1(x), chunks=2, axis=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        #x = x.transpose(perm=dim2perm(x.ndim, 1, 2)).view(B, C, H, W)
        x = x.transpose([0, 2, 1]).reshape([B, C, H, W])
        return x


class ResidualBlock(paddle.nn.Layer):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, padding=1, reduction=16
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.relu = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = paddle.nn.BatchNorm2D(num_features=out_channels)
        self.glu = ConvolutionalGLU(in_channels, out_channels, out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.glu(out)
        out += residual
        out = self.relu(out)
        return out


class PromptResNet(paddle.nn.Layer):
    def __init__(self):
        super(PromptResNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(
            in_channels=768, out_channels=768, kernel_size=3, padding=1
        )
        self.relu = paddle.nn.ReLU()
        self.Block1 = ResidualBlock(768, 768)
        self.Block2 = ResidualBlock(768, 768)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.Block1(out)
        return out