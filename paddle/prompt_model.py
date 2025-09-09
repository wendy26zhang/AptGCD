import paddle

############################## 相关utils函数，如下 ##############################

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
############################## 相关utils函数，如上 ##############################



class sa_layer(paddle.nn.Layer):
    """Constructs a Channel Spatial Group module.
        https://github.com/wofmanaf/SA-Net/blob/main/models/sa_resnet.py
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = paddle.nn.AdaptiveAvgPool2D(output_size=1)
        self.cweight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=[1, channel // (8 * groups), 1, 1])
        )
        self.cbias = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.ones(shape=[1, channel // (8 * groups), 1, 1])
        )
        self.sweight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=[1, channel // (8 * groups), 1, 1])
        )
        self.sbias = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.ones(shape=[1, channel // (8 * groups), 1, 1])
        )
        self.sigmoid = paddle.nn.Sigmoid()
        self.gn = paddle.nn.GroupNorm(
            num_groups=channel // (8 * groups), num_channels=channel // (8 * groups)
        )

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = tuple(x.shape)
        x = x.reshape(b, groups, -1, h, w)
        x = x.transpose(perm=[0, 2, 1, 3, 4])
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = tuple(x.shape)
        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(chunks=2, axis=1)
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)
        out = paddle.concat(x=[xn, xs], axis=1)
        out = out.reshape(b, -1, h, w)
        out = self.channel_shuffle(out, 2)
        return out


class ResidualBlock(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
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
        self.sa = sa_layer(out_channels * 4)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
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
        out = self.Block2(out)
        return out