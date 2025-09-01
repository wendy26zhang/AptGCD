import torch.nn as nn
import torch
class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
        https://github.com/wofmanaf/SA-Net/blob/main/models/sa_resnet.py
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, channel // (8 * groups), 1, 1))
        # self.cweight = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (8 * groups), 1, 1))
        # self.cbias = nn.Parameter(torch.ones(1, channel, 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (8 * groups), 1, 1))
        # self.sweight = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (8 * groups), 1, 1))
        # self.sbias = nn.Parameter(torch.ones(1, channel, 1, 1))
 
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (8 * groups), channel // (8 * groups))
        # self.gn = nn.BatchNorm2d(channel)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        # print(x_0.shape)
        # channel attention
        xn = self.avg_pool(x_0)
        # print(xn.shape)
        # print(self.cweight.shape)
        # print(self.cbias.shape)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sa = sa_layer(out_channels * 4)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        # out = self.sa(out)
        out += residual
        out = self.relu(out)
        return out

class PromptResNet(nn.Module):
    def __init__(self):
        super(PromptResNet, self).__init__()
        self.conv1 = nn.Conv2d(768, 768, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.Block1 = ResidualBlock(768, 768)
        self.Block2 = ResidualBlock(768, 768)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.Block1(out)
        out = self.Block2(out)
        return out