import torch.nn as nn
import torch
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, reduction=16):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.sa = sa_layer(out_channels * 4)
        self.se = SEBlock(out_channels, reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.se(out)
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