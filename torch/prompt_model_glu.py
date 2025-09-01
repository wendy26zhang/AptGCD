import torch.nn as nn
import torch
# https://github.com/DaiShiResearch/TransNeXt
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    # def forward(self, x, H, W):
    def forward(self, x):
        B,C,H,W = x.size()
        x = x.view(B, C, -1).transpose(1, 2)
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, reduction=16):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.sa = sa_layer(out_channels * 4)
        # self.se = SEBlock(out_channels, reduction)
        self.glu = ConvolutionalGLU(in_channels,out_channels,out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        # out = self.se(out)
        out = self.glu(out)
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
        # out = self.Block2(out)
        return out