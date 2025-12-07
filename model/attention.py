import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# NonLocal Attention
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.inter_channels = in_channels // 2

        # θ, φ, g 모두 1×1 conv
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1)

        # W_z
        self.W_z = nn.Conv2d(self.inter_channels, in_channels, 1)
        nn.init.constant_(self.W_z.weight, 0)
        nn.init.constant_(self.W_z.bias, 0)

    def forward(self, x):
        B, C, H, W = x.size()

        theta_x = self.theta(x).view(B, self.inter_channels, -1) # B, C/2, HW
        phi_x = self.phi(x).view(B, self.inter_channels, -1) # B, C/2, HW
        g_x = self.g(x).view(B, self.inter_channels, -1)  # B, C/2,
        theta_x = theta_x.permute(0, 2, 1)  # B, HW, C/2

        # Attention map: HW × HW
        f = torch.matmul(theta_x, phi_x)    # B, HW, HW
        f_div_C = F.softmax(f, dim=-1)

        # Apply attention weights
        y = torch.matmul(f_div_C, g_x.permute(0, 2, 1))   # B, HW, C/2
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.inter_channels, H, W)

        z = self.W_z(y) + x   # residual 연결
        return z


# ECA Attention 
class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg = nn.AdaptiveAvgPool2d(1)

        
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


# CBAM Attention 
# Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.channel = ChannelAttention(channels, ratio)
        self.spatial = SpatialAttention()

    def forward(self, x):
        x = x * self.channel(x)
        x = x * self.spatial(x)
        return x


