# (替换 EGHF-Net/network/base.py)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    """通道注意力模块（SE）：增强关键特征通道"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积 (!!!) 速度优化版 (!!!)
    移除了第一层 GroupNorm
    """

    # (定义一个辅助函数来计算组数)
    def _get_num_groups(self, channels):
        # 目标是 32 个组, 但不能超过通道数
        if channels >= 32:
            return 32
        elif channels >= 16:
            return 16
        elif channels >= 8:
            return 8
        else:
            return 1  # (如果通道太少，退化为 LayerNorm)

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()

        # 1. 计算 GroupNorm 的组数 (只为第二层计算)
        gn_groups_out = self._get_num_groups(out_channels)

        # 深度卷积（逐通道）
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=False,
            dilation=dilation
        )

        # (!!!) --- 核心修改：移除 BN1/GN1 --- (!!!)
        # self.bn1 = nn.GroupNorm(gn_groups_in, in_channels)
        # (!!!) --- 修改结束 --- (!!!)

        self.relu = nn.ReLU(inplace=True)

        # 点卷积（通道融合）
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

        # (!!!) --- 核心修改：保留 BN2/GN2 --- (!!!)
        self.bn2 = nn.GroupNorm(gn_groups_out, out_channels)
        # (!!!) --- 修改结束 --- (!!!)

    def forward(self, x):
        x = self.depthwise(x)

        # (!!!) --- 核心修改：移除 BN1/GN1 --- (!!!)
        # x = self.bn1(x)
        # (!!!) --- 修改结束 --- (!!!)

        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return x