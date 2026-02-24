import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ================= [保持] 时间注意力模块 =================
class TemporalAttention(nn.Module):
    """
    Squeeze-and-Excitation style Temporal Attention.
    用于在时间轴(Channel维度)上筛选关键的 Event Bins。
    """

    def __init__(self, num_bins, reduction_ratio=2):
        super(TemporalAttention, self).__init__()
        hidden_dim = max(num_bins // reduction_ratio, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(num_bins, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_bins, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, T, H, W = x.shape
        avg_out = self.mlp(self.avg_pool(x).view(B, T))
        max_out = self.mlp(self.max_pool(x).view(B, T))
        attn = self.sigmoid(avg_out + max_out).view(B, T, 1, 1)
        return x * attn


# ================= [重构] 空间注意力 (改为坐标注意力风格) =================
class CoordinateSpatialAttention(nn.Module):
    """
    轻量级的空间注意力，替代简单的 Conv 7x7
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        # 使用 Depthwise Conv 减少参数并保持感受野
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W] -> 压缩通道
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.conv(torch.cat([avg_out, max_out], dim=1))
        return scale


# ================= [重构] Asymmetric Enhanced Input Module =================
class AEIM(nn.Module):
    """
    [New Design] Asymmetric Feature Enhancement Module (AFEM)
    不再使用简单的 Pooling，而是使用非对称卷积提取各向异性运动特征。
    Story: 事件相机对水平/垂直运动的响应存在差异，非对称卷积能更好地捕捉这种特性。
    """

    def __init__(self, in_dim=1, out_dim=32, num_bins=5):
        super().__init__()
        self.in_dim = in_dim

        # 1. Base Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        # 2. Asymmetric Branches (替代原有的 Pooling)
        # Branch A: 标准 3x3 捕捉局部细节
        self.branch_local = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        # Branch B: 非对称大感受野 (1x5 + 5x1) 捕捉长距离运动依赖
        self.branch_asym = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, (5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        # 3. Fusion & Calibration
        # Concat 后降维，比单纯的 Add 更强
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim * 3, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        # Attention Modules
        self.spatial_attn = CoordinateSpatialAttention()
        self.temporal_attn = TemporalAttention(num_bins=num_bins)

    def forward(self, ev, map_data):
        B, T, H, W = ev.shape

        # Reshape Activity Map
        map_data_reshaped = rearrange(map_data, 'B (D C) H W -> (B D) C H W', C=self.in_dim)

        # Feature Extraction
        feat_base = self.stem(map_data_reshaped)  # Base
        feat_local = self.branch_local(feat_base)  # Branch A
        feat_asym = self.branch_asym(feat_base)  # Branch B (New)

        # Multi-scale Fusion
        feat_fused = self.fusion(torch.cat([feat_base, feat_local, feat_asym], dim=1))

        # Generate Spatial Mask
        spatial_mask = self.spatial_attn(feat_fused)
        spatial_mask = F.interpolate(spatial_mask, size=(H, W), mode='bilinear', align_corners=False)
        spatial_mask = rearrange(spatial_mask, '(B D) 1 H W -> B D H W', D=T)

        # Apply Attentions
        # 1. Spatial Calibration
        ev_spat = ev * spatial_mask + ev

        # 2. Temporal Calibration
        ev_final = self.temporal_attn(ev_spat)

        return ev_final


# [保留] 其他辅助类 (DeformableCrossAttention, MLP 等) 保持原样
# 为了完整性，这里列出 DeformableCrossAttention 头部，其余部分可以直接复用你原来的代码
class DeformableCrossAttention(nn.Module):
    def __init__(self, channels, num_heads=8, num_points=9):
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.channels = channels
        self.head_dim = channels // num_heads

        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_heads * num_points * 2, 3, padding=1)
        )
        nn.init.constant_(self.offset_conv[2].weight, 0)
        nn.init.constant_(self.offset_conv[2].bias, 0)

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5

    def _get_reference_points(self, H, W, B, device):
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        ref_y = y.float() / (H - 1)
        ref_x = x.float() / (W - 1)
        ref = torch.stack((ref_x, ref_y), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        return ref

    def forward(self, rgb, evt):
        B, C, H, W = evt.shape
        q = self.q_proj(evt).view(B, self.num_heads, self.head_dim, H, W)
        combined = torch.cat([evt, rgb], dim=1)
        offsets = self.offset_conv(combined)
        offsets = offsets.view(B, self.num_heads, self.num_points, 2, H, W)
        offsets = torch.tanh(offsets) * 0.25

        ref_points = self._get_reference_points(H, W, B, evt.device).unsqueeze(1).unsqueeze(1)
        sampling_locations = ref_points + offsets
        sampling_grid = sampling_locations * 2.0 - 1.0
        sampling_grid = sampling_grid.permute(0, 1, 4, 5, 2, 3).flatten(0, 1).flatten(2, 3)

        k_in = self.k_proj(rgb).view(B * self.num_heads, self.head_dim, H, W)
        v_in = self.v_proj(rgb).view(B * self.num_heads, self.head_dim, H, W)

        k_sampled = F.grid_sample(k_in, sampling_grid, align_corners=False)
        v_sampled = F.grid_sample(v_in, sampling_grid, align_corners=False)

        k_sampled = k_sampled.view(B, self.num_heads, self.head_dim, H, W, self.num_points)
        v_sampled = v_sampled.view(B, self.num_heads, self.head_dim, H, W, self.num_points)

        q_ex = q.unsqueeze(-1)
        attn_logits = (q_ex * k_sampled).sum(dim=2)
        attn_weights = F.softmax(attn_logits * self.scale, dim=-1)

        attn_weights = attn_weights.unsqueeze(2)
        out = (attn_weights * v_sampled).sum(dim=-1)
        out = out.reshape(B, C, H, W)

        return evt + self.out_proj(out)