import torch
import torch.nn as nn
import torch.nn.functional as F
from network.backbone import DualSegFormerBackbone
from network.modules import DeformableCrossAttention, AEIM


# ==============================================================================
# [重构版] RobustFusion -> Geometric-Deformable Fusion (GDF) Module
# 适配 DDD17 模型，保持与主模型一致的创新点：DCAR + Deformable + RRF
# ==============================================================================
class RobustFusion(nn.Module):
    def __init__(self, dim_rgb, dim_evt, num_heads):
        super().__init__()
        self.dim_rgb = dim_rgb
        self.dim_evt = dim_evt

        # --- 1. Dilated Context-Aware Recalibration (DCAR) ---
        # [创新点] 替代原本的 Global Pooling。利用空洞卷积提取像素级上下文权重。
        total_dim = dim_rgb + dim_evt
        self.context_encoder = nn.Sequential(
            nn.Conv2d(total_dim, total_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(total_dim // 4),
            nn.ReLU(True),
            # dilation=2 扩大感受野，适配 DDD17 较低的分辨率也能捕捉全局信息
            nn.Conv2d(total_dim // 4, total_dim, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.Sigmoid()
        )

        # --- 通道对齐投影 ---
        if dim_evt != dim_rgb:
            self.proj_evt = nn.Sequential(
                nn.Conv2d(dim_evt, dim_rgb, 1, bias=False),
                nn.BatchNorm2d(dim_rgb)
            )
        else:
            self.proj_evt = nn.Identity()

        # --- 2. Geometry-Guided Interaction ---
        # [核心] Deformable Attention 用于解决 DDD17 中严重的运动模糊和不对齐
        self.deform_attn = DeformableCrossAttention(dim_rgb, num_heads=num_heads)

        # --- 3. Residual Refinement Fusion (RRF) ---
        # [创新点] 替代 Softmax 门控。
        # 逻辑：RGB 提供基础纹理，Event 提供基于置信度的残差修补。
        self.confidence_gate = nn.Sequential(
            nn.Conv2d(dim_rgb * 2, dim_rgb // 2, 1, bias=False),
            nn.BatchNorm2d(dim_rgb // 2),
            nn.ReLU(True),
            nn.Conv2d(dim_rgb // 2, 1, 1),  # 输出单通道 Confidence Map
            nn.Sigmoid()
        )

        self.proj = nn.Sequential(
            nn.Conv2d(dim_rgb, dim_rgb, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_rgb),
            nn.ReLU(True)
        )

    def forward(self, rgb, evt):
        # -----------------------------------------------
        # Step 1: DCAR (Pixel-wise Recalibration)
        # -----------------------------------------------
        cat_feat = torch.cat([rgb, evt], dim=1)

        # 生成权重图 [B, C_total, H, W]
        attn_map = self.context_encoder(cat_feat)

        rgb_rec = rgb * attn_map[:, :self.dim_rgb, :, :]
        evt_rec = evt * attn_map[:, self.dim_rgb:, :, :]

        # -----------------------------------------------
        # Step 2: Alignment & Interaction
        # -----------------------------------------------
        evt_aligned = self.proj_evt(evt_rec)

        # Deformable Interaction: 利用 RGB 引导 Event 对齐
        feat_aligned = self.deform_attn(rgb_rec, evt_aligned)

        # -----------------------------------------------
        # Step 3: Residual Refinement Fusion
        # -----------------------------------------------
        # 计算 Event 特征的置信度
        fusion_input = torch.cat([rgb_rec, feat_aligned], dim=1)
        confidence = self.confidence_gate(fusion_input)  # [B, 1, H, W]

        # 残差融合：RGB + (Confidence * Event_Aligned)
        fused = rgb_rec + (confidence * feat_aligned)

        return self.proj(fused)


# ==============================================================================
# MLP Decoder & Head (保持通用结构)
# ==============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    def __init__(self, in_channels_list, embedding_dim=256, num_classes=6):
        super().__init__()
        c1, c2, c3, c4 = in_channels_list

        self.linear_c4 = MLP(input_dim=c4, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(True)
        )

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, x1, x2, x3, x4):
        c1, c2, c3, c4 = x1, x2, x3, x4
        target_size = c1.shape[2:]

        def process(feat, mlp_layer):
            n, _, h, w = feat.shape
            _c = mlp_layer(feat).permute(0, 2, 1).reshape(n, -1, h, w)
            if _c.shape[2:] != target_size:
                _c = F.interpolate(_c, size=target_size, mode='bilinear', align_corners=False)
            return _c

        _c4 = process(c4, self.linear_c4)
        _c3 = process(c3, self.linear_c3)
        _c2 = process(c2, self.linear_c2)
        _c1 = process(c1, self.linear_c1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.classifier(x)
        return x


# ==============================================================================
# [主模型] EGHFNet (DDD17 Version)
# ==============================================================================
class EGHFNet(nn.Module):
    def __init__(self, num_classes=6, num_bins=3):
        super().__init__()
        self.num_bins = num_bins

        self.rgb_chans = [64, 128, 320, 512]  # mit_b2
        self.evt_chans = [32, 64, 160, 256]  # mit_b0

        # 1. AEIM (调用 network/modules.py 中的新版 AEIM)
        # 注意：DDD17 中我们保持 out_dim=32 以适配 mit_b0 的通道数
        self.aeim = AEIM(in_dim=1, out_dim=32, num_bins=self.num_bins)

        # 2. Backbone (Dual SegFormer)
        # DDD17 特殊设置：event_in_channels=3，利用 ImageNet 权重
        self.backbone = DualSegFormerBackbone(event_in_channels=3)

        # 3. Fusion (使用新的 RobustFusion)
        self.fusion1 = RobustFusion(dim_rgb=self.rgb_chans[0], dim_evt=self.evt_chans[0], num_heads=4)
        self.fusion2 = RobustFusion(dim_rgb=self.rgb_chans[1], dim_evt=self.evt_chans[1], num_heads=4)
        self.fusion3 = RobustFusion(dim_rgb=self.rgb_chans[2], dim_evt=self.evt_chans[2], num_heads=8)
        self.fusion4 = RobustFusion(dim_rgb=self.rgb_chans[3], dim_evt=self.evt_chans[3], num_heads=8)

        # 4. Decoder
        self.decoder = SegFormerHead(in_channels_list=self.rgb_chans,
                                     embedding_dim=256,
                                     num_classes=num_classes)

    def forward(self, rgb, evt):
        # 1. RGB 处理策略: 伪装成 3 通道
        if rgb.shape[1] == 1:
            x_rgb = rgb.repeat(1, 3, 1, 1)
        else:
            x_rgb = rgb

        # 2. Event 处理策略: AEIM -> Direct Backbone
        # 输入 evt: [B, 2*num_bins, H, W]
        if evt.shape[1] == 2 * self.num_bins:
            x_ev = evt[:, :self.num_bins, :, :]  # Voxel
            x_map = evt[:, self.num_bins:, :, :]  # Activity Map

            # 新版 AEIM 会自动进行非对称卷积增强和时空校准
            x_evt = self.aeim(x_ev, x_map)
        else:
            # Fallback (仅测试用)
            x_evt = evt[:, :3, :, :]

        input_size = rgb.shape[2:]

        # 3. Backbone
        r_feats, e_feats = self.backbone(x_rgb, x_evt)

        # 4. Fusion (GDF Modules)
        f1 = self.fusion1(r_feats[0], e_feats[0])
        f2 = self.fusion2(r_feats[1], e_feats[1])
        f3 = self.fusion3(r_feats[2], e_feats[2])
        f4 = self.fusion4(r_feats[3], e_feats[3])

        # 5. Decode
        logits = self.decoder(f1, f2, f3, f4)
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)

        return logits