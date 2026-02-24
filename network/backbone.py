import torch
import torch.nn as nn
import os
import math
from functools import partial
from timm.layers import DropPath, to_2tuple, trunc_normal_
from configs.config import config
from collections import OrderedDict


# ==============================================================================
# Part 1: SegFormer (MixVisionTransformer) 基础组件
# ==============================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1): x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2): x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3): x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4): x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


def mit_b0(in_chans=3, **kwargs):
    model = MixVisionTransformer(
        patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0, drop_path_rate=0.1, in_chans=in_chans)
    return model


def mit_b2(in_chans=3, **kwargs):
    model = MixVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0, drop_path_rate=0.1, in_chans=in_chans)
    return model


# ==============================================================================
# Part 2: Dual Backbone with HUGGINGFACE AUTO-CONVERSION
# ==============================================================================

class DualSegFormerBackbone(nn.Module):
    def __init__(self, event_in_channels=12):
        super().__init__()

        # 1. RGB Branch
        rgb_type = getattr(config, 'RGB_BACKBONE', 'b2')
        print(f"[DualBackbone] Initializing RGB Branch: SegFormer-{rgb_type.upper()}")

        if rgb_type == 'b0':
            self.rgb_net = mit_b0(in_chans=3)
        else:
            self.rgb_net = mit_b2(in_chans=3)

        self._load_pretrained_mapped(self.rgb_net, config.PRETRAINED_RGB_PATH)

        # 2. Event Branch
        evt_type = getattr(config, 'EVENT_BACKBONE', 'b0')
        print(f"[DualBackbone] Initializing Event Branch: SegFormer-{evt_type.upper()}")

        if evt_type == 'b0':
            self.evt_net = mit_b0(in_chans=event_in_channels)
        else:
            self.evt_net = mit_b2(in_chans=event_in_channels)

        # Load Weights for Event Branch (with mapping)
        print(f"[Optimization] Loading Pretrained weights for Event Branch ({evt_type.upper()})...")
        self._load_pretrained_mapped(self.evt_net, config.PRETRAINED_EVT_PATH)

        # Event Input Layer Init
        nn.init.kaiming_normal_(self.evt_net.patch_embed1.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.evt_net.patch_embed1.proj.bias is not None:
            nn.init.zeros_(self.evt_net.patch_embed1.proj.bias)

    def _load_pretrained_mapped(self, model, path):
        """
        自动检测并转换 HuggingFace 格式的权重到 timm 格式
        """
        path = str(path)
        if not os.path.exists(path):
            print(f"[Warning] Weights not found at {path}. Training from scratch.")
            return

        print(f"[Smart Load] Loading weights from: {path}")
        try:
            checkpoint = torch.load(path, map_location='cpu')
            if 'state_dict' in checkpoint:
                sd = checkpoint['state_dict']
            elif 'model' in checkpoint:
                sd = checkpoint['model']
            else:
                sd = checkpoint

            # --- 开始映射转换 (HuggingFace -> Timm) ---
            new_sd = OrderedDict()
            mapped_count = 0

            # 临时存储 key/value 以便合并
            temp_store = {}

            for k, v in sd.items():
                # 1. 忽略解码器头部
                if k.startswith('decode_head'):
                    continue

                new_k = k
                # 2. 移除前缀
                if new_k.startswith('segformer.encoder.'):
                    new_k = new_k.replace('segformer.encoder.', '')

                # 3. Patch Embeddings
                if 'patch_embeddings.0' in new_k:
                    new_k = new_k.replace('patch_embeddings.0', 'patch_embed1')
                elif 'patch_embeddings.1' in new_k:
                    new_k = new_k.replace('patch_embeddings.1', 'patch_embed2')
                elif 'patch_embeddings.2' in new_k:
                    new_k = new_k.replace('patch_embeddings.2', 'patch_embed3')
                elif 'patch_embeddings.3' in new_k:
                    new_k = new_k.replace('patch_embeddings.3', 'patch_embed4')

                # Norms
                if 'patch_embed' in new_k and 'layer_norm' in new_k:
                    new_k = new_k.replace('layer_norm', 'norm')

                # 4. Blocks
                if 'block.0.' in new_k:
                    new_k = new_k.replace('block.0.', 'block1.')
                elif 'block.1.' in new_k:
                    new_k = new_k.replace('block.1.', 'block2.')
                elif 'block.2.' in new_k:
                    new_k = new_k.replace('block.2.', 'block3.')
                elif 'block.3.' in new_k:
                    new_k = new_k.replace('block.3.', 'block4.')

                # 5. Layer Norms (Stage Norms)
                if 'layer_norm.0' in new_k:
                    new_k = new_k.replace('layer_norm.0', 'norm1')
                elif 'layer_norm.1' in new_k:
                    new_k = new_k.replace('layer_norm.1', 'norm2')
                elif 'layer_norm.2' in new_k:
                    new_k = new_k.replace('layer_norm.2', 'norm3')
                elif 'layer_norm.3' in new_k:
                    new_k = new_k.replace('layer_norm.3', 'norm4')

                # 6. Block Internals
                if 'layer_norm_1' in new_k: new_k = new_k.replace('layer_norm_1', 'norm1')
                if 'layer_norm_2' in new_k: new_k = new_k.replace('layer_norm_2', 'norm2')
                if 'mlp.dense1' in new_k: new_k = new_k.replace('mlp.dense1', 'mlp.fc1')
                if 'mlp.dense2' in new_k: new_k = new_k.replace('mlp.dense2', 'mlp.fc2')

                # 7. Attention Components
                if 'attention.self.query' in new_k:
                    new_k = new_k.replace('attention.self.query', 'attn.q')
                    new_sd[new_k] = v
                elif 'attention.self.key' in new_k:
                    # 暂存 Key
                    temp_key = new_k.replace('attention.self.key', 'attn.kv')
                    # 逻辑: 需要找到对应的 value 合并
                    if temp_key not in temp_store: temp_store[temp_key] = {}
                    temp_store[temp_key]['key'] = v
                elif 'attention.self.value' in new_k:
                    # 暂存 Value
                    temp_key = new_k.replace('attention.self.value', 'attn.kv')
                    if temp_key not in temp_store: temp_store[temp_key] = {}
                    temp_store[temp_key]['value'] = v
                elif 'attention.output.dense' in new_k:
                    new_k = new_k.replace('attention.output.dense', 'attn.proj')
                    new_sd[new_k] = v
                elif 'attention.self.sr' in new_k:
                    new_k = new_k.replace('attention.self.sr', 'attn.sr')
                    new_sd[new_k] = v
                elif 'attention.self.layer_norm' in new_k:
                    new_k = new_k.replace('attention.self.layer_norm', 'attn.norm')
                    new_sd[new_k] = v
                else:
                    new_sd[new_k] = v

            # 8. 合并 Key 和 Value
            for k, val_dict in temp_store.items():
                if 'key' in val_dict and 'value' in val_dict:
                    # Cat dim=0 (Output features dim)
                    new_sd[k] = torch.cat([val_dict['key'], val_dict['value']], dim=0)
                    mapped_count += 1

            # --- 最终加载 ---
            model_state = model.state_dict()
            filtered_dict = {}
            ignored_keys = []

            for k, v in new_sd.items():
                if k in model_state:
                    if v.shape == model_state[k].shape:
                        filtered_dict[k] = v
                    else:
                        ignored_keys.append(f"{k} (shape {v.shape} != {model_state[k].shape})")
                # else: print(f"Unused key: {k}") # Debug only

            msg = model.load_state_dict(filtered_dict, strict=False)

            # Report
            real_missing = [k for k in msg.missing_keys if "head" not in k and "decode" not in k]
            print(f"✅ Auto-converted and loaded {len(filtered_dict)} keys.")
            if len(real_missing) > 0:
                print(f"⚠️  Still missing {len(real_missing)} keys (e.g. {real_missing[:3]}).")
            else:
                print("🎉 Perfect Match!")

        except Exception as e:
            print(f"Weight loading failed: {e}")

    def forward(self, rgb, evt):
        rgb_feats = self.rgb_net(rgb)
        evt_feats = self.evt_net(evt)
        return rgb_feats, evt_feats