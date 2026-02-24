import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms
from configs.config import config
import random
import cv2
import torch.nn.functional as F

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class CustomDataset(Dataset):
    def __init__(self, split="train"):
        self.split = split
        self.data_dir = config.PREPROCESSED_TRAIN if split == "train" else config.PREPROCESSED_TEST
        self.files = sorted(list(self.data_dir.rglob("*.pt")))
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # [新增] 确保一定有一个固定的 Crop 尺寸，否则 Scale Jitter 后无法组成 Batch
        # 优先读取 config.TRAIN_CROP_SIZE，如果没有则兜底使用 config.IMG_SIZE (440, 640)
        self.crop_size = getattr(config, 'TRAIN_CROP_SIZE', None) or getattr(config, 'IMG_SIZE', (440, 640))
        if self.crop_size is None:
            # 双重兜底，防止 config 里也没写
            self.crop_size = (440, 640)

        if len(self.files) == 0:
            print(f"Error: No .pt files found in {self.data_dir}")
        else:
            print(f"[{split.upper()}] Loaded {len(self.files)} files.")

    def __len__(self):
        return len(self.files)

    def _load_and_decompress(self, path):
        data = torch.load(path)

        # --- 1. RGB 处理 ---
        rgb = data['rgb']
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        rgb = (rgb - self.mean) / self.std

        # --- 2. Event (AET) 处理 ---
        evt = data['event']
        if evt.dtype == torch.float16:
            evt = evt.float()

        # AET 格式 (10通道: 5 Voxel + 5 Map) 处理
        if evt.shape[0] == 10:
            voxel = evt[:5, :, :]
            activity = evt[5:, :, :]

            # === [删除或注释掉这段] ===
            # mask_vox = (voxel != 0)
            # if mask_vox.any():
            #     std_vox = voxel[mask_vox].std()
            #     if std_vox > 0:
            #         voxel = voxel / std_vox
            # =========================

            # [新增] 简单的数值截断，防止离群点
            voxel = torch.clamp(voxel, min=0.0, max=10.0)

            activity = torch.log1p(activity)
            evt = torch.cat([voxel, activity], dim=0)
        else:
            mask_evt = (evt != 0).float()
            num_valid = mask_evt.sum()
            if num_valid > 0:
                mean = evt.sum() / num_valid
                std = torch.sqrt((evt ** 2).sum() / num_valid - mean ** 2 + 1e-6)
                evt = (evt - mean) / (std + 1e-6)
                evt = evt * mask_evt

        # --- 3. Mask 处理 ---
        mask = data['mask']
        if mask.dtype == torch.uint8:
            mask = mask.long()

        return rgb, evt, mask

    def __getitem__(self, idx):
        try:
            rgb, evt, mask = self._load_and_decompress(self.files[idx])

            # 尺寸对齐保护
            if mask.shape[0] < rgb.shape[1]:
                diff = rgb.shape[1] - mask.shape[0]
                rgb = rgb[:, :-diff, :]
                evt = evt[:, :-diff, :]

            # 维度保护
            if evt.dim() == 2: evt = evt.unsqueeze(0)

            # [关键修复] 通道数强制对齐
            target_chans = config.EVENT_INPUT_CHANS
            current_chans = evt.shape[0]

            if current_chans != target_chans:
                if current_chans * 2 == target_chans:
                    # 10 -> 20: 简单重复
                    half = current_chans // 2
                    vox = evt[:half]
                    act = evt[half:]
                    evt = torch.cat([vox, vox, act, act], dim=0)
                elif target_chans % current_chans == 0:
                    repeat_factor = target_chans // current_chans
                    evt = evt.repeat(repeat_factor, 1, 1)

            if self.split == "train":
                # --- 1. Scale Jitter (随机缩放) ---
                rand_scale = random.uniform(0.5, 2.0)
                h_orig, w_orig = rgb.shape[1], rgb.shape[2]
                new_h, new_w = int(h_orig * rand_scale), int(w_orig * rand_scale)

                # 双线性插值缩放 RGB/Event
                rgb = F.interpolate(rgb.unsqueeze(0), size=(new_h, new_w), mode='bilinear',
                                    align_corners=False).squeeze(0)
                evt = F.interpolate(evt.unsqueeze(0), size=(new_h, new_w), mode='bilinear',
                                    align_corners=False).squeeze(0)
                # 最近邻插值缩放 Mask
                mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(new_h, new_w),
                                     mode='nearest').squeeze(0).squeeze(0).long()

                # --- 2. Copy-Paste Augmentation ---
                if getattr(config, 'ENABLE_COPY_PASTE', True) and random.random() < 0.5:
                    rand_idx = random.randint(0, len(self.files) - 1)
                    src_rgb, src_evt, src_mask = self._load_and_decompress(self.files[rand_idx])

                    if src_evt.dim() == 2: src_evt = src_evt.unsqueeze(0)

                    # 同样的通道对齐
                    if src_evt.shape[0] != target_chans:
                        if src_evt.shape[0] * 2 == target_chans:
                            half = src_evt.shape[0] // 2
                            vox = src_evt[:half]
                            act = src_evt[half:]
                            src_evt = torch.cat([vox, vox, act, act], dim=0)
                        elif target_chans % src_evt.shape[0] == 0:
                            src_evt = src_evt.repeat(target_chans // src_evt.shape[0], 1, 1)

                    # [新增] 必须把 src 也缩放到当前 new_h, new_w，否则无法粘贴
                    if src_rgb.shape[1:] != (new_h, new_w):
                        src_rgb = F.interpolate(src_rgb.unsqueeze(0), size=(new_h, new_w), mode='bilinear',
                                                align_corners=False).squeeze(0)
                        src_evt = F.interpolate(src_evt.unsqueeze(0), size=(new_h, new_w), mode='bilinear',
                                                align_corners=False).squeeze(0)
                        src_mask = F.interpolate(src_mask.unsqueeze(0).unsqueeze(0).float(), size=(new_h, new_w),
                                                 mode='nearest').squeeze(0).squeeze(0).long()

                    target_classes = [2, 3, 4, 9, 10]
                    paste_mask = torch.zeros_like(src_mask, dtype=torch.bool)
                    for c in target_classes:
                        paste_mask |= (src_mask == c)

                    if paste_mask.any():
                        paste_mask_ex = paste_mask.unsqueeze(0)
                        rgb = torch.where(paste_mask_ex, src_rgb, rgb)
                        evt = torch.where(paste_mask_ex, src_evt, evt)
                        mask = torch.where(paste_mask, src_mask, mask)

                # --- 3. Random Crop (强制对齐尺寸) ---
                # [关键修正] 使用 self.crop_size 作为强制目标
                c_h, c_w = self.crop_size

                # 如果当前尺寸小于目标尺寸，进行 Padding
                pad_h = max(c_h - rgb.shape[1], 0)
                pad_w = max(c_w - rgb.shape[2], 0)
                if pad_h > 0 or pad_w > 0:
                    rgb = TF.pad(rgb, (0, 0, pad_w, pad_h), fill=0)
                    evt = TF.pad(evt, (0, 0, pad_w, pad_h), fill=0)
                    mask = TF.pad(mask, (0, 0, pad_w, pad_h), fill=255)  # Ignore Label

                # 随机裁剪到目标尺寸
                i, j, h, w = transforms.RandomCrop.get_params(rgb, output_size=(c_h, c_w))
                rgb = TF.crop(rgb, i, j, h, w)
                evt = TF.crop(evt, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)

                # --- 4. Flip ---
                if random.random() > 0.5:
                    rgb = TF.hflip(rgb)
                    evt = TF.hflip(evt)
                    mask = TF.hflip(mask)

            return {"rgb": rgb, "event": evt, "mask": mask, "filename": self.files[idx].stem}

        except Exception as e:
            # 打印详细错误方便调试
            print(f"Error loading index {idx}: {e}")
            # 如果出错，随机换一个索引重试，避免训练中断
            rand_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(rand_idx)


def get_dataloaders(split):
    ds = CustomDataset(split)
    bs = config.BATCH_SIZE if split == "train" else 8
    workers = 8
    # 训练时 Drop Last 防止最后不足一个 batch 导致问题
    return DataLoader(ds, batch_size=bs, shuffle=(split == "train"),
                      num_workers=workers, pin_memory=True,
                      drop_last=(split == "train"), persistent_workers=True)