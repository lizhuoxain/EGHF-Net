import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from configs.config_ddd17 import config
import random
import cv2
from pathlib import Path

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class CustomDataset(Dataset):
    def __init__(self, split="train"):
        self.split = split
        self.data_dir = config.PREPROCESSED_TRAIN if split == "train" else config.PREPROCESSED_TEST
        self.data_dir = Path(self.data_dir)
        self.files = sorted(list(self.data_dir.rglob("*.pt")))

        # 仅保留变量防止报错，不再使用 ImageNet 均值
        self.mean = torch.tensor([0.0]).view(1, 1, 1)
        self.std = torch.tensor([1.0]).view(1, 1, 1)

        self.class_indices = None
        if len(self.files) == 0:
            print(f"Error: No .pt files found in {self.data_dir}")
        else:
            print(f"[{split.upper()}] Loaded {len(self.files)} files.")

    def __len__(self):
        return len(self.files)

    def _load_and_decompress(self, path):
        data = torch.load(path, weights_only=False)

        # RGB 处理: uint8 -> float
        rgb = data['rgb']
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0

        # Event 处理: float16 -> float
        evt = data['event']
        if evt.dtype == torch.float16:
            evt = evt.float()

        # ================= 1. 尺寸对齐 (硬裁剪) =================
        if rgb.shape[1] == 260:
            rgb = rgb[:, :-60, :]  # [C, 200, 346]
        if evt.shape[1] == 260:
            evt = evt[:, :-60, :]

        # Mask 裁剪
        mask = data['mask']
        if mask.dtype == torch.uint8:
            mask = mask.long()
        if mask.shape[0] == 260:
            mask = mask[:-60, :]

        # ================= [核心修改] 图像处理策略 =================
        # 策略: 严格单通道 [1, H, W]
        # EISNet 在 DDD17 上使用的是单通道灰度图，不重复堆叠
        if rgb.shape[0] == 3:
            rgb = rgb.mean(dim=0, keepdim=True)

        # 注意：这里删除了原来的 rgb.repeat(3, 1, 1)，保持 1 通道！

        # ================= [核心修改] 事件处理策略 =================
        # 适配 config.NUM_BINS (3)。如果原始数据是 10 通道 (5 Voxel + 5 Map)，我们需要切片。

        target_bins = config.NUM_BINS  # 3

        # 假设数据格式是 [Voxel_1...Voxel_5, Map_1...Map_5] (Total 10)
        # 我们取前 3 个 Voxel 和前 3 个 Map
        if evt.shape[0] > 2 * target_bins:
            half_c = evt.shape[0] // 2  # 通常是 5

            voxel = evt[0:target_bins, :, :]  # 取前 3 个
            activity = evt[half_c:half_c + target_bins, :, :]  # 取中间段的前 3 个

            evt = torch.cat([voxel, activity], dim=0)  # [6, H, W]

        elif evt.shape[0] < 2 * target_bins:
            print(f"⚠️ Warning: Event channels {evt.shape[0]} < required {2 * target_bins}")

        return rgb, evt, mask

    def _random_scale_and_crop(self, rgb, evt, mask):
        """
        随机多尺度训练策略 (保持不变，禁止缩小)
        """
        h, w = rgb.shape[1], rgb.shape[2]
        scale = random.uniform(1.0, 1.75)

        new_h, new_w = int(h * scale), int(w * scale)

        # 双线性插值
        rgb = F.interpolate(rgb.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=True).squeeze(0)
        evt = F.interpolate(evt.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=True).squeeze(0)
        mask = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=(new_h, new_w),
                             mode='nearest').squeeze().long()

        final_rgb = torch.zeros((rgb.shape[0], h, w), dtype=rgb.dtype)  # 注意这里用 rgb.shape[0] 而不是 3
        final_evt = torch.zeros((evt.shape[0], h, w), dtype=evt.dtype)
        final_mask = torch.ones((h, w), dtype=mask.dtype) * 255

        pad_h = max(0, h - new_h)
        pad_w = max(0, w - new_w)
        crop_h = max(0, new_h - h)
        crop_w = max(0, new_w - w)

        start_h = random.randint(0, crop_h) if crop_h > 0 else 0
        start_w = random.randint(0, crop_w) if crop_w > 0 else 0
        offset_h = random.randint(0, pad_h) if pad_h > 0 else 0
        offset_w = random.randint(0, pad_w) if pad_w > 0 else 0
        copy_h = min(h, new_h)
        copy_w = min(w, new_w)

        final_rgb[:, offset_h:offset_h + copy_h, offset_w:offset_w + copy_w] = rgb[:, start_h:start_h + copy_h,
                                                                               start_w:start_w + copy_w]
        final_evt[:, offset_h:offset_h + copy_h, offset_w:offset_w + copy_w] = evt[:, start_h:start_h + copy_h,
                                                                               start_w:start_w + copy_w]
        final_mask[offset_h:offset_h + copy_h, offset_w:offset_w + copy_w] = mask[start_h:start_h + copy_h,
                                                                             start_w:start_w + copy_w]

        return final_rgb, final_evt, final_mask

    def __getitem__(self, idx):
        try:
            rgb, evt, mask = self._load_and_decompress(self.files[idx])

            if evt.dim() == 2: evt = evt.unsqueeze(0)

            if self.split == "train":
                # 1. Random Scale & Crop
                if random.random() < 0.5:
                    rgb, evt, mask = self._random_scale_and_crop(rgb, evt, mask)

                # 2. Flip Augmentation
                if random.random() > 0.5:
                    rgb = TF.hflip(rgb)
                    evt = TF.hflip(evt)
                    mask = TF.hflip(mask)

            return {"rgb": rgb, "event": evt, "mask": mask, "filename": self.files[idx].stem}

        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))


def get_dataloaders(split):
    ds = CustomDataset(split)
    bs = config.BATCH_SIZE if split == "train" else 8
    workers = 8
    return DataLoader(ds, batch_size=bs, shuffle=(split == "train"),
                      num_workers=workers, pin_memory=True,
                      drop_last=(split == "train"), persistent_workers=True)