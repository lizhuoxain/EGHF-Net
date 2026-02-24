import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from configs.config_ddd17 import config

# ================= 配置区域 =================
# 输出路径
OUTPUT_ROOT = Path(r"F:\Downloads\datasets\DDD17-Events\dataset_DDD17-Events_our_codification\DDD17_Preprocessed_EGHF")

# 原始数据路径
DATA_PATHS = {
    "train": {
        "images": Path(r"F:\Downloads\datasets\DDD17-Events\dataset_DDD17-Events_our_codification\images\train"),
        "events": Path(r"F:\Downloads\datasets\DDD17-Events\dataset_DDD17-Events_our_codification\events_hdf5\events"),
        "labels": Path(r"F:\Downloads\datasets\DDD17-Events\dataset_DDD17-Events_our_codification\labels\train"),
    },
    "test": {
        "images": Path(r"F:\Downloads\datasets\DDD17-Events\dataset_DDD17-Events_our_codification\images\test"),
        "events": Path(r"F:\Downloads\datasets\DDD17-Events\dataset_DDD17-Events_our_codification\events_hdf5\rec1487417411_dvs_events_npy"),
        "labels": Path(r"F:\Downloads\datasets\DDD17-Events\dataset_DDD17-Events_our_codification\labels\test"),
    }
}

# 尺寸配置
DDD17_H, DDD17_W = 260, 346
NUM_BINS = config.NUM_BINS


def generate_aet_representation(events_np, H, W):
    """
    生成 AET 表征 (10通道):
    - Channels 0-4: Voxel Grid (带极性 p*w)
    - Channels 5-9: Activity Map (无极性 |1|*w)
    """
    if events_np.shape[0] == 0:
        return torch.zeros((NUM_BINS * 2, H, W), dtype=torch.float32)

    # 1. 提取数据
    t = events_np[:, 0].astype(int)
    x = events_np[:, 1].astype(int)
    y = events_np[:, 2].astype(float)
    p = events_np[:, 3].astype(int)

    # 坐标过滤
    valid_mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y, t, p = x[valid_mask], y[valid_mask], t[valid_mask], p[valid_mask]

    if len(t) == 0:
        return torch.zeros((NUM_BINS * 2, H, W), dtype=torch.float32)

    # 时间归一化
    t_min, t_max = t.min(), t.max()
    if t_max - t_min > 0:
        t_norm = (t - t_min) / (t_max - t_min)
    else:
        t_norm = np.zeros_like(t)

    # 极性处理 0/1 -> -1/1
    p = (p * 2 - 1) if p.min() >= 0 else p

    # 插值权重计算
    t_idx_float = t_norm * (NUM_BINS - 1)
    t_idx_floor = np.floor(t_idx_float).astype(int)
    t_idx_ceil = t_idx_floor + 1
    t_idx_ceil[t_idx_ceil >= NUM_BINS] = NUM_BINS - 1

    w_ceil = t_idx_float - t_idx_floor
    w_floor = 1.0 - w_ceil

    # 转 Tensor
    idx_floor = torch.tensor(t_idx_floor, dtype=torch.long)
    idx_ceil = torch.tensor(t_idx_ceil, dtype=torch.long)
    y_tens = torch.tensor(y, dtype=torch.long)
    x_tens = torch.tensor(x, dtype=torch.long)
    p_tens = torch.tensor(p, dtype=torch.float32)
    w_floor = torch.tensor(w_floor, dtype=torch.float32)
    w_ceil = torch.tensor(w_ceil, dtype=torch.float32)

    # Part A: Voxel Grid (5 Channels)
    voxel_grid = torch.zeros((NUM_BINS, H, W), dtype=torch.float32)
    voxel_grid.index_put_((idx_floor, y_tens, x_tens), p_tens * w_floor, accumulate=True)
    voxel_grid.index_put_((idx_ceil, y_tens, x_tens), p_tens * w_ceil, accumulate=True)

    # Part B: Activity Map (5 Channels) - 仅计数，不乘 p
    activity_map = torch.zeros((NUM_BINS, H, W), dtype=torch.float32)
    activity_map.index_put_((idx_floor, y_tens, x_tens), w_floor, accumulate=True)
    activity_map.index_put_((idx_ceil, y_tens, x_tens), w_ceil, accumulate=True)

    # 拼接 -> 10 Channels
    aet = torch.cat([voxel_grid, activity_map], dim=0)
    return aet


def process_dataset(split):
    paths = DATA_PATHS[split]
    output_dir = OUTPUT_ROOT / split
    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(list(paths["labels"].glob("*.png")))
    if not label_files:
        print(f"No labels found in {paths['labels']}")
        return

    print(f"Processing {split} set: {len(label_files)} samples -> AET (10 Channels)...")

    for label_path in tqdm(label_files):
        stem = label_path.stem

        # 匹配 Image
        img_candidates = list(paths["images"].glob(f"{stem}*"))
        if not img_candidates: continue
        img_path = img_candidates[0]

        # 匹配 Event
        evt_candidates = list(paths["events"].glob(f"{stem}*.npy"))
        if not evt_candidates: continue
        evt_path = evt_candidates[0]

        # 2. 读取与转换
        try:
            # Mask
            mask_np = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if mask_np is None: continue
            mask_t = torch.from_numpy(mask_np).long()

            # RGB
            rgb_np = cv2.imread(str(img_path))
            if rgb_np is None: continue
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
            rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1)

            # Event -> AET
            events_np = np.load(evt_path, allow_pickle=True)
            if events_np.dtype == np.object_:
                events_np = events_np.item()
            if len(events_np.shape) != 2 or events_np.shape[1] != 4:
                if events_np.shape[0] == 4: events_np = events_np.T

            # 生成 10 通道数据
            aet_t = generate_aet_representation(events_np, DDD17_H, DDD17_W)
            aet_t = aet_t.half()  # 存 float16 省空间

            data_dict = {
                "rgb": rgb_t,
                "event": aet_t,  # [10, H, W]
                "mask": mask_t,
                "filename": stem
            }
            torch.save(data_dict, output_dir / f"{stem}.pt")

        except Exception as e:
            print(f"Error {stem}: {e}")


if __name__ == "__main__":
    process_dataset("train")
    process_dataset("test")