import cv2
import torch
import numpy as np
import multiprocessing
import yaml
import traceback
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot
from configs.config import config
# ================= 配置区域 =================
# [请修改] 您的 DSEC 数据集根目录
DSEC_ROOT = Path(r"/root/autodl-tmp/DSEC")

# [请修改] 输出目录 (建议使用新目录，以免混淆)
OUTPUT_ROOT = Path(r"/root/autodl-tmp/DSEC/preprocessed_eisnet_aet")

# [SOTA核心] 目标分辨率：DSEC 有效区域 640x440
TARGET_W, TARGET_H = 640, 440

# [关键修改] EISNet AET 配置
# 我们使用 5 个时间切片。AET 会生成 2 倍通道数 (Voxel + Map)
# 所以最终保存的 event tensor 通道数为 5 * 2 = 10
NUM_TIME_BINS = config.NUM_BINS

# 并行进程数
NUM_WORKERS = 12


# ===========================================


class Transform:
    """辅助类：处理 3D 刚体变换 (旋转 + 平移)"""

    def __init__(self, translation: np.ndarray, rotation: Rot):
        if translation.ndim > 1:
            self._translation = translation.flatten()
        else:
            self._translation = translation
        self._rotation = rotation

    @staticmethod
    def from_rotation(rotation: Rot):
        return Transform(np.zeros(3), rotation)

    @staticmethod
    def from_transform_matrix(transform_matrix: np.ndarray):
        translation = transform_matrix[:3, 3]
        rotation = Rot.from_matrix(transform_matrix[:3, :3])
        return Transform(translation, rotation)

    def R(self):
        return self._rotation

    def inverse(self):
        rotation = self._rotation.inv()
        translation = -rotation.apply(self._translation)
        return Transform(translation, rotation)

    def __matmul__(self, other):
        rotation = self._rotation * other._rotation
        translation = self._rotation.apply(other._translation) + self._translation
        return Transform(translation, rotation)


def get_accurate_mapping(calib_path):
    """计算 RGB -> Event 的像素映射表"""
    try:
        with open(calib_path, 'r') as f:
            conf = yaml.safe_load(f)

        K_r0 = np.eye(3)
        K_r0[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect0']['camera_matrix']  # Event

        K_r1 = np.eye(3)
        K_r1[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect1']['camera_matrix']  # RGB

        R_r0_0 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect0']))
        R_r1_1 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect1']))
        T_1_0 = Transform.from_transform_matrix(np.array(conf['extrinsics']['T_10']))

        T_r0_0 = Transform.from_rotation(R_r0_0)
        T_r1_1 = Transform.from_rotation(R_r1_1)
        T_r1_r0 = T_r1_1 @ T_1_0 @ T_r0_0.inverse()

        R_r1_r0_matrix = T_r1_r0.R().as_matrix()
        P_r1_r0 = K_r1 @ R_r1_r0_matrix @ np.linalg.inv(K_r0)

        ht, wd = 480, 640
        coords = np.stack(np.meshgrid(np.arange(wd), np.arange(ht)), axis=-1)
        coords_hom = np.concatenate((coords, np.ones((ht, wd, 1))), axis=-1)

        mapping = (P_r1_r0 @ coords_hom[..., None]).squeeze()
        mapping = (mapping / mapping[..., -1][..., None])[..., :2]

        return mapping.astype('float32')
    except Exception as e:
        return None


def load_events_robust(path):
    """健壮的事件加载函数"""
    try:
        data = np.load(str(path), allow_pickle=True)
        if data.ndim == 0: data = data.item()
        if isinstance(data, dict):
            if 'x' in data:
                return np.stack([data['x'].flatten(), data['y'].flatten(), data['t'].flatten(), data['p'].flatten()],
                                axis=1).astype(np.float32)
        if isinstance(data, np.ndarray):
            return data.astype(np.float32) if data.shape[1] == 4 else data.T.astype(np.float32)
        return None
    except:
        return None


def events_to_aet(events, num_bins, height, width):
    """
    [修改] 生成 EISNet 所需的 AET 表征 (Voxel Grid + Activity Map)
    输出通道数: 2 * num_bins (前一半是 Voxel, 后一半是 Map)
    """
    # 默认全0
    if events is None or len(events) == 0:
        return np.zeros((2 * num_bins, height, width), dtype=np.float32)

    x = events[:, 0].astype(int)
    y = events[:, 1].astype(int)
    t = events[:, 2]
    p = events[:, 3]

    valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t, p = x[valid_mask], y[valid_mask], t[valid_mask], p[valid_mask]

    if len(x) == 0:
        return np.zeros((2 * num_bins, height, width), dtype=np.float32)

    # 归一化时间戳
    t_min, t_max = t.min(), t.max()
    if t_max == t_min:
        t_norm = np.zeros_like(t, dtype=np.float32)
    else:
        t_norm = (t - t_min) / (t_max - t_min)

    # 缩放到 bin 索引范围 (0 到 num_bins-1)
    t_scaled = t_norm * (num_bins - 1)

    # === 1. 构建 Voxel Grid (双线性插值) ===
    voxel_grid = np.zeros((num_bins, height, width), dtype=np.float32).ravel()

    p = np.where(p == 0, -1, p)  # 确保极性是 -1, 1

    tis = t_scaled.astype(int)
    dts = t_scaled - tis

    # 左边 bin 的值
    vals_left = np.abs(p) * (1.0 - dts) * np.sign(p)
    idx_left = y * width + x + tis * (height * width)
    # 边界检查
    valid_left = (tis >= 0) & (tis < num_bins)
    np.add.at(voxel_grid, idx_left[valid_left], vals_left[valid_left])

    # 右边 bin 的值
    vals_right = np.abs(p) * dts * np.sign(p)
    idx_right = y * width + x + (tis + 1) * (height * width)
    valid_right = (tis + 1) < num_bins
    np.add.at(voxel_grid, idx_right[valid_right], vals_right[valid_right])

    voxel_grid = voxel_grid.reshape((num_bins, height, width))

    # === 2. 构建 Activity Map (仅计数，无符号) ===
    # 用于 AEIM 模块的注意力机制
    activity_map = np.zeros((num_bins, height, width), dtype=np.float32).ravel()

    # 简单的线性插值计数
    count_left = (1.0 - dts)
    count_right = dts

    np.add.at(activity_map, idx_left[valid_left], count_left[valid_left])
    np.add.at(activity_map, idx_right[valid_right], count_right[valid_right])

    activity_map = activity_map.reshape((num_bins, height, width))

    # 拼接: 通道数变为 2 * bins (5+5=10)
    return np.concatenate((voxel_grid, activity_map), axis=0)


def process_single_file(args):
    """单个样本处理函数"""
    rgb_path, evt_npy_path, label_path, calib_path, save_path = args

    if save_path.exists(): return

    try:
        # 1. 计算映射表
        map_x_y = get_accurate_mapping(str(calib_path))
        if map_x_y is None: return

        # 2. RGB 处理
        if not rgb_path.exists(): return
        rgb = cv2.imread(str(rgb_path))
        if rgb is None: return

        rgb_aligned = cv2.remap(rgb, map_x_y, None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rgb_crop = rgb_aligned[:TARGET_H, :TARGET_W, :]
        rgb_t = torch.from_numpy(rgb_crop).permute(2, 0, 1).float() / 255.0

        # 3. Event 处理 (改为 AET)
        events = load_events_robust(evt_npy_path)
        # 生成 AET (10通道: 5 Voxel + 5 Map)
        aet = events_to_aet(events, NUM_TIME_BINS, 480, 640)
        # 裁剪
        aet_crop = aet[:, :TARGET_H, :TARGET_W]
        evt_t = torch.from_numpy(aet_crop)

        # 4. Label 处理
        mask_t = torch.full((TARGET_H, TARGET_W), 255, dtype=torch.long)
        if label_path.exists():
            lbl = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if lbl is not None:
                h_lbl, w_lbl = lbl.shape
                if h_lbl == TARGET_H and w_lbl == TARGET_W:
                    mask_t = torch.from_numpy(lbl).long()
                elif h_lbl == 480 and w_lbl == 640:
                    lbl_crop = lbl[:TARGET_H, :TARGET_W]
                    mask_t = torch.from_numpy(lbl_crop).long()
                elif h_lbl > 900:
                    lbl_aligned = cv2.remap(lbl, map_x_y, None, interpolation=cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                    lbl_crop = lbl_aligned[:TARGET_H, :TARGET_W]
                    mask_t = torch.from_numpy(lbl_crop).long()

        # 5. 保存
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "rgb": rgb_t,
            "event": evt_t,  # 现在是 AET 格式
            "mask": mask_t,
            "filename": save_path.stem
        }, save_path)

    except Exception as e:
        print(f"Error processing {save_path.name}: {e}")
        traceback.print_exc()


def scan_dataset_and_build_tasks():
    tasks = []
    print("Scanning dataset directory...")
    for split in ["train", "test"]:
        split_root = DSEC_ROOT / split
        img_root = split_root / "images"
        evt_root = split_root / "events"
        lbl_root = split_root / "labels"
        calib_root = split_root / "calibration"
        out_split_root = OUTPUT_ROOT / split

        if not img_root.exists(): continue
        sequences = sorted([d.name for d in img_root.iterdir() if d.is_dir()])

        for seq in sequences:
            p_rgb_dir = img_root / seq / "left" / "rectified"
            if not p_rgb_dir.exists(): p_rgb_dir = img_root / seq

            p_evt_dir = evt_root / seq / "left"
            if not p_evt_dir.exists(): p_evt_dir = evt_root / seq

            p_lbl_dir = lbl_root / seq / "11classes"
            if not p_lbl_dir.exists(): p_lbl_dir = lbl_root / "semantic" / "left" / "11classes" / seq
            if not p_lbl_dir.exists(): p_lbl_dir = lbl_root / seq

            p_calib = calib_root / seq / "cam_to_cam.yaml"
            p_save_dir = out_split_root / seq

            if not p_rgb_dir.exists() or not p_calib.exists(): continue

            rgb_files = sorted(list(p_rgb_dir.glob("*.png")))
            print(f"Adding sequence: {seq:<20} | Frames: {len(rgb_files)}")

            for f in rgb_files:
                idx = f.stem
                tasks.append((f, p_evt_dir / f"{idx}.npy", p_lbl_dir / f"{idx}.png", p_calib, p_save_dir / f"{idx}.pt"))

    return tasks


def main():
    if not OUTPUT_ROOT.exists(): OUTPUT_ROOT.mkdir(parents=True)
    print(f"=== EISNet AET Preprocessing ===")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Time Bins: {NUM_TIME_BINS} (Total Channels: {NUM_TIME_BINS * 2})")

    tasks = scan_dataset_and_build_tasks()
    if len(tasks) == 0:
        print("No files found!")
        return

    print(f"Starting {NUM_WORKERS} workers...")
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(process_single_file, tasks), total=len(tasks)))
    print("Done.")


if __name__ == "__main__":
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    main()