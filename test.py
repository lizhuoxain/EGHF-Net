import torch
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2

# ==============================================================================
# [兼容性补丁] 解决 NumPy 1.x 读取 NumPy 2.0 权重的问题
# ==============================================================================
try:
    import numpy._core
except ImportError:
    sys.modules['numpy._core'] = np.core
    if hasattr(np.core, 'multiarray'):
        sys.modules['numpy._core.multiarray'] = np.core.multiarray

from configs.config import config
from tools.loader import get_dataloaders
from network.model import EGHFNet
from tools.utils import RunningConfusionMatrix, colorize_mask

# [重要] 强制覆盖配置以匹配权重文件
# 报错显示权重是 [32, 5, 7, 7]，说明训练时 num_bins=5
config.NUM_BINS = 5
config.EVENT_INPUT_CHANS = 10  # 5 Voxel + 5 Map

# 定义微调后的权重名称
FINETUNED_CKPT_NAME = "best_miou_checkpoint.pth"


def calculate_pixel_accuracy(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    total_pixels = np.sum(confusion_matrix)
    if total_pixels == 0: return 0.0
    return np.sum(intersection) / total_pixels


def save_single_prediction(logit, filename, save_dir):
    """保存单张预测结果 (上色后)"""
    pred = torch.argmax(logit, dim=0).detach().cpu().numpy()
    pred_color = colorize_mask(pred)
    name = Path(filename).stem
    save_path = os.path.join(save_dir, f"{name}.png")
    cv2.imwrite(save_path, pred_color)


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Manually specify checkpoint path")
    args = parser.parse_args()

    # 1. 确定权重路径
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = config.CKPT_DIR / FINETUNED_CKPT_NAME
        if not ckpt_path.exists():
            print(f"⚠️ Warning: Checkpoint not found at {ckpt_path}")
            # 自动回退逻辑
            if config.BEST_MIOU_CKPT_PATH.exists():
                print(f"   Falling back to {config.BEST_MIOU_CKPT_PATH}")
                ckpt_path = config.BEST_MIOU_CKPT_PATH

    print(f"=== Testing Model: {ckpt_path} ===")

    # 2. 模型初始化 (已强制 NUM_BINS=5)
    model = EGHFNet().to(config.DEVICE)

    # 3. 加载权重
    if ckpt_path.exists():
        print(f"⚡ Loading weights...")
        try:
            ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

            new_state_dict = {}
            for k, v in state_dict.items():
                if k == "n_averaged": continue
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict, strict=False)
            best_miou = ckpt.get('miou', 0.0) if isinstance(ckpt, dict) else 0.0
            print(f"✅ Weights loaded! (Recorded mIoU: {best_miou:.4f})")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            return
    else:
        print(f"❌ Error: Checkpoint file not found!")
        return

    # 4. 推理循环
    model.eval()
    conf_mat = RunningConfusionMatrix(config.NUM_CLASSES, ignore_label=255)

    vis_dir = config.RUNS_DIR / "predictions_dsec_all"
    os.makedirs(vis_dir, exist_ok=True)
    print(f"📂 Saving ALL predictions to: {vis_dir}")

    # 获取 Loader (使用 test split)
    test_loader = get_dataloaders("test")

    print("Running Inference...")
    with torch.no_grad():
        pbar = tqdm(test_loader, ncols=100)
        for i, batch in enumerate(pbar):
            rgb = batch["rgb"].to(config.DEVICE)
            event = batch["event"].to(config.DEVICE)
            mask = batch["mask"].to(config.DEVICE)

            # [核心修复] 通道修正逻辑
            # 模型权重期望 5 通道，但 Loader 可能输出 10 或 20
            # 我们强制取前 5 个通道 (通常是 Voxel Grid)
            if event.shape[1] > 5:
                event = event[:, :5, :, :]

            # 处理文件名
            if "filename" in batch:
                filenames = batch["filename"]
            else:
                bs = rgb.shape[0]
                filenames = [f"frame_{i * bs + b:06d}" for b in range(bs)]

            # 推理
            logits = model(rgb, event)
            if isinstance(logits, tuple): logits = logits[0]

            # 统计指标
            preds = torch.argmax(logits, dim=1)
            conf_mat.update(preds, mask)

            # 保存每一张图
            batch_size = logits.shape[0]
            for b in range(batch_size):
                save_single_prediction(logits[b], filenames[b], vis_dir)

    # 5. 输出结果
    miou, class_ious = conf_mat.compute()
    pa = calculate_pixel_accuracy(conf_mat.confusion_matrix)

    print(f"\n{'=' * 30}")
    print(f"🏆 Final DSEC Test Results")
    print(f"{'=' * 30}")
    print(f"👉 mIoU: {miou:.4f}")
    print(f"👉 PA:   {pa:.4f}")
    print(f"{'-' * 30}")

    class_names = ['Background', 'Building', 'Fence', 'Person', 'Pole', 'Road',
                   'Sidewalk', 'Vegetation', 'Car', 'Wall', 'TrafficSign']

    print("Class-wise IoUs:")
    for i, iou in enumerate(class_ious):
        name = class_names[i] if i < len(class_names) else str(i)
        print(f"{name:<12}: {iou:.4f}")


if __name__ == "__main__":
    test()