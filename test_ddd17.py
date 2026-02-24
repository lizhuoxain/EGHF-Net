import torch
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2

# 1. 导入配置
from configs.config_ddd17 import config

# 2. 导入数据加载器
try:
    from tools.loader_DDD17 import get_dataloaders
except ImportError:
    print("❌ Error: Could not import 'loader_DDD17'.")
    sys.exit(1)

# 3. 导入模型与工具
from network.model_DDD17 import EGHFNet
from tools.utils import RunningConfusionMatrix, colorize_mask


def calculate_pixel_accuracy(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    total_pixels = np.sum(confusion_matrix)
    if total_pixels == 0: return 0.0
    return np.sum(intersection) / total_pixels


def save_single_prediction(logit, filename, save_dir):
    """
    保存单张预测结果
    logit: [C, H, W] (Single Image Tensor)
    """
    # 转为 numpy 索引图 [H, W]
    pred = torch.argmax(logit, dim=0).detach().cpu().numpy()

    # 上色
    pred_color = colorize_mask(pred)

    # 构造文件名
    name = Path(filename).stem
    save_path = os.path.join(save_dir, f"{name}.png")

    cv2.imwrite(save_path, pred_color)


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    # --- 1. 确定权重 ---
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = config.BEST_MIOU_CKPT_PATH
        if not ckpt_path.exists():
            potential = Path("checkpoints/_DDD17_Target76_EIFNet_Align/best_miou_checkpoint.pth")
            if potential.exists(): ckpt_path = potential

    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return

    print(f"\n=== 🚀 Testing DDD17 (Metrics + Save ALL Images) ===")
    print(f"   Checkpoint: {ckpt_path}")

    # --- 2. 模型与数据 ---
    model = EGHFNet(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    print(f"⚡ Loading weights...")
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k != 'n_averaged'}
    model.load_state_dict(new_state_dict, strict=False)

    # 获取测试集
    test_loader = get_dataloaders("test")
    print(f"📊 Total Batches: {len(test_loader)}")
    # 注意：这里 len 是 batch 数量，不是图片总数

    # --- 3. 准备保存路径 ---
    save_dir = config.RUNS_DIR / "predictions_all_frames"
    os.makedirs(save_dir, exist_ok=True)
    print(f"📂 Output Directory: {save_dir}")

    # --- 4. 初始化指标计算器 ---
    model.eval()
    conf_mat = RunningConfusionMatrix(config.NUM_CLASSES, ignore_label=255)

    print("Running Inference...")
    with torch.no_grad():
        pbar = tqdm(test_loader, ncols=100)
        for i, batch in enumerate(pbar):
            rgb = batch["rgb"].to(config.DEVICE)
            event = batch["event"].to(config.DEVICE)
            mask = batch["mask"].to(config.DEVICE)
            filenames = batch["filename"]  # 这是一个列表，长度为 Batch Size

            # 1. 推理
            logits = model(rgb, event)
            if isinstance(logits, tuple): logits = logits[0]

            # 2. 更新指标 (支持 Batch 计算)
            preds = torch.argmax(logits, dim=1)
            conf_mat.update(preds, mask)

            # 3. [核心修复] 遍历 Batch 中的每一张图片进行保存
            batch_size = logits.shape[0]
            for b in range(batch_size):
                # 提取单张图的 logit 和对应的文件名
                single_logit = logits[b]  # shape: [C, H, W]
                fname = filenames[b]  # 对应的文件名字符串

                save_single_prediction(single_logit, fname, save_dir)

    # --- 5. 计算并输出最终指标 ---
    miou, class_ious = conf_mat.compute()
    pa = calculate_pixel_accuracy(conf_mat.confusion_matrix)

    print(f"\n{'=' * 30}")
    print(f"🏆 Final DDD17 Results")
    print(f"{'=' * 30}")
    print(f"👉 mIoU: {miou:.4f}")
    print(f"👉 PA:   {pa:.4f}")
    print(f"{'-' * 30}")

    # 您的 6 个类别
    class_names = ['Back', 'Build', 'Fence', 'Person', 'Pole', 'Road']
    for i, iou in enumerate(class_ious):
        name = class_names[i] if i < len(class_names) else str(i)
        print(f"   {name:<8}: {iou:.4f}")
    print("-" * 30)
    print(f"✅ All images saved to: {save_dir}")


if __name__ == "__main__":
    test()