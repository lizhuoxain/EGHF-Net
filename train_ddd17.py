import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import sys
import time
import argparse

# 导入你的项目模块
from configs.config_ddd17 import config
from tools.loader_DDD17 import get_dataloaders
from network.model_DDD17 import EGHFNet
from tools.utils import RunningConfusionMatrix
import random
import numpy as np

# 启用 AMP 显存优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_iters, power=1.0, last_epoch=-1, min_lr=0.0):
        self.total_iters = total_iters
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]
        decay_factor = ((1.0 - self.last_epoch / self.total_iters) / (
                1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [max(group["lr"] * decay_factor, self.min_lr) for group in self.optimizer.param_groups]


def ensure_backbone_weights(model):
    """
    [EIFNet 策略] 加载标准的 ImageNet 预训练权重。
    由于我们在 model_DDD17.py 中添加了 InputAdapter，Backbone 的输入层保持了原生的 3 通道，
    因此可以直接加载权重，无需修改网络结构。
    """
    print("\n Loading Standard Backbone Weights (EIFNet Strategy)...")
    path_rgb = config.PRETRAINED_RGB_PATH
    path_evt = config.PRETRAINED_EVT_PATH

    # Load RGB (mit_b2)
    if os.path.exists(path_rgb):
        try:
            # strict=False 是安全的，因为 Backbone 内部 key 是匹配的
            # 未匹配的 key (如 adapter, decoder) 将保持随机初始化
            model.backbone.rgb_net.load_state_dict(torch.load(path_rgb, map_location='cpu'), strict=False)
            print(f"   ✅ Loaded RGB Backbone (B2) from {path_rgb}")
        except Exception as e:
            print(f"    Failed to load RGB weights: {e}")
    else:
        print(f"   ⚠️ Warning: RGB Weights not found at {path_rgb}")

    # Load Event (mit_b0)
    if os.path.exists(path_evt):
        try:
            model.backbone.evt_net.load_state_dict(torch.load(path_evt, map_location='cpu',weights_only=False), strict=False)
            print(f"    Loaded Event Backbone (B0) from {path_evt}")
        except Exception as e:
            print(f"    Failed to load Event weights: {e}")
    else:
        print(f"    Warning: Event Weights not found at {path_evt}")


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, writer, scaler):
    model.train()
    total_loss_meter = 0.0
    accum_steps = config.ACCUMULATION_STEPS

    pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}/{config.EPOCHS}", ncols=120)
    optimizer.zero_grad()

    for i, batch in enumerate(pbar):
        rgb = batch["rgb"].to(config.DEVICE)
        event = batch["event"].to(config.DEVICE)
        mask = batch["mask"].to(config.DEVICE)

        with torch.amp.autocast('cuda'):
            logits = model(rgb, event)
            if isinstance(logits, tuple): logits = logits[0]

            loss = criterion(logits, mask)
            loss_norm = loss / accum_steps

        scaler.scale(loss_norm).backward()
        total_loss_meter += loss.item()

        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        pbar.set_postfix({"L": f"{loss.item():.4f}", "LR": f"{optimizer.param_groups[0]['lr']:.2e}"})

    avg_loss = total_loss_meter / len(train_loader)
    writer.add_scalar("Train/Loss", avg_loss, epoch)
    return avg_loss


def validate(model, val_loader, epoch, writer, best_miou, optimizer):
    model.eval()
    torch.cuda.empty_cache()
    conf_mat = RunningConfusionMatrix(config.NUM_CLASSES, ignore_label=255)

    print("Validating...")
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Val Ep {epoch + 1}", ncols=100)
        for batch in pbar:
            rgb = batch["rgb"].to(config.DEVICE)
            event = batch["event"].to(config.DEVICE)
            mask = batch["mask"].to(config.DEVICE)

            logits = model(rgb, event)
            if isinstance(logits, tuple): logits = logits[0]

            preds = torch.argmax(logits, dim=1)
            conf_mat.update(preds, mask)

    avg_miou, class_ious = conf_mat.compute()
    writer.add_scalar("Val/MIoU", avg_miou, epoch)
    print(f"\n>>> Global mIoU: {avg_miou:.4f} (Best: {max(best_miou, avg_miou):.4f}) <<<")

    class_names = ['Flat', 'Construct', 'Object', 'Nature', 'Human', 'Vehicle']
    for i, iou in enumerate(class_ious):
        name = class_names[i] if i < len(class_names) else str(i)
        print(f"{name:<9}: {iou:.4f}")

    if avg_miou > best_miou:
        print(f"🔥 Best model updated (mIoU: {avg_miou:.4f})")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "miou": avg_miou,
            "optimizer_state_dict": optimizer.state_dict(),
        }, config.BEST_MIOU_CKPT_PATH)
        return avg_miou, avg_miou

    return avg_miou, best_miou


def main():
    set_seed(42)
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.TXT_LOG_DIR, exist_ok=True)
    sys.stdout = Logger(os.path.join(config.TXT_LOG_DIR, f"log_DDD17_{timestamp}.txt"))

    print(f"=== EGHF-Net Target 76.56% (EIFNet Style) ===")
    print(f"   - Strategy: Full Pretraining with Input Adapters")
    print(f"   - Input: 1-ch RGB -> 3-ch Adapter | {config.NUM_BINS}-Bin Event -> 3-ch Adapter")
    print(f"   - Base LR: {config.LEARNING_RATE} | Epochs: {config.EPOCHS}")

    # 1. Dataset
    train_loader = get_dataloaders("train")
    val_loader = get_dataloaders("test")

    # 2. Model (Make sure InputAdapter is added in model_DDD17.py)
    model = EGHFNet(num_classes=config.NUM_CLASSES, num_bins=config.NUM_BINS).to(config.DEVICE)

    # 3. Initialization Logic
    # 策略变化：不再需要 surgical_stem_replacement，因为我们用 Adapter 适配了输入
    if args.resume:
        if os.path.exists(config.LATEST_CKPT_PATH):
            print(f"   🔄 Resuming from {config.LATEST_CKPT_PATH}...")
            ckpt = torch.load(config.LATEST_CKPT_PATH, map_location=config.DEVICE, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            print("   ⚠️ No checkpoint found. Starting fresh.")
            ensure_backbone_weights(model)
    else:
        # 正常训练：直接加载 ImageNet 权重
        ensure_backbone_weights(model)

    # 4. Optimizer with Layer-wise LR
    # 关键点：Backbone 使用较小 LR (如 6e-5)，新层 (Adapter/Head/Fusion) 使用较大 LR (如 6e-4)
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "backbone" in name:
            backbone_params.append(param)
        else:
            # 包括 adapter, aeim, fusion, decoder 等
            head_params.append(param)

    print(f"\n⚡ Optimizer Setup:")
    print(f"   - Backbone params: {len(backbone_params)} tensors (LR: {config.LEARNING_RATE})")
    print(f"   - Head/Adapter params: {len(head_params)} tensors (LR: {config.LEARNING_RATE * 10.0})")

    # 修正后的 Optimizer 定义
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.LEARNING_RATE},  # 骨干用小 LR
        {'params': head_params, 'lr': config.LEARNING_RATE * 10.0}  # AEIM/Head 用大 LR
    ], lr=config.LEARNING_RATE)  # 默认 lr，作为兜底

    scaler = torch.amp.GradScaler('cuda')

    # 5. Scheduler
    total_iters = config.EPOCHS * len(train_loader)
    scheduler = PolynomialLR(optimizer, total_iters=total_iters, power=1.0, min_lr=1e-5)

    # 6. Loss
    criterion = nn.CrossEntropyLoss(ignore_index=255).to(config.DEVICE)

    writer = SummaryWriter(config.LOG_DIR)
    start_epoch = 0
    best_miou = 0.0

    # 7. Restore Optimizer State (if Resuming)
    if args.resume and 'ckpt' in locals():
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch = ckpt['epoch']
                best_miou = ckpt.get('miou', 0.0)
                scheduler.last_epoch = start_epoch * len(train_loader)
                print(f"   ✅ Optimizer state restored. Start Epoch: {start_epoch + 1}")
            except Exception as e:
                print(f"   ⚠️ Failed to restore optimizer state: {e}. Starting with fresh optimizer.")

    # 8. Training Loop
    for epoch in range(start_epoch, config.EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, writer, scaler)

        # Save Latest
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "miou": best_miou,
            "optimizer_state_dict": optimizer.state_dict(),
        }, config.LATEST_CKPT_PATH)

        # Validation
        if (epoch + 1) % config.VAL_INTERVAL == 0:
            _, new_best_miou = validate(model, val_loader, epoch, writer, best_miou, optimizer)
            if new_best_miou > best_miou:
                best_miou = new_best_miou

    writer.close()
    print(f"Training Finished. Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()