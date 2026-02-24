import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from configs.config import config
from tools.loader import get_dataloaders
from network.model import EGHFNet
from tools.utils import RunningConfusionMatrix
import os
import sys
import time
import argparse

# 防止显存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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


# [保留] Warmup PolyLR (这对 Transformer 很重要)
class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, warmup_iters=1500, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / float(max(1, self.warmup_iters))
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            factor = ((1 - (self.last_epoch - self.warmup_iters) / (self.max_iters - self.warmup_iters)) ** self.power)
            return [max(base_lr * factor, self.min_lr) for base_lr in self.base_lrs]


# [新增] 内置 OHEM Loss (解决难样本学习问题)
class OhemCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True):
        super(OhemCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)

        # DSEC 专用类别权重 (Stage 2 Hard Mining)
        # Class Order: 0:Back, 1:Build, 2:Fence, 3:Person, 4:Pole, 5:Road, 6:Walk, 7:Veg, 8:Car, 9:Wall, 10:Sign
        if use_weight:
            # 适度加权：降低 Wall/Fence 的极端权重，防止 Loss 震荡
            weights = torch.tensor([1.0, 1.2, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0]).to(config.DEVICE)
            print(f"⚡ [Loss] Initialized with Class Weights: {weights.tolist()}")
            self.criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index, reduction='none')
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, pred, target):
        # 1. 计算所有像素的 Loss
        loss = self.criterion(pred, target).view(-1)

        # 2. 排序并筛选 Top-K 难样本
        # min_kept 保证至少保留一定数量的像素(防止初期 Loss 全部小于阈值导致梯度为0)
        loss, _ = torch.sort(loss, descending=True)

        if loss[self.min_kept] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.min_kept]

        return torch.mean(loss)


def smart_init_event_stem(model):
    print("⚡ [Smart Init] Initializing Event Stem with RGB weights...")
    try:
        rgb_stem = model.backbone.rgb_net.patch_embed1.proj
        evt_stem = model.backbone.evt_net.patch_embed1.proj
        with torch.no_grad():
            rgb_mean = rgb_stem.weight.mean(dim=1, keepdim=True)
            # 自动适配通道数差异
            if rgb_mean.shape[0] != evt_stem.weight.shape[0]:
                if rgb_mean.shape[0] > evt_stem.weight.shape[0]:
                    rgb_mean = rgb_mean[:evt_stem.weight.shape[0]]
                else:
                    repeats = (evt_stem.weight.shape[0] // rgb_mean.shape[0]) + 1
                    rgb_mean = rgb_mean.repeat(repeats, 1, 1, 1)[:evt_stem.weight.shape[0]]

            evt_stem.weight.copy_(rgb_mean.repeat(1, evt_stem.in_channels, 1, 1))

            if rgb_stem.bias is not None and evt_stem.bias is not None:
                rgb_bias = rgb_stem.bias
                if rgb_bias.shape[0] > evt_stem.bias.shape[0]:
                    evt_stem.bias.copy_(rgb_bias[:evt_stem.bias.shape[0]])
                else:
                    evt_stem.bias.copy_(rgb_bias)
        print("✅ Event Stem initialized!")
    except AttributeError as e:
        print(f"⚠️ [Smart Init Warning] Skipping: {e}")


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

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{current_lr:.2e}"})

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
            preds = torch.argmax(logits, dim=1)
            conf_mat.update(preds, mask)

    avg_miou, class_ious = conf_mat.compute()
    writer.add_scalar("Val/MIoU", avg_miou, epoch)
    print(f"\n>>> Global mIoU: {avg_miou:.4f} (Best: {max(best_miou, avg_miou):.4f}) <<<")

    # 打印详细 IoU 方便对比
    class_names = ['Back', 'Build', 'Fence', 'Person', 'Pole', 'Road', 'Walk', 'Veg', 'Car', 'Wall', 'Sign']
    for i, iou in enumerate(class_ious):
        name = class_names[i] if i < len(class_names) else str(i)
        print(f"{name:<6}: {iou:.4f}")

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
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.TXT_LOG_DIR, exist_ok=True)
    sys.stdout = Logger(os.path.join(config.TXT_LOG_DIR, f"log_{timestamp}.txt"))

    print(f"=== EGHF-Net Final Fix (Low LR + OHEM) ===")

    # [核心修改 1] 降低学习率，防止冲击
    config.LEARNING_RATE = 2e-4
    print(f"⚡ Learning Rate set to: {config.LEARNING_RATE}")

    # [核心修改 2] 保持 CopyPaste 开启 (Stage 2 默认已在 config.py 开启，这里确认一下)
    config.ENABLE_COPY_PASTE = True

    train_loader = get_dataloaders("train")
    val_loader = get_dataloaders("test")

    model = EGHFNet().to(config.DEVICE)
    smart_init_event_stem(model)

    # 差分学习率
    gdca_params = [p for n, p in model.named_parameters() if "offset_conv" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "offset_conv" not in n and p.requires_grad]
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': config.LEARNING_RATE},
        {'params': gdca_params, 'lr': config.LEARNING_RATE}  # 可以尝试 x2, 但先稳一手
    ], weight_decay=config.WEIGHT_DECAY)

    scaler = torch.amp.GradScaler('cuda')

    # Warmup + Poly
    total_iters = config.EPOCHS * len(train_loader)
    scheduler = WarmupPolyLR(optimizer, max_iters=total_iters, warmup_iters=1500, power=0.9)

    # [核心修改 3] 使用 OHEM Loss 替代普通 CrossEntropy
    # 计算 min_kept: Batch(4) * 440 * 640 * 10% ≈ 112,640
    pixels_per_batch = config.BATCH_SIZE * config.IMG_SIZE[0] * config.IMG_SIZE[1]
    min_kept = int(pixels_per_batch * 0.10)  # 保持 10%
    print(f"⚡ OHEM min_kept pixels: {min_kept}")

    criterion = OhemCrossEntropyLoss(
        ignore_index=255,
        thresh=0.7,
        min_kept=min_kept,
        use_weight=True
    )

    writer = SummaryWriter(config.LOG_DIR)
    start_epoch = 0
    best_miou = 0.0

    if args.resume and os.path.exists(config.LATEST_CKPT_PATH):
        print(f"Resuming from {config.LATEST_CKPT_PATH}...")
        ckpt = torch.load(config.LATEST_CKPT_PATH, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        best_miou = ckpt.get('miou', 0.0)
        scheduler.last_epoch = start_epoch * len(train_loader)
        new_lr = 1e-4
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"⚡ FORCE RESET Learning Rate to {new_lr} for Resuming!")

    for epoch in range(start_epoch, config.EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, writer, scaler)

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "miou": best_miou,
            "optimizer_state_dict": optimizer.state_dict(),
        }, config.LATEST_CKPT_PATH)

        if (epoch + 1) % config.VAL_INTERVAL == 0:
            _, new_best_miou = validate(model, val_loader, epoch, writer, best_miou, optimizer)
            if new_best_miou > best_miou:
                best_miou = new_best_miou
    writer.close()


if __name__ == "__main__":
    main()