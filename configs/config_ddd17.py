import os
from pathlib import Path


class Config:
    def __init__(self):
        self.PROJECT_NAME = "EGHF-Net"
        self.NUM_CLASSES = 6
        # 标记为 SOTA 冲击版本
        self.VERSION_SUFFIX = f"_DDD17_Target76_EIFNet_Align_210"

        # 路径配置
        self.PREPROCESSED_ROOT = Path(r"F:\Downloads\datasets\DDD17-Events\dataset_DDD17-Events_our_codification\DDD17_Preprocessed_EGHF")
        self.PREPROCESSED_TRAIN = self.PREPROCESSED_ROOT / "train"
        self.PREPROCESSED_TEST = self.PREPROCESSED_ROOT / "test"

        self.PRETRAINED_RGB_PATH = Path("../pretrained/mit_b2.pth")
        self.PRETRAINED_EVT_PATH = Path("../pretrained/mit_b0.pth")

        self.RUNS_DIR = Path("./runs") / (self.PROJECT_NAME + self.VERSION_SUFFIX)
        self.CKPT_DIR = Path("./checkpoints") / self.VERSION_SUFFIX
        self.LOG_DIR = Path("./logs") / self.VERSION_SUFFIX
        self.TXT_LOG_DIR = Path("./txt_logs") / self.VERSION_SUFFIX
        self.LATEST_CKPT_PATH = self.CKPT_DIR / "latest_checkpoint.pth"
        self.BEST_MIOU_CKPT_PATH = self.CKPT_DIR / "best_miou_checkpoint.pth"

        for p in [self.RUNS_DIR, self.CKPT_DIR, self.LOG_DIR, self.TXT_LOG_DIR]:
            p.mkdir(parents=True, exist_ok=True)

        # ================= 模型参数 =================
        self.RGB_BACKBONE = 'b2'
        self.EVENT_BACKBONE = 'b0'

        # [策略] 使用 3 Bins (EIFNet/EISNet 通用设置)，通过 Adapter 映射
        self.NUM_BINS = 3
        self.EVENT_INPUT_CHANS = self.NUM_BINS * 2

        self.BACKBONE_CHANNELS = [64, 128, 320, 512]
        self.IMG_SIZE = (200, 346)

        # ================= 训练参数 (EIFNet SOTA Setting) =================
        # 预训练微调通常 LR 较小，Epoch 适中
        self.LEARNING_RATE = 2e-4
        self.EPOCHS = 100
        self.ACCUMULATION_STEPS = 1
        self.VAL_INTERVAL = 1
        self.DEVICE = "cuda:0"
        self.WEIGHT_DECAY = 0.01

        self.BATCH_SIZE = 16
        self.ENABLE_COPY_PASTE = True  # SOTA 模型通常需要强增强
        self.TRAIN_CROP_SIZE = (200, 346)
        self.MANUAL_CLASS_WEIGHTS = None
        self.OHEM_KEEP_RATIO = 1.0  # 先跑通 Baseline，再开 OHEM


config = Config()