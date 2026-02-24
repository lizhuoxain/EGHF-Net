import os
from pathlib import Path
import torch


class Config:
    def __init__(self):
        # ================= 项目基础配置 =================
        self.PROJECT_NAME = "EGHF-Net"
        self.DSEC_CLASS_COUNT = 11
        # [版本号] Adaptive Dual-Stage (Fixed for 440x640)
        self.VERSION_SUFFIX = f"_DSEC_{self.DSEC_CLASS_COUNT}cls_RGB-B2_Event-B0_Adaptive_DualStage_27"

        # ================= 路径配置 =================
        self.DSEC_ROOT = Path(r"E:\Datasets\DSEC")
        self.PREPROCESSED_ROOT = self.DSEC_ROOT / "preprocessed_eisnet_aet"
        self.PREPROCESSED_TRAIN = self.PREPROCESSED_ROOT / "train"
        self.PREPROCESSED_TEST = self.PREPROCESSED_ROOT / "test"

        self.PRETRAINED_RGB_PATH = Path("../pretrained/mit_b2.pth")
        self.PRETRAINED_EVT_PATH = Path("../pretrained/mit_b0.pth")

        self.RUNS_DIR = Path("./runs") / (self.PROJECT_NAME + self.VERSION_SUFFIX)
        self.CKPT_DIR = Path("./checkpoints") / self.VERSION_SUFFIX
        self.LOG_DIR = Path("./logs") / self.VERSION_SUFFIX
        self.TXT_LOG_DIR = Path("./txt_logs") / self.VERSION_SUFFIX

        for p in [self.RUNS_DIR, self.CKPT_DIR, self.LOG_DIR, self.TXT_LOG_DIR]:
            p.mkdir(parents=True, exist_ok=True)

        self.LATEST_CKPT_PATH = self.CKPT_DIR / "latest_checkpoint.pth"
        self.BEST_MIOU_CKPT_PATH = self.CKPT_DIR / "best_miou_checkpoint.pth"

        # ================= 模型核心参数 =================
        self.RGB_BACKBONE = 'b2'
        self.EVENT_BACKBONE = 'b0'
        self.NUM_BINS = 5
        self.EVENT_INPUT_CHANS = self.NUM_BINS * 2
        self.NUM_CLASSES = self.DSEC_CLASS_COUNT
        self.BACKBONE_CHANNELS = [64, 128, 320, 512]

        # [原生分辨率确认]
        self.IMG_SIZE = (440, 640)

        # ================= 基础训练参数 =================
        self.EPOCHS = 100
        self.ACCUMULATION_STEPS = 4
        self.VAL_INTERVAL = 1
        self.DEVICE = "cuda:0"
        self.WEIGHT_DECAY = 0.01
        self.LOSS_WEIGHTS = {'lovasz': 0.75, 'ce': 1.0, 'boundary': 0.5, 'detail': 1.0}

        # 自动切换耐心值
        self.AUTO_FINETUNE_PATIENCE = 6

        # ================= [核心] 双阶段配置 =================

        # --- Stage 1: 基础筑基 (Generalization) ---
        # 目标：快速收敛，学习 Road, Building 等大类
        self.STAGE1_CONFIG = {
            'lr': 2e-4,
            'copy_paste': True,  # 开启增强
            'ohem_keep_ratio': 0.20,  # 宽松 OHEM (20%)
            'class_weights': None,  # 不加权
            'batch_size': 8
        }

        # --- Stage 2: 攻坚克难 (Hard Mining) ---
        # 目标：在不破坏大类的前提下，提升 Fence, Pole, Wall
        # [关键调整]：针对 440x640 分辨率，降低烈度，保留 CopyPaste
        self.STAGE2_CONFIG = {
            'lr': 5e-6,  # 低学习率微调

            # [修正] 必须开启 CopyPaste！否则 Fence/Pole 样本太少，只会死记硬背
            'copy_paste': True,

            # [修正] 放宽到 10%，0.05 对于低分辨率来说信息量太少了
            'ohem_keep_ratio': 0.10,

            # [修正] 权重温和化。Wall=8.0 太激进，改为 4.0
            # Class Order: 0:Back, 1:Build, 2:Fence, 3:Person, 4:Pole, 5:Road, 6:Walk, 7:Veg, 8:Car, 9:Wall, 10:Sign
            'class_weights': [1.0, 1.0, 4.0, 3.0, 4.0, 1.0, 1.0, 1.0, 2.0, 4.0, 4.0],

            'batch_size': 8
        }

        # 当前状态标记 (自动管理)
        self.CURRENT_STAGE = 1
        self.IS_FINETUNE_STAGE = False

        # 动态属性 (默认 Stage 1)
        self.LEARNING_RATE = self.STAGE1_CONFIG['lr']
        self.ENABLE_COPY_PASTE = self.STAGE1_CONFIG['copy_paste']
        self.TRAIN_CROP_SIZE = None
        self.BATCH_SIZE = self.STAGE1_CONFIG['batch_size']
        self.MANUAL_CLASS_WEIGHTS = self.STAGE1_CONFIG['class_weights']
        self.OHEM_KEEP_RATIO = self.STAGE1_CONFIG['ohem_keep_ratio']

        self._SUFFIX = self.VERSION_SUFFIX


config = Config()