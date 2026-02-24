import torch
import numpy as np
import cv2
from configs.config import config


class RGBPreprocessor:
    """
    RGB 预处理：仅做简单的归一化和 Resize (如果需要)。
    不再强求 ACE 增强，交给网络去适应。
    """

    def __init__(self):
        pass

    def __call__(self, rgb_img):
        # 输入: Tensor [3, H, W], 范围 0-1
        if not isinstance(rgb_img, torch.Tensor):
            # 如果输入是 numpy (H, W, 3)
            rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0

        # 如果需要 ACE，可以在这里加回来，但 DCN 对纹理更敏感，保持原始纹理可能更好
        # 这里仅确保数值范围安全
        return torch.clamp(rgb_img, 0.0, 1.0)


class EventPreprocessor:
    """
    Event 预处理：
    读取 Voxel Grid，归一化。
    Event 数据是主路，必须保持其原始几何结构（不进行 Warp）。
    """

    def __init__(self):
        pass

    def __call__(self, event_input):
        """
        Args:
            event_input: 可以是路径(str) 或者 numpy array (H, W) 或 (H, W, C)
        Returns:
            Tensor [1, H, W] (如果单通道) 或 [C, H, W]
        """
        # 1. 读取/转换
        if isinstance(event_input, str) or isinstance(event_input, type(None)):
            # 路径模式 (例如读取 Voxel Grid PNG)
            if event_input is None:
                return torch.zeros((1, config.IMG_SIZE[0], config.IMG_SIZE[1]), dtype=torch.float32)

            # 读取 PNG (0-255)
            evt = cv2.imread(event_input, cv2.IMREAD_GRAYSCALE)
            if evt is None:
                return torch.zeros((1, config.IMG_SIZE[0], config.IMG_SIZE[1]), dtype=torch.float32)

            # 归一化到 0-1
            evt = evt.astype(np.float32) / 255.0
            tensor = torch.from_numpy(evt).unsqueeze(0)  # [1, H, W]

        elif isinstance(event_input, np.ndarray):
            # Numpy 模式
            evt = event_input.astype(np.float32)
            if evt.max() > 1.0:
                evt /= 255.0

            if len(evt.shape) == 2:
                tensor = torch.from_numpy(evt).unsqueeze(0)
            else:
                tensor = torch.from_numpy(evt).permute(2, 0, 1)  # [H,W,C] -> [C,H,W]

        elif isinstance(event_input, torch.Tensor):
            tensor = event_input
        else:
            raise TypeError(f"Unknown event input type: {type(event_input)}")

        return tensor