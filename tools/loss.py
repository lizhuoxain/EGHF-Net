import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import config
from lovasz_losses import LovaszSoftmax


class OhemCELoss(nn.Module):
    def __init__(self, thresh=0.7, min_kept=20000, ignore_index=255, weights=None, keep_ratio=0.2):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).to(config.DEVICE)
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.keep_ratio = keep_ratio  # 动态比例

        self.update_weights(weights)

    def update_weights(self, weights):
        """动态更新类别权重"""
        if weights is not None:
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights)
            weights = weights.float().to(config.DEVICE)

        # 重新初始化内部 CE Loss
        self.ce = nn.CrossEntropyLoss(
            weight=weights,
            ignore_index=self.ignore_index,
            reduction='none'
        )

    def forward(self, logits, target):
        pixel_losses = self.ce(logits, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_index
        valid_losses = pixel_losses[mask]

        if len(valid_losses) == 0:
            return torch.tensor(0.0).to(logits.device)

        num_valid = valid_losses.numel()

        # [动态使用 keep_ratio]
        # Stage 1: 0.2 (20%) -> 包含一部分简单样本
        # Stage 2: 0.05 (5%) -> 仅关注极难样本
        keep_num = max(int(num_valid * self.keep_ratio), self.min_kept)
        keep_num = min(keep_num, num_valid)

        topk_loss, _ = valid_losses.topk(keep_num)
        return topk_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=None, weights=None):
        super().__init__()
        weights_config = config.LOSS_WEIGHTS
        self.w_lov = weights_config['lovasz']
        self.w_ce = weights_config['ce']
        self.w_bnd = weights_config['boundary']

        # 初始化时使用 Config 中的当前比例
        self.ohem = OhemCELoss(
            thresh=0.7,
            min_kept=25000,
            ignore_index=255,
            weights=weights,
            keep_ratio=config.OHEM_KEEP_RATIO
        )
        self.lovasz = LovaszSoftmax(classes='present', per_image=True, ignore=255)

    def update_stage_params(self, new_weights, new_ratio):
        """外部调用此方法切换阶段参数"""
        print(f"⚖️ [Loss] Updating parameters: Ratio={new_ratio}, Weights={new_weights}")
        self.ohem.keep_ratio = new_ratio
        self.ohem.update_weights(new_weights)

    def forward(self, logits, targets, detail_logits=None, detail_targets=None):
        ce_loss = self.ohem(logits, targets)
        probas = F.softmax(logits, dim=1)
        lov_loss = self.lovasz(probas, targets)

        bnd_loss = torch.tensor(0.0, device=logits.device)
        if detail_logits is not None and self.w_bnd > 0:
            if detail_targets is None:
                detail_targets = self._get_boundary(targets)

            bnd_loss = F.binary_cross_entropy_with_logits(
                detail_logits, detail_targets.float(),
                pos_weight=torch.tensor([5.0], device=logits.device)
            )

        total = self.w_lov * lov_loss + self.w_ce * ce_loss + self.w_bnd * bnd_loss
        return total, {
            "total": total.item(),
            "lovasz": lov_loss.item(),
            "ce(ohem)": ce_loss.item(),
            "bnd": bnd_loss.item()
        }

    def _get_boundary(self, mask):
        mask = mask.unsqueeze(1).float()
        dilated = F.max_pool2d(mask, 3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask, 3, stride=1, padding=1)
        return (dilated != eroded)