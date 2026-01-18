"""
分割损失函数模块
"""

import torch
import torch.nn.functional as F


class SegmentationLoss:
    """分割损失函数"""

    def __init__(self, bce_weight=0.3, dice_smooth=1e-5):
        self.bce_weight = bce_weight
        self.dice_smooth = dice_smooth

    def compute_total_loss(self, pred, target):
        """
        组合损失：Dice损失 + BCE损失
        """
        dice_loss = self.dice_loss_with_logits(pred, target)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target)
        total_loss = dice_loss + self.bce_weight * bce_loss
        return total_loss

    def dice_loss_with_logits(self, pred, target):
        """计算Dice Loss（使用logits输入）"""
        pred_sigmoid = torch.sigmoid(pred)

        # 计算每个类别的Dice，然后取平均
        dice_per_class = []
        num_classes = pred.shape[1]

        for c in range(num_classes):
            pred_c = pred_sigmoid[:, c]
            target_c = target[:, c]

            intersection = (pred_c * target_c).sum(dim=(1, 2, 3))
            union = pred_c.sum(dim=(1, 2, 3)) + target_c.sum(dim=(1, 2, 3))

            dice_c = (2. * intersection + self.dice_smooth) / (union + self.dice_smooth)
            dice_per_class.append(dice_c)

        # 每个类别的平均Dice
        dice_per_class = torch.stack(dice_per_class, dim=0)  # [C, B]

        # 对batch取平均
        dice_per_class = dice_per_class.mean(dim=1)  # [C]

        # 返回平均Dice损失
        return 1 - dice_per_class.mean()

    def __call__(self, pred, target):
        return self.compute_total_loss(pred, target)


class BraTSDiceLoss:
    """BraTS专用Dice损失"""

    def __init__(self, smooth=1e-5):
        self.smooth = smooth

    def compute_per_class(self, pred, target):
        """计算每个类别的Dice分数"""
        pred_sigmoid = torch.sigmoid(pred)

        dice_scores = []
        for c in range(pred.shape[1]):
            pred_c = pred_sigmoid[:, c]
            target_c = target[:, c]

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        return torch.stack(dice_scores)