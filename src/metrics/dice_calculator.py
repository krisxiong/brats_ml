"""
Dice指标计算模块
"""

import torch
import numpy as np
from typing import List, Dict, Any


class BraTSDiceCalculator:
    """
    BraTS官方标准Dice计算
    关键区别：
    1. 对每个病例单独计算，然后平均（不是整个batch合并）
    2. 空标签/空预测的严格处理规则
    3. 返回每个类别的Dice分数
    """

    def __init__(self, threshold=0.5, eps=1e-10):
        self.threshold = threshold
        self.eps = eps

    def compute_dice_per_class(self, pred, target):
        """计算每个类别的Dice分数"""
        pred_sigmoid = torch.sigmoid(pred)
        pred_binary = (pred_sigmoid > self.threshold).float()

        batch_size = pred.shape[0]
        num_classes = pred.shape[1]

        # 存储每个病例每个类别的Dice
        dice_per_case = []

        for i in range(batch_size):
            case_dices = []
            for c in range(num_classes):
                pred_c = pred_binary[i, c]  # 单个病例的单个类别
                target_c = target[i, c]

                # ============ BraTS官方规则 ============
                # 规则1：计算交集和并集
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()

                # 规则2：处理空标签
                if target_c.sum() == 0:
                    # 真实标签为空
                    if pred_c.sum() == 0:
                        dice = 1.0  # 预测也为空 -> 完美
                    else:
                        dice = 0.0  # 预测不为空 -> 全错
                elif union == 0:
                    # 这种情况不应该发生（target_c.sum()>0但union=0意味着pred_c.sum()<0）
                    dice = 0.0
                else:
                    dice = (2. * intersection) / (union + self.eps)

                case_dices.append(dice)

            dice_per_case.append(case_dices)

        # 转换为tensor: [batch_size, num_classes]
        dice_per_case = torch.tensor(dice_per_case, device=pred.device)

        # 对病例取平均，得到每个类别的平均Dice
        dice_per_class = dice_per_case.mean(dim=0)  # [num_classes]

        return dice_per_class.tolist()

    def compute_detailed_metrics(self, pred, target):
        """计算详细指标"""
        dice_per_class = self.compute_dice_per_class(pred, target)

        metrics = {
            'dice_wt': dice_per_class[0] if len(dice_per_class) > 0 else 0.0,
            'dice_tc': dice_per_class[1] if len(dice_per_class) > 1 else 0.0,
            'dice_et': dice_per_class[2] if len(dice_per_class) > 2 else 0.0,
            'dice_mean': np.mean(dice_per_class),
            'wt': dice_per_class[0] if len(dice_per_class) > 0 else 0.0,
            'tc': dice_per_class[1] if len(dice_per_class) > 1 else 0.0,
            'et': dice_per_class[2] if len(dice_per_class) > 2 else 0.0,
            'mean': np.mean(dice_per_class)
        }

        return metrics