"""
梯度相关工具函数
"""

import torch


class GradientUtils:
    """梯度工具类"""

    @staticmethod
    def clip_gradients(parameters, max_norm=1.0):
        """梯度裁剪"""
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)

    @staticmethod
    def zero_grad(optimizer):
        """清空梯度"""
        optimizer.zero_grad()

    @staticmethod
    def compute_gradient_norm(parameters):
        """计算梯度范数"""
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm