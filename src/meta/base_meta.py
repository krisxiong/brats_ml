"""
元学习器基类
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseMetaLearner(ABC):
    """元学习器基类"""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device

    @abstractmethod
    def train_step(self, task_batch):
        """训练步骤"""
        pass

    @abstractmethod
    def validate(self, val_sampler, num_tasks=10):
        """验证"""
        pass

    @abstractmethod
    def save_checkpoint(self, path, epoch, metrics):
        """保存检查点"""
        pass

    @abstractmethod
    def load_checkpoint(self, path):
        """加载检查点"""
        pass

    def _forward_with_params(self, x, params):
        """使用给定参数前向传播"""
        # 临时替换参数
        original_params = {}
        for name, param in self.model.named_parameters():
            if name in params:
                original_params[name] = param.data
                param.data = params[name].data if hasattr(params[name], 'data') else params[name]

        try:
            output = self.model(x)
        finally:
            # 恢复原始参数
            for name, param in self.model.named_parameters():
                if name in original_params:
                    param.data = original_params[name]

        return output