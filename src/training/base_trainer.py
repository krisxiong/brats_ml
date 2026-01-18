"""
基础训练器类
"""

import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class BaseTrainer(ABC):
    """基础训练器抽象类"""

    def __init__(self, config: Dict[str, Any], device: str = None):
        """
        初始化训练器

        Args:
            config: 训练配置字典
            device: 训练设备 (None则自动选择)
        """
        self.config = config
        self.device = self._setup_device(device)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = 0
        self.best_metric = 0.0
        self.history = {}

    def _setup_device(self, device: Optional[str]) -> torch.device:
        """设置训练设备"""
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                print("使用CPU")
        return torch.device(device)

    @abstractmethod
    def setup_model(self):
        """设置模型"""
        pass

    @abstractmethod
    def setup_data(self):
        """设置数据加载器"""
        pass

    @abstractmethod
    def setup_optimizer(self):
        """设置优化器"""
        pass

    @abstractmethod
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        pass

    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """验证"""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str, is_best: bool = False):
        """保存检查点"""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str):
        """加载检查点"""
        pass

    def train(self) -> Dict[str, Any]:
        """主训练循环"""
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()

        # 初始化历史记录
        self.history = {
            'train_loss': [],
            'train_metrics': [],
            'train_dice': [],
            'val_metrics': [],
            'learning_rates': []
        }

        # 训练循环
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch

            # 训练一个epoch
            train_results = self.train_epoch(epoch)
            self.history['train_loss'].append(train_results.get('loss', 0.0))
            self.history['train_metrics'].append(train_results)

            # 验证
            if self.config['validation']['enabled'] and \
                    epoch % self.config['validation']['interval'] == 0:
                val_results = self.validate()
                self.history['val_metrics'].append(val_results)

            # 保存检查点
            if epoch % self.config['checkpoint']['save_interval'] == 0:
                self.save_checkpoint(f"checkpoint_epoch{epoch}.pth")

        return self.history