"""
MAML训练器
"""

import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_trainer import BaseTrainer
from ..utils.early_stopping import ImprovedEarlyStopping
from ..utils.visualization import plot_training_curves
from ..utils.memory_estimator import estimate_memory_requirements
from ..meta.maml import FirstOrderMAML
from ..data.datasets.task_sampler import MetaTaskSampler
from ..model.Resnet import ResUNet

class MAMLTrainer(BaseTrainer):
    """MAML训练器"""

    def __init__(self, config: Dict[str, Any], device: str = None):
        super().__init__(config, device)

        self.maml: Optional[FirstOrderMAML] = None
        self.train_sampler: Optional[MetaTaskSampler] = None
        self.val_sampler: Optional[MetaTaskSampler] = None
        self.early_stopping: Optional[ImprovedEarlyStopping] = None

        # 扩展历史记录
        self.history.update({
            'train_dice_wt': [],
            'train_dice_tc': [],
            'train_dice_et': [],
            'support_dice': [],
            'val_dice_wt': [],
            'val_dice_tc': [],
            'val_dice_et': []
        })

    def setup_model(self):
        """设置MAML模型"""
        print("\n" + "=" * 70)
        print("设置模型和MAML框架")
        print("=" * 70)

        # 导入模型类

        # 创建模型
        model = ResUNet(
            in_channels=self.config['model']['in_channels'],
            out_channels=self.config['model']['out_channels'],
            base_channels=self.config['model']['base_channels']
        )

        # 打印模型信息
        model_info = model.get_model_size()
        print(f"模型架构: {self.config['model']['name']}")
        print(f"参数量: {model_info['total_params']:,}")
        print(f"模型大小: {model_info['size_mb']:.2f} MB")

        # 创建MAML
        self.maml = FirstOrderMAML(
            model=model,
            inner_lr=self.config['maml']['inner_lr'],
            outer_lr=self.config['maml']['outer_lr'],
            inner_steps=self.config['maml']['inner_steps'],
            device=self.device,
            use_amp=self.config['training']['use_amp'],
            l2_reg=self.config['maml']['l2_reg']
        )

        # 显存估算
        if torch.cuda.is_available():
            memory_req = estimate_memory_requirements(
                self.config,
                model_params=model_info['total_params']
            )
            print(f"估算显存需求: {memory_req:.2f} GB")

    def setup_data(self):
        """设置数据加载器"""
        print("\n" + "=" * 70)
        print("设置数据加载器")
        print("=" * 70)

        # 创建训练采样器
        self.train_sampler = MetaTaskSampler(
            data_root=self.config['data']['data_root'],
            k_shot=self.config['meta_learning']['k_shot'],
            k_query=self.config['meta_learning']['k_query'],
            crop_size=tuple(self.config['data']['crop_size']),
            crop_strategy=self.config['data']['crop_strategy'],
            task_names=self.config['data'].get('tasks', None)
        )

        print(f"训练任务: {len(self.train_sampler.datasets)}个")

        # 创建验证采样器（如果启用）
        if self.config['validation']['enabled']:
            self.val_sampler = MetaTaskSampler(
                data_root=self.config['data']['data_root'],
                k_shot=self.config['meta_learning']['k_shot'],
                k_query=self.config['meta_learning']['k_query'],
                crop_size=tuple(self.config['data']['crop_size']),
                crop_strategy=self.config['data']['crop_strategy'],
                task_names=self.config['validation'].get('val_tasks', None)
            )
            print(f"验证任务: {len(self.val_sampler.datasets)}个")

        # 打印数据统计
        print(f"K-shot: {self.config['meta_learning']['k_shot']}")
        print(f"K-query: {self.config['meta_learning']['k_query']}")
        print(f"Meta batch size: {self.config['meta_learning']['meta_batch_size']}")

    def setup_optimizer(self):
        """设置优化器（MAML已经包含）"""
        pass

    def setup_early_stopping(self):
        """设置早停机制"""
        if self.config['training']['early_stopping']['enabled']:
            self.early_stopping = ImprovedEarlyStopping(
                patience=self.config['training']['early_stopping']['patience'],
                min_delta=self.config['training']['early_stopping']['min_delta'],
                verbose=True
            )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}")
        print('=' * 70)

        epoch_losses = []
        epoch_metrics = {
            'dice_wt': [],
            'dice_tc': [],
            'dice_et': [],
            'dice_mean': [],
            'support_dice': []
        }

        # 进度条
        pbar = tqdm(
            range(self.config['training']['iterations_per_epoch']),
            desc=f'Training Epoch {epoch + 1}'
        )

        for iteration in pbar:
            try:
                # 采样任务批次
                task_batch = self.train_sampler.create_batch(
                    self.config['meta_learning']['meta_batch_size']
                )

                # MAML元训练步骤
                meta_loss, metrics = self.maml.meta_train_step(task_batch)

                # 记录指标
                epoch_losses.append(meta_loss)
                epoch_metrics['dice_wt'].append(metrics['dice_wt'])
                epoch_metrics['dice_tc'].append(metrics['dice_tc'])
                epoch_metrics['dice_et'].append(metrics['dice_et'])
                epoch_metrics['dice_mean'].append(metrics['dice_mean'])
                epoch_metrics['support_dice'].append(metrics['support_dice'])

                # 更新进度条
                mem_info = ""
                if torch.cuda.is_available():
                    mem_gb = torch.cuda.memory_allocated() / 1e9
                    mem_info = f", GPU: {mem_gb:.1f}GB"

                pbar.set_postfix_str(
                    f"Loss: {meta_loss:.4f}, "
                    f"Dice: {metrics['dice_mean']:.4f}"
                    f"{mem_info}"
                )

                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️  显存不足，跳过迭代 {iteration}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        # 汇总epoch指标
        summary = self._summarize_epoch(epoch_losses, epoch_metrics, epoch)
        return summary

    def _summarize_epoch(self, losses: List[float], metrics: Dict[str, List[float]],
                         epoch: int) -> Dict[str, float]:
        """汇总epoch指标"""
        summary = {
            'loss': np.mean(losses),
            'dice_wt': np.mean(metrics['dice_wt']),
            'dice_tc': np.mean(metrics['dice_tc']),
            'dice_et': np.mean(metrics['dice_et']),
            'dice_mean': np.mean(metrics['dice_mean']),
            'support_dice': np.mean(metrics['support_dice'])
        }

        # 打印epoch结果
        print(f"\nEpoch {epoch + 1} 训练结果:")
        print(f"  损失: {summary['loss']:.4f}")
        print(f"  Query Dice: {summary['dice_mean']:.4f}")
        print(f"    - WT: {summary['dice_wt']:.4f}")
        print(f"    - TC: {summary['dice_tc']:.4f}")
        print(f"    - ET: {summary['dice_et']:.4f}")
        print(f"  Support Dice: {summary['support_dice']:.4f}")

        # 检查过拟合
        gap = summary['support_dice'] - summary['dice_mean']
        if gap > 0.15:
            print(f"  ⚠️  Support-Query差距较大 ({gap:.3f})，可能过拟合")

        # 显存信息
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  峰值显存: {max_mem:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        return summary

    def validate(self) -> Dict[str, float]:
        """验证"""
        print("\n验证...")

        val_metrics = self.maml.validate(
            self.val_sampler,
            num_tasks=self.config['validation']['num_val_tasks']
        )

        print(f"验证结果:")
        print(f"  Dice: {val_metrics['dice_mean']:.4f}")
        print(f"    - WT: {val_metrics['dice_wt']:.4f}")
        print(f"    - TC: {val_metrics['dice_tc']:.4f}")
        print(f"    - ET: {val_metrics['dice_et']:.4f}")

        # 学习率调度
        self.maml.scheduler.step(val_metrics['dice_mean'])

        return val_metrics

    def train(self) -> Dict[str, Any]:
        """完整的训练循环"""
        print("\n" + "=" * 70)
        print("开始MAML训练")
        print("=" * 70)

        # 设置组件

        self.setup_model()
        self.setup_data()
        self.setup_early_stopping()

        # 训练开始时间
        start_time = datetime.now()

        # 训练循环
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch

            # 训练一个epoch
            train_results = self.train_epoch(epoch)

            # 记录历史
            self.history['train_loss'].append(train_results['loss'])
            self.history['train_dice_wt'].append(train_results['dice_wt'])
            self.history['train_dice_tc'].append(train_results['dice_tc'])
            self.history['train_dice_et'].append(train_results['dice_et'])
            self.history['support_dice'].append(train_results['support_dice'])

            # 验证
            if self.config['validation']['enabled'] and \
                    epoch % self.config['validation']['interval'] == 0:
                val_results = self.validate()
                self.history['val_metrics'].append(val_results)

                # 保存最佳模型
                current_dice = val_results['dice_mean']
                if current_dice > self.best_metric:
                    self.best_metric = current_dice
                    save_path = Path(self.config['checkpoint']['save_dir']) / 'best_model.pth'
                    self.save_checkpoint(str(save_path), is_best=True)
                    print(f"✓ 保存最佳模型 (Dice: {self.best_metric:.4f})")

            # 定期保存检查点
            if epoch % self.config['checkpoint']['save_interval'] == 0:
                save_path = Path(self.config['checkpoint']['save_dir']) / f'checkpoint_epoch{epoch}.pth'
                self.save_checkpoint(str(save_path))

            # 绘制训练曲线
            if epoch % self.config['logging']['log_interval'] == 0:
                log_dir = Path(self.config['logging']['log_dir'])
                plot_training_curves(self.history, str(log_dir))

            # 早停检查
            if self.early_stopping is not None:
                self.early_stopping(epoch, train_results['dice_mean'])
                if self.early_stopping.early_stop:
                    print("\n触发早停，停止训练")
                    break

        # 训练结束
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 70)
        print("训练完成!")
        print("=" * 70)
        print(f"最佳Dice: {self.best_metric:.4f}")
        print(f"训练时长: {duration}")
        print(f"总epoch数: {self.current_epoch + 1}")
        print(f"模型保存目录: {self.config['checkpoint']['save_dir']}")
        print("=" * 70)

        # 保存最终模型
        final_path = Path(self.config['checkpoint']['save_dir']) / 'final_model.pth'
        self.save_checkpoint(str(final_path))

        return self.history

    def save_checkpoint(self, path: str, is_best: bool = False):
        """保存检查点"""
        if self.maml is not None:
            self.maml.save_checkpoint(path, self.current_epoch, {
                'best_metric': self.best_metric,
                'history': self.history
            })

    def load_checkpoint(self, path: str):
        """加载检查点"""
        if self.maml is not None:
            checkpoint = self.maml.load_checkpoint(path)
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_metric = checkpoint.get('metrics', {}).get('dice_mean', 0.0)
            self.history = checkpoint.get('metrics', {}).get('history', {})
            return checkpoint
        return None