"""
train.py - MAML训练脚本
支持配置文件、验证、早停、checkpoint保存
"""
import torch
import argparse
import yaml
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from model import ResUNet
from maml import FirstOrderMAML
from dataloader import MetaTaskSampler


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def setup_directories(config):
    """创建必要的目录"""
    dirs = [
        config['checkpoint']['save_dir'],
        config['logging']['log_dir'],
        config['testing']['output_dir']
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def print_config(config):
    """打印配置信息"""
    print("\n" + "=" * 70)
    print("训练配置")
    print("=" * 70)
    print(f"数据:")
    print(f"  根目录: {config['data']['data_root']}")
    print(f"  任务数: {len(config['data']['tasks'])}")
    print(f"  Crop大小: {config['data']['crop_size']}")
    print(f"  Crop策略: {config['data']['crop_strategy']}")

    print(f"\n模型:")
    print(f"  架构: {config['model']['name']}")
    print(f"  基础通道: {config['model']['base_channels']}")

    print(f"\nMAML:")
    print(f"  算法: {config['maml']['algorithm']}")
    print(f"  Inner LR: {config['maml']['inner_lr']}")
    print(f"  Outer LR: {config['maml']['outer_lr']}")
    print(f"  Inner Steps: {config['maml']['inner_steps']}")

    print(f"\n元学习:")
    print(f"  K-shot: {config['meta_learning']['k_shot']}")
    print(f"  K-query: {config['meta_learning']['k_query']}")
    print(f"  Meta Batch: {config['meta_learning']['meta_batch_size']}")

    print(f"\n训练:")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  迭代/轮: {config['training']['iterations_per_epoch']}")
    print(f"  混合精度: {config['training']['use_amp']}")

    print("=" * 70)


def estimate_memory(config):
    """估算显存需求"""
    crop_size = config['data']['crop_size']
    base_ch = config['model']['base_channels']
    k_shot = config['meta_learning']['k_shot']
    k_query = config['meta_learning']['k_query']

    # 单样本大小 (MB)
    sample_mb = np.prod(crop_size) * 7 * 4 / (1024**2)  # 4 input + 3 output

    # 任务大小
    task_mb = sample_mb * (k_shot + k_query)

    # 模型大小估算
    model_mb = (base_ch ** 2) * 100 / (1024**2)

    # 总显存（×2.5考虑梯度和激活）
    total_mb = (task_mb + model_mb) * 2.5

    print(f"\n显存需求估算:")
    print(f"  单样本: {sample_mb:.1f} MB")
    print(f"  单任务: {task_mb:.1f} MB")
    print(f"  模型: {model_mb:.1f} MB")
    print(f"  总需求: {total_mb/1024:.2f} GB")

    if total_mb > 20000:
        print(f"\n⚠️  警告: 显存需求较高 ({total_mb/1024:.1f}GB)")
        print(f"   建议: 减小crop_size或base_channels")

    return total_mb


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class ImprovedEarlyStopping:
    """改进的早停机制"""

    def __init__(self, patience=20, min_delta=0.0001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, epoch, score):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                print(f"  [早停] 初始最佳Dice: {score:.4f}")

        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  [早停] 无改进 {self.counter}/{self.patience} "
                      f"(当前: {score:.4f}, 最佳: {self.best_score:.4f})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n  [早停] 触发！")
                    print(f"    最佳Dice: {self.best_score:.4f} (Epoch {self.best_epoch})")
                    print(f"    当前Dice: {score:.4f} (Epoch {epoch})")

        else:
            improvement = score - self.best_score
            if self.verbose:
                print(f"  [早停] 改进! +{improvement:.4f} "
                      f"(从 {self.best_score:.4f} → {score:.4f})")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0  # 重置计数器

    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

def plot_training_curves(history, save_dir):
    """绘制训练曲线"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train')
    if 'val_loss' in history:
        axes[0, 0].plot(epochs, history['val_loss'], 'r--', linewidth=2, label='Val')
    axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Mean Dice
    axes[0, 1].plot(epochs, history['train_dice'], 'b-', linewidth=2, label='Train')
    if len(history['val_dice']) == len(epochs):
        axes[0, 1].plot(epochs, history['val_dice'], 'r--', linewidth=2, label='Val')

    axes[0, 1].set_title('Mean Dice', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Per-class Dice
    for i, (name, color) in enumerate(zip(['WT', 'TC', 'ET'], ['red', 'green', 'blue'])):
        key = f'train_dice_{name.lower()}'
        if key in history:
            axes[1, 0].plot(epochs, history[key], color=color, linewidth=2,
                          label=name, marker='o', markersize=3)
    axes[1, 0].set_title('Per-Class Dice', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Support vs Query Dice
    if 'support_dice' in history:
        axes[1, 1].plot(epochs, history['support_dice'], 'g-', linewidth=2,
                       label='Support', marker='s', markersize=3)
        axes[1, 1].plot(epochs, history['train_dice'], 'b-', linewidth=2,
                       label='Query', marker='o', markersize=3)
        axes[1, 1].set_title('Support vs Query Dice', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Dice')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 检查过拟合
        if len(history['support_dice']) > 0 and len(history['train_dice']) > 0:
            gap = np.array(history['support_dice']) - np.array(history['train_dice'])
            if gap[-1] > 0.15:
                axes[1, 1].text(0.5, 0.95, '⚠️ 可能过拟合',
                              transform=axes[1, 1].transAxes,
                              ha='center', va='top',
                              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()


def train(config):
    """主训练函数"""
    # 设置目录
    setup_directories(config)

    # 打印配置
    print_config(config)

    # 估算显存
    estimate_memory(config)

    # 设置设备
    device = torch.device(config['hardware']['device']
                         if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ========== 1. 创建数据加载器 ==========
    print("\n" + "=" * 70)
    print("创建数据加载器")
    print("=" * 70)

    try:
        train_sampler = MetaTaskSampler(
            data_root=config['data']['data_root'],
            k_shot=config['meta_learning']['k_shot'],
            k_query=config['meta_learning']['k_query'],
            crop_size=tuple(config['data']['crop_size']),
            crop_strategy=config['data']['crop_strategy']
        )

        # 验证集（如果启用）
        val_sampler = None
        if config['validation']['enabled']:
            # 使用相同的sampler，但在验证时会采样不同的任务
            val_sampler = train_sampler

    except Exception as e:
        print(f"\n❌ 数据加载失败: {e}")
        print("\n请检查:")
        print("1. 数据目录结构是否正确")
        print("2. 至少有2个任务（不同的医疗中心）")
        print("3. 每个任务至少有k_shot+k_query个样本")
        return None

    # ========== 2. 创建模型 ==========
    print("\n" + "=" * 70)
    print("创建模型")
    print("=" * 70)

    model = ResUNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels']
    )

    model_info = model.get_model_size()
    print(f"模型: {config['model']['name']}")
    print(f"参数量: {model_info['total_params']:,}")
    print(f"大小: {model_info['size_mb']:.2f} MB")

    # ========== 3. 创建MAML ==========
    print("\n" + "=" * 70)
    print("创建MAML框架")
    print("=" * 70)

    maml = FirstOrderMAML(
        model=model,
        inner_lr=config['maml']['inner_lr'],
        outer_lr=config['maml']['outer_lr'],
        inner_steps=config['maml']['inner_steps'],
        device=device,
        use_amp=config['training']['use_amp'],
        l2_reg=config['maml']['l2_reg']
    )

    # ========== 4. 训练循环 ==========
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)

    best_dice = 0.0
    history = {
        'train_loss': [],
        'train_dice': [],
        'train_dice_wt': [],
        'train_dice_tc': [],
        'train_dice_et': [],
        'support_dice': []
    }

    if config['validation']['enabled']:
        history['val_dice'] = []

    # 早停
    # 创建早停
    if config['training']['early_stopping'].get('enabled', True):
        early_stopping = ImprovedEarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta'],
            verbose=True
        )

    # 训练开始时间
    start_time = datetime.now()

    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config['training']['num_epochs']}")
        print('='*70)

        # 训练
        epoch_metrics = train_epoch(
            maml=maml,
            sampler=train_sampler,
            config=config,
            epoch=epoch
        )

        # 记录历史
        history['train_loss'].append(epoch_metrics['loss'])
        history['train_dice'].append(epoch_metrics['dice_mean'])
        history['train_dice_wt'].append(epoch_metrics['dice_wt'])
        history['train_dice_tc'].append(epoch_metrics['dice_tc'])
        history['train_dice_et'].append(epoch_metrics['dice_et'])
        history['support_dice'].append(epoch_metrics['support_dice'])

        # 验证
        if config['validation']['enabled'] and epoch % config['validation']['interval'] == 0:
            print(f"\n验证...")
            val_metrics = validate(maml, val_sampler, config)
            history['val_dice'].append(val_metrics['dice_mean'])

            print(f"验证Dice: {val_metrics['dice_mean']:.4f}")

            # 学习率调度
            maml.scheduler.step(val_metrics['dice_mean'])

        # 保存最佳模型
        current_dice = epoch_metrics['dice_mean']
        if current_dice > best_dice:
            best_dice = current_dice
            save_path = os.path.join(
                config['checkpoint']['save_dir'],
                'best_model.pth'
            )
            maml.save_checkpoint(save_path, epoch, epoch_metrics)
            print(f"✓ 保存最佳模型 (Dice: {best_dice:.4f})")

        # 定期保存
        if epoch % config['checkpoint']['save_interval'] == 0:
            save_path = os.path.join(
                config['checkpoint']['save_dir'],
                f'checkpoint_epoch{epoch}.pth'
            )
            maml.save_checkpoint(save_path, epoch, epoch_metrics)

        # 绘制曲线
        if epoch % config['logging']['log_interval'] == 0:
            plot_training_curves(history, config['logging']['log_dir'])

        # 早停检查
        if early_stopping is not None:
            early_stopping(epoch, current_dice)  # 传入epoch
            if early_stopping.early_stop:
                break

    # 训练结束
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"最佳Dice: {best_dice:.4f}")
    print(f"训练时长: {duration}")
    print(f"模型保存: {config['checkpoint']['save_dir']}/best_model.pth")
    print("=" * 70)

    return maml


def train_epoch(maml, sampler, config, epoch):
    """训练一个epoch"""
    epoch_losses = []
    epoch_metrics = {
        'dice_wt': [],
        'dice_tc': [],
        'dice_et': [],
        'dice_mean': [],
        'support_dice': []
    }

    pbar = tqdm(
        range(config['training']['iterations_per_epoch']),
        desc=f'Epoch {epoch}'
    )

    for iteration in pbar:
        try:
            # 采样任务批次
            task_batch = sampler.create_batch(
                config['meta_learning']['meta_batch_size']
            )

            # 元训练
            meta_loss, metrics = maml.meta_train_step(task_batch)

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
                mem_info = f", mem: {mem_gb:.1f}GB"

            pbar.set_postfix_str(
                f"loss: {meta_loss:.4f}, "
                f"dice: {metrics['dice_mean']:.4f} "
                f"(WT:{metrics['dice_wt']:.3f} "
                f"TC:{metrics['dice_tc']:.3f} "
                f"ET:{metrics['dice_et']:.3f})"
                f"{mem_info}"
            )

            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠️  OOM at iteration {iteration}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e

    # 汇总epoch指标
    summary = {
        'loss': np.mean(epoch_losses),
        'dice_wt': np.mean(epoch_metrics['dice_wt']),
        'dice_tc': np.mean(epoch_metrics['dice_tc']),
        'dice_et': np.mean(epoch_metrics['dice_et']),
        'dice_mean': np.mean(epoch_metrics['dice_mean']),
        'support_dice': np.mean(epoch_metrics['support_dice'])
    }

    # 打印epoch结果
    print(f"\nEpoch {epoch} 结果:")
    print(f"  Loss: {summary['loss']:.4f}")
    print(f"  Query Dice: {summary['dice_mean']:.4f}")
    print(f"    - WT: {summary['dice_wt']:.4f}")
    print(f"    - TC: {summary['dice_tc']:.4f}")
    print(f"    - ET: {summary['dice_et']:.4f}")
    print(f"  Support Dice: {summary['support_dice']:.4f}")

    # 检查过拟合
    gap = summary['support_dice'] - summary['dice_mean']
    if gap > 0.15:
        print(f"  ⚠️  Support-Query差距较大 ({gap:.3f})，可能过拟合")

    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  峰值显存: {max_mem:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    return summary


def validate(maml, val_sampler, config):
    """验证函数"""
    val_dice = maml.validate(
        val_sampler,
        num_tasks=config['validation']['num_val_tasks']
    )
    return val_dice


def main():
    parser = argparse.ArgumentParser(description='MAML训练脚本')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的checkpoint路径')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 开始训练
    try:
        maml = train(config)
        print("\n✅ 训练成功完成！")

    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")

    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()