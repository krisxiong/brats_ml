"""
可视化工具模块
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import os


def plot_training_curves(history: Dict[str, Any], save_dir: str, filename: str = 'training_curves.png'):
    """
    绘制训练曲线

    Args:
        history: 训练历史字典
        save_dir: 保存目录
        filename: 保存文件名
    """
    if len(history.get('train_loss', [])) == 0:
        print("没有训练数据可绘制")
        return

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss曲线
    _plot_loss_curve(axes[0, 0], epochs, history)

    # Mean Dice曲线
    _plot_dice_curve(axes[0, 1], epochs, history)

    # 各类别Dice曲线
    _plot_per_class_dice(axes[1, 0], epochs, history)

    # Support vs Query Dice曲线
    _plot_support_vs_query(axes[1, 1], epochs, history)

    plt.tight_layout()

    # 保存图像
    save_path = Path(save_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"训练曲线已保存: {save_path}")


def _plot_loss_curve(ax, epochs, history):
    """绘制损失曲线"""
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train')

    if 'val_loss' in history and len(history['val_loss']) == len(epochs):
        ax.plot(epochs, history['val_loss'], 'r--', linewidth=2, label='Val')

    ax.set_title('Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_dice_curve(ax, epochs, history):
    """绘制Dice曲线"""
    # 训练Dice
    if 'train_dice_wt' in history and len(history['train_dice_wt']) == len(epochs):
        train_dice_mean = np.mean([
            history['train_dice_wt'],
            history['train_dice_tc'],
            history['train_dice_et']
        ], axis=0)
        ax.plot(epochs, train_dice_mean, 'b-', linewidth=2, label='Train')

    # 验证Dice
    if 'val_metrics' in history and history['val_metrics']:
        val_dice_mean = [m.get('dice_mean', 0) for m in history['val_metrics']]
        if len(val_dice_mean) == len(epochs):
            ax.plot(epochs, val_dice_mean, 'r--', linewidth=2, label='Val')

    ax.set_title('Mean Dice', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_per_class_dice(ax, epochs, history):
    """绘制各类别Dice曲线"""
    colors = ['red', 'green', 'blue']
    labels = ['WT', 'TC', 'ET']

    for i, (label, color) in enumerate(zip(labels, colors)):
        key = f'train_dice_{label.lower()}'
        if key in history and len(history[key]) == len(epochs):
            ax.plot(epochs, history[key], color=color, linewidth=2,
                    label=label, marker='o', markersize=3)

    ax.set_title('Per-Class Dice', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_support_vs_query(ax, epochs, history):
    """绘制Support vs Query Dice对比"""
    if 'support_dice' in history and len(history['support_dice']) == len(epochs):
        ax.plot(epochs, history['support_dice'], 'g-', linewidth=2,
                label='Support', marker='s', markersize=3)

        # 训练Query Dice
        if 'train_dice_wt' in history and len(history['train_dice_wt']) == len(epochs):
            train_dice_mean = np.mean([
                history['train_dice_wt'],
                history['train_dice_tc'],
                history['train_dice_et']
            ], axis=0)
            ax.plot(epochs, train_dice_mean, 'b-', linewidth=2,
                    label='Query', marker='o', markersize=3)

            # 检查过拟合
            gap = np.array(history['support_dice']) - np.array(train_dice_mean)
            if len(gap) > 0 and gap[-1] > 0.15:
                ax.text(0.5, 0.95, '⚠️ 可能过拟合',
                        transform=ax.transAxes,
                        ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        ax.set_title('Support vs Query Dice', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice')
        ax.legend()
        ax.grid(True, alpha=0.3)


def plot_learning_rate(optimizer_history: List[float], save_path: str):
    """绘制学习率变化曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(optimizer_history)), optimizer_history, 'b-', linewidth=2)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()