"""
可视化工具模块
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import os


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