"""
配置文件加载模块 - 支持您原有的配置结构
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch

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