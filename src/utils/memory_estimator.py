"""
显存需求估算模块
"""

import numpy as np
import torch
from typing import Dict, Any


def estimate_memory_requirements(config: Dict[str, Any], model_params: int = None) -> float:
    """
    估算显存需求

    Args:
        config: 训练配置
        model_params: 模型参数量

    Returns:
        估算的显存需求（GB）
    """
    # 训练参数
    crop_size = config['data']['crop_size']
    k_shot = config['meta_learning']['k_shot']
    k_query = config['meta_learning']['k_query']
    meta_batch_size = config['meta_learning']['meta_batch_size']

    # 单样本大小 (MB)
    # 4个输入模态 + 3个输出通道
    voxels = np.prod(crop_size)
    bytes_per_voxel = 4  # float32
    sample_mb = voxels * (4 + 3) * bytes_per_voxel / (1024 ** 2)

    # 任务大小
    task_mb = sample_mb * (k_shot + k_query) * meta_batch_size

    # 模型大小估算
    if model_params is None:
        # 粗略估算：每个参数4字节
        base_ch = config['model']['base_channels']
        model_mb = (base_ch ** 2) * 100 * 4 / (1024 ** 2)
    else:
        model_mb = model_params * 4 / (1024 ** 2)

    # 总显存（考虑梯度、激活等，乘系数2.5）
    total_mb = (task_mb + model_mb) * 2.5

    # 转换为GB
    total_gb = total_mb / 1024

    print("\n显存需求估算:")
    print(f"  单样本: {sample_mb:.1f} MB")
    print(f"  单任务: {task_mb:.1f} MB")
    print(f"  模型: {model_mb:.1f} MB")
    print(f"  总需求: {total_gb:.2f} GB")

    if total_gb > 20:
        print(f"\n⚠️  警告: 显存需求较高 ({total_gb:.1f}GB)")
        print("   建议调整以下参数:")
        print("   1. 减小 crop_size")
        print("   2. 减小 model.base_channels")
        print("   3. 减小 meta_batch_size")

    # 检查实际可用显存
    if torch.cuda.is_available():
        available_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if total_gb > available_gb * 0.8:
            print(f"\n⚠️  警告: 估算需求 ({total_gb:.1f}GB) 超过可用显存的80%")
            print(f"  可用显存: {available_gb:.1f} GB")

    return total_gb