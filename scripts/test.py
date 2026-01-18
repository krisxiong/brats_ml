"""
BraTS多中心MAML元测试脚本（完整版）
支持：快速迁移、BraTS标准评估、多中心测试、小样本适应
所有参数通过配置文件导入，无需命令行参数
"""

import torch
import torch.nn.functional as F
import argparse
import yaml
import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import time
import copy
import warnings
import sys
warnings.filterwarnings('ignore')

from model import ResUNet
from maml import FirstOrderMAML
from dataloader import BraTSDataset


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config):
    """验证配置文件的必要参数"""
    required_sections = ['data', 'model', 'maml', 'hardware', 'testing']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"配置文件中缺少必要部分: {section}")

    # 检查必要参数
    if 'checkpoint' not in config['testing']:
        raise ValueError("配置文件中缺少测试checkpoint路径: testing.checkpoint")

    if 'test_tasks' not in config['testing'] or not config['testing']['test_tasks']:
        raise ValueError("配置文件中缺少测试任务列表: testing.test_tasks")

    return True


def enforce_brats_hierarchy(pred_probs, threshold=0.5):
    """
    强制执行BraTS层次关系：ET ⊆ TC ⊆ WT
    参数:
        pred_probs: [3, D, H, W] 或 [B, 3, D, H, W] 概率图
    返回: 修正后的二值化预测
    """
    # 二值化
    if isinstance(pred_probs, np.ndarray):
        pred_binary = (pred_probs > threshold).astype(np.float32)
    else:  # torch.Tensor
        pred_binary = (pred_probs > threshold).float()

    # 处理不同维度
    if pred_binary.ndim == 4:  # [3, D, H, W]
        wt, tc, et = pred_binary[0], pred_binary[1], pred_binary[2]
        tc_corrected = np.clip(tc + et, 0, 1) if isinstance(pred_binary, np.ndarray) else torch.clamp(tc + et, 0, 1)
        wt_corrected = np.clip(wt + tc_corrected, 0, 1) if isinstance(pred_binary, np.ndarray) else torch.clamp(wt + tc_corrected, 0, 1)
        result = np.stack([wt_corrected, tc_corrected, et], axis=0) if isinstance(pred_binary, np.ndarray) else torch.stack([wt_corrected, tc_corrected, et], dim=0)
    else:  # [B, 3, D, H, W]
        wt, tc, et = pred_binary[:, 0], pred_binary[:, 1], pred_binary[:, 2]
        tc_corrected = np.clip(tc + et, 0, 1) if isinstance(pred_binary, np.ndarray) else torch.clamp(tc + et, 0, 1)
        wt_corrected = np.clip(wt + tc_corrected, 0, 1) if isinstance(pred_binary, np.ndarray) else torch.clamp(wt + tc_corrected, 0, 1)
        result = np.stack([wt_corrected, tc_corrected, et], axis=1) if isinstance(pred_binary, np.ndarray) else torch.stack([wt_corrected, tc_corrected, et], dim=1)

    return result


def compute_brats_metrics(pred_logits, target, threshold=0.5):
    """
    按照BraTS官方标准计算评估指标
    返回: {'WT': {'dice': ..., 'iou': ..., ...}, 'TC': ..., 'ET': ..., 'mean': ...}
    """
    with torch.no_grad():
        # 1. 获取预测概率
        pred_probs = torch.sigmoid(pred_logits)

        # 2. 二值化
        pred_binary = (pred_probs > threshold).float()

        # 3. 强制执行层次关系
        pred_binary = enforce_brats_hierarchy(pred_binary)

        # 4. 确保维度一致
        if pred_binary.dim() == 4:
            pred_binary = pred_binary.unsqueeze(0)  # [1, 3, D, H, W]
            target = target.unsqueeze(0)

        # 5. 初始化结果
        results = {}
        class_names = ['WT', 'TC', 'ET']

        # 6. 逐类别计算
        for idx, name in enumerate(class_names):
            pred_c = pred_binary[:, idx].flatten(start_dim=1)  # [B, D*H*W]
            target_c = target[:, idx].flatten(start_dim=1)

            batch_dice = []
            batch_iou = []
            batch_sens = []
            batch_spec = []

            # 对每个样本单独计算（BraTS标准）
            for b in range(pred_c.shape[0]):
                p = pred_c[b]
                t = target_c[b]

                # BraTS Dice计算
                intersection = (p * t).sum()
                union = p.sum() + t.sum()

                if t.sum() == 0:
                    dice = float('nan')
                elif union == 0:
                    dice = 0.0
                else:
                    dice = (2.0 * intersection) / union
                batch_dice.append(dice)

                # IoU
                iou = intersection / ((p + t).clamp(0, 1).sum() + 1e-8)
                batch_iou.append(iou)

                # 敏感性和特异性
                tp = (p * t).sum()
                fp = (p * (1 - t)).sum()
                fn = ((1 - p) * t).sum()
                tn = ((1 - p) * (1 - t)).sum()

                sens = tp / (tp + fn + 1e-8)
                spec = tn / (tn + fp + 1e-8)

                batch_sens.append(sens)
                batch_spec.append(spec)

            # 存储该类别结果
            results[name] = {
                'dice': torch.tensor(batch_dice).mean().item(),
                'dice_list': [d.item() if hasattr(d, 'item') else float(d) for d in batch_dice],
                'iou': torch.tensor(batch_iou).mean().item(),
                'sensitivity': torch.tensor(batch_sens).mean().item(),
                'specificity': torch.tensor(batch_spec).mean().item(),
                'volume_pred': pred_c.sum(dim=1).mean().item(),
                'volume_target': target_c.sum(dim=1).mean().item()
            }

        # 7. 计算平均指标
        results['mean'] = {
            'dice': np.mean([results[n]['dice'] for n in class_names]),
            'iou': np.mean([results[n]['iou'] for n in class_names]),
            'sensitivity': np.mean([results[n]['sensitivity'] for n in class_names]),
            'specificity': np.mean([results[n]['specificity'] for n in class_names])
        }

        return results


def fast_adaptation(maml, adaptation_dataset, k_shot=3, inner_steps=10):
    """
    在新中心的少量样本上快速适应（内存优化版）
    """
    print(f"\n🚀 快速适应: 使用{k_shot}个样本进行{inner_steps}步适应")

    # 0. 首先清理显存
    torch.cuda.empty_cache()

    # 1. 选择适应样本（但只加载索引，不立即加载数据）
    adapt_indices = []
    tumor_samples = []
    normal_samples = []

    # 预先扫描样本，只记录索引
    for idx in range(len(adaptation_dataset)):
        # 快速检查是否有肿瘤（使用轻量方法）
        sample = adaptation_dataset[idx]
        if sample['label'].sum() > 0:  # 有肿瘤的样本
            tumor_samples.append(idx)
        else:
            normal_samples.append(idx)

    # 优先选择有肿瘤的样本
    if len(tumor_samples) >= k_shot:
        adapt_indices = np.random.choice(tumor_samples, k_shot, replace=False)
    else:
        adapt_indices = tumor_samples.copy()
        remaining = k_shot - len(tumor_samples)
        if remaining > 0 and len(normal_samples) > 0:
            additional = np.random.choice(normal_samples, min(remaining, len(normal_samples)), replace=False)
            adapt_indices.extend(additional)

    if len(adapt_indices) == 0:
        print("⚠️  没有可用于适应的样本，使用原始模型")
        return maml.model

    print(f"  选择的样本索引: {adapt_indices}")

    # 2. 克隆模型进行适应
    adapted_model = copy.deepcopy(maml.model)
    adapted_model.train()

    # 3. 使用梯度累积（内存优化）
    inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=maml.inner_lr)

    for step in range(inner_steps):
        total_loss = 0.0
        inner_optimizer.zero_grad()

        # 逐个样本处理，使用梯度累积
        for idx in adapt_indices:
            # 加载单个样本（避免同时加载所有样本）
            sample = adaptation_dataset[idx]

            # 使用更小的裁剪或下采样
            image = sample['image'].unsqueeze(0).to(maml.device)  # [1, 4, D, H, W]
            label = sample['label'].unsqueeze(0).to(maml.device)  # [1, 3, D, H, W]

            # 可选：如果图像太大，使用中心裁剪
            if image.shape[2] * image.shape[3] * image.shape[4] > 128 * 128 * 64:  # 大约8百万体素
                print(f"  样本 {idx} 太大，使用中心裁剪")
                D, H, W = image.shape[2], image.shape[3], image.shape[4]
                crop_size = (min(128, D), min(128, H), min(128, W))
                d_start = (D - crop_size[0]) // 2
                h_start = (H - crop_size[1]) // 2
                w_start = (W - crop_size[2]) // 2

                image = image[:, :,
                        d_start:d_start + crop_size[0],
                        h_start:h_start + crop_size[1],
                        w_start:w_start + crop_size[2]]
                label = label[:, :,
                        d_start:d_start + crop_size[0],
                        h_start:h_start + crop_size[1],
                        w_start:w_start + crop_size[2]]

            # 前向传播
            with torch.cuda.amp.autocast():  # 使用混合精度减少内存
                pred = adapted_model(image)
                loss = F.binary_cross_entropy_with_logits(pred, label)

                # 缩放损失，因为要累积梯度
                loss = loss / len(adapt_indices)
                total_loss += loss.item()

                # 反向传播（累积梯度）
                loss.backward()

            # 清理当前样本的显存
            del image, label, pred, loss
            torch.cuda.empty_cache()

        # 更新参数（使用累积的梯度）
        inner_optimizer.step()

        # 清理优化器状态
        inner_optimizer.zero_grad(set_to_none=True)  # 更彻底地清理梯度

        if (step + 1) % max(1, inner_steps // 5) == 0:
            print(f"    适应步 [{step + 1}/{inner_steps}], Loss: {total_loss:.4f}")

        # 每步后清理显存
        torch.cuda.empty_cache()

    adapted_model.eval()
    print("  ✅ 适应完成")

    return adapted_model


def sliding_window_inference(model, image, window_size=(224, 224, 128),
                            overlap=0.5, device='cuda', threshold=0.5):
    """
    滑动窗口推理（修正版，使用Sigmoid）
    """
    C, D, H, W = image.shape
    num_classes = 3  # WT, TC, ET

    # 步长
    step_d = int(window_size[0] * (1 - overlap))
    step_h = int(window_size[1] * (1 - overlap))
    step_w = int(window_size[2] * (1 - overlap))

    # 输出累积
    prediction = np.zeros((num_classes, D, H, W), dtype=np.float32)
    weight_map = np.zeros((D, H, W), dtype=np.float32)

    # 生成窗口位置
    positions = []
    for d in range(0, D - window_size[0] + 1, step_d):
        for h in range(0, H - window_size[1] + 1, step_h):
            for w in range(0, W - window_size[2] + 1, step_w):
                positions.append((d, h, w))

    # 添加边界位置确保覆盖
    if D > window_size[0]:
        for h in range(0, H - window_size[1] + 1, step_h):
            for w in range(0, W - window_size[2] + 1, step_w):
                positions.append((D - window_size[0], h, w))

    if H > window_size[1]:
        for d in range(0, D - window_size[0] + 1, step_d):
            for w in range(0, W - window_size[2] + 1, step_w):
                positions.append((d, H - window_size[1], w))

    if W > window_size[2]:
        for d in range(0, D - window_size[0] + 1, step_d):
            for h in range(0, H - window_size[1] + 1, step_h):
                positions.append((d, h, W - window_size[2]))

    # 去重
    positions = list(set(positions))

    print(f"  滑动窗口推理: {len(positions)} 个窗口")

    # 推理每个窗口
    model.eval()
    with torch.no_grad():
        for d, h, w in tqdm(positions, desc='窗口推理', leave=False):
            # 提取窗口
            patch = image[:,
                         d:d+window_size[0],
                         h:h+window_size[1],
                         w:w+window_size[2]]

            # 推理（使用Sigmoid）
            patch_tensor = torch.FloatTensor(patch).unsqueeze(0).to(device)
            output = model(patch_tensor)  # [1, 3, D, H, W]
            output = torch.sigmoid(output)  # 关键修改！
            output = output.squeeze(0).cpu().numpy()

            # 累积
            prediction[:,
                      d:d+window_size[0],
                      h:h+window_size[1],
                      w:w+window_size[2]] += output

            weight_map[d:d+window_size[0],
                      h:h+window_size[1],
                      w:w+window_size[2]] += 1

    # 平均
    weight_map = np.maximum(weight_map, 1)
    prediction = prediction / weight_map[np.newaxis, :, :, :]

    # 强制执行层次关系
    prediction = enforce_brats_hierarchy(prediction, threshold)

    return prediction


def center_crop_inference(model, image, crop_size=(224, 224, 128),
                         device='cuda', threshold=0.5):
    """
    中心crop推理（修正版）
    """
    C, D, H, W = image.shape

    # 计算中心crop
    d_start = max(0, (D - crop_size[0]) // 2)
    h_start = max(0, (H - crop_size[1]) // 2)
    w_start = max(0, (W - crop_size[2]) // 2)

    # Crop
    cropped = image[:,
                   d_start:d_start+crop_size[0],
                   h_start:h_start+crop_size[1],
                   w_start:w_start+crop_size[2]]

    # Padding如果需要
    if cropped.shape[1:] != crop_size:
        pad_d = max(0, crop_size[0] - cropped.shape[1])
        pad_h = max(0, crop_size[1] - cropped.shape[2])
        pad_w = max(0, crop_size[2] - cropped.shape[3])

        cropped = np.pad(cropped,
                        [(0, 0), (0, pad_d), (0, pad_h), (0, pad_w)],
                        mode='constant')

    # 推理
    model.eval()
    with torch.no_grad():
        image_tensor = torch.FloatTensor(cropped).unsqueeze(0).to(device)
        output = model(image_tensor)
        output = torch.sigmoid(output)  # 关键修改！
        prediction = output.squeeze(0).cpu().numpy()

    # 还原到原始尺寸
    full_prediction = np.zeros((3, D, H, W), dtype=np.float32)
    full_prediction[:,
                   d_start:d_start+crop_size[0],
                   h_start:h_start+crop_size[1],
                   w_start:w_start+crop_size[2]] = prediction[:, :cropped.shape[1],
                                                                  :cropped.shape[2],
                                                                  :cropped.shape[3]]

    # 强制执行层次关系
    full_prediction = enforce_brats_hierarchy(full_prediction, threshold)

    return full_prediction


# def visualize_result_3d(image, label, prediction, save_path, patient_id, metrics=None):
#     """
#     可视化3D分割结果（增强版）
#     """
#     # 选择中间切片（3个不同平面）
#     d, h, w = image.shape[1:]
#
#     # 获取概率最大的类别
#     pred_class = np.argmax(prediction, axis=0)
#     target_class = np.argmax(label, axis=0)
#
#     fig = plt.figure(figsize=(18, 10))
#
#     # 定义三个平面
#     planes = [
#         ('Axial', d // 2, 'axial'),
#         ('Coronal', h // 2, 'coronal'),
#         ('Sagittal', w // 2, 'sagittal')
#     ]
#
#     for row, (plane_name, slice_idx, plane_type) in enumerate(planes):
#         # 提取切片
#         if plane_type == 'axial':
#             img_slice = image[0, slice_idx, :, :]  # T1 modality
#             pred_slice = pred_class[slice_idx, :, :]
#             target_slice = target_class[slice_idx, :, :]
#         elif plane_type == 'coronal':
#             img_slice = image[0, :, slice_idx, :]
#             pred_slice = pred_class[:, slice_idx, :]
#             target_slice = target_class[:, slice_idx, :]
#         else:  # sagittal
#             img_slice = image[0, :, :, slice_idx]
#             pred_slice = pred_class[:, :, slice_idx]
#             target_slice = target_class[:, :, slice_idx]
#
#         # 图像
#         ax1 = plt.subplot(3, 4, row*4 + 1)
#         ax1.imshow(img_slice, cmap='gray')
#         ax1.set_title(f'{plane_name} - T1', fontsize=11, fontweight='bold')
#         ax1.axis('off')
#
#         # 真实标签
#         ax2 = plt.subplot(3, 4, row*4 + 2)
#         im2 = ax2.imshow(target_slice, cmap='jet', vmin=0, vmax=2)
#         ax2.set_title('Ground Truth', fontsize=11, fontweight='bold')
#         ax2.axis('off')
#
#         # 预测结果
#         ax3 = plt.subplot(3, 4, row*4 + 3)
#         im3 = ax3.imshow(pred_slice, cmap='jet', vmin=0, vmax=2)
#         ax3.set_title('Prediction', fontsize=11, fontweight='bold')
#         ax3.axis('off')
#
#         # 叠加显示
#         ax4 = plt.subplot(3, 4, row*4 + 4)
#         ax4.imshow(img_slice, cmap='gray')
#         ax4.imshow(pred_slice, cmap='jet', alpha=0.5, vmin=0, vmax=2)
#         ax4.set_title('Overlay', fontsize=11, fontweight='bold')
#         ax4.axis('off')
#
#     # 添加颜色条
#     plt.subplots_adjust(right=0.85)
#     cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
#     cbar = fig.colorbar(im3, cax=cbar_ax)
#     cbar.set_ticks([0, 1, 2])
#     cbar.set_ticklabels(['Background', 'TC/NCR', 'WT/ED'])
#
#     # 添加指标文本
#     if metrics:
#         metrics_text = f"Patient: {patient_id}\n\n"
#         metrics_text += "Dice Scores:\n"
#         for region in ['WT', 'TC', 'ET']:
#             metrics_text += f"  {region}: {metrics[region]['dice']:.4f}\n"
#         metrics_text += f"\nMean Dice: {metrics['mean']['dice']:.4f}"
#
#         fig.text(0.02, 0.5, metrics_text, fontsize=10,
#                 verticalalignment='center', family='monospace',
#                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
#
#     plt.suptitle(f'BraTS Segmentation Results - {patient_id}',
#                 fontsize=14, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 0.85, 0.95])
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close()

def visualize_result_3d(image, label, prediction, save_path, patient_id, metrics=None):
    """
    可视化3D分割结果（修正版 - 多标签显示）
    """
    d, h, w = image.shape[1:]

    # ⚠️ 关键修复：不使用argmax！
    # BraTS标签是多标签，需要分别显示

    fig = plt.figure(figsize=(20, 12))

    # 选择三个正交平面
    planes = [
        ('Axial', d // 2, 0),
        ('Coronal', h // 2, 1),
        ('Sagittal', w // 2, 2)
    ]

    class_names = ['WT', 'TC', 'ET']
    class_colors = ['Reds', 'Greens', 'Blues']

    for row, (plane_name, slice_idx, axis) in enumerate(planes):
        # 提取切片
        if axis == 0:  # Axial
            img_slice = image[0, slice_idx, :, :]
            label_slices = [label[i, slice_idx, :, :] for i in range(3)]
            pred_slices = [prediction[i, slice_idx, :, :] for i in range(3)]
        elif axis == 1:  # Coronal
            img_slice = image[0, :, slice_idx, :]
            label_slices = [label[i, :, slice_idx, :] for i in range(3)]
            pred_slices = [prediction[i, :, slice_idx, :] for i in range(3)]
        else:  # Sagittal
            img_slice = image[0, :, :, slice_idx]
            label_slices = [label[i, :, :, slice_idx] for i in range(3)]
            pred_slices = [prediction[i, :, :, slice_idx] for i in range(3)]

        # 第1列：原始图像
        ax1 = plt.subplot(3, 5, row * 5 + 1)
        ax1.imshow(img_slice, cmap='gray')
        ax1.set_title(f'{plane_name}\nT1', fontsize=10, fontweight='bold')
        ax1.axis('off')

        # 第2列：GT组合显示（RGB）
        ax2 = plt.subplot(3, 5, row * 5 + 2)
        gt_rgb = np.zeros((*img_slice.shape, 3))
        gt_rgb[..., 0] = label_slices[0]  # WT -> Red
        gt_rgb[..., 1] = label_slices[1]  # TC -> Green
        gt_rgb[..., 2] = label_slices[2]  # ET -> Blue
        ax2.imshow(img_slice, cmap='gray')
        ax2.imshow(gt_rgb, alpha=0.5)
        ax2.set_title('Ground Truth\n(WT+TC+ET)', fontsize=10, fontweight='bold')
        ax2.axis('off')

        # 第3列：预测组合显示（RGB）
        ax3 = plt.subplot(3, 5, row * 5 + 3)
        pred_rgb = np.zeros((*img_slice.shape, 3))
        pred_rgb[..., 0] = pred_slices[0]  # WT -> Red
        pred_rgb[..., 1] = pred_slices[1]  # TC -> Green
        pred_rgb[..., 2] = pred_slices[2]  # ET -> Blue
        ax3.imshow(img_slice, cmap='gray')
        ax3.imshow(pred_rgb, alpha=0.5)
        ax3.set_title('Prediction\n(WT+TC+ET)', fontsize=10, fontweight='bold')
        ax3.axis('off')

        # 第4列：分别显示三个类别（GT vs Pred）
        ax4 = plt.subplot(3, 5, row * 5 + 4)
        # 创建三通道图：R=WT, G=TC, B=ET
        comparison = np.zeros((*img_slice.shape, 3))
        for i, (gt, pred) in enumerate(zip(label_slices, pred_slices)):
            # TP: 白色, FP: 预测颜色, FN: GT颜色半透明
            tp = (gt > 0.5) & (pred > 0.5)
            fp = (gt <= 0.5) & (pred > 0.5)
            fn = (gt > 0.5) & (pred <= 0.5)

            comparison[tp, i] = 1.0  # 正确预测
            comparison[fp, i] = 0.7  # 假阳性（橙色系）
            comparison[fn, i] = 0.3  # 假阴性（暗色系）

        ax4.imshow(comparison)
        ax4.set_title('Comparison\n(White=TP)', fontsize=10, fontweight='bold')
        ax4.axis('off')

        # 第5列：单独显示ET（因为ET容易漏掉）
        ax5 = plt.subplot(3, 5, row * 5 + 5)
        ax5.imshow(img_slice, cmap='gray')

        # ET的GT和预测
        et_combined = np.zeros((*img_slice.shape, 3))
        et_combined[label_slices[2] > 0.5, 0] = 1.0  # GT红色
        et_combined[pred_slices[2] > 0.5, 1] = 1.0  # Pred绿色
        # 重叠部分会变成黄色

        ax5.imshow(et_combined, alpha=0.8)
        ax5.set_title(f'ET Only\n(GT={label_slices[2].sum():.0f})',
                      fontsize=10, fontweight='bold')
        ax5.axis('off')

    # 添加指标文本
    if metrics:
        metrics_text = f"Patient: {patient_id}\n\n"
        metrics_text += "Dice Scores:\n"
        for region in ['WT', 'TC', 'ET']:
            metrics_text += f"  {region}: {metrics[region]['dice']:.4f}\n"
            if region == 'ET' and metrics[region]['dice'] == 0:
                metrics_text += f"       ⚠️ ET prediction is empty!\n"
        metrics_text += f"\nMean: {metrics['mean']['dice']:.4f}\n"
        metrics_text += f"\nLegend:\n"
        metrics_text += f"  Red=WT, Green=TC, Blue=ET\n"
        metrics_text += f"  White=Correct Prediction\n"
        metrics_text += f"  Orange=False Positive\n"
        metrics_text += f"  Dark=False Negative"

        fig.text(0.02, 0.5, metrics_text, fontsize=9,
                 verticalalignment='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.suptitle(f'BraTS Multi-Label Segmentation - {patient_id}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.15, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def restore_to_original_size(prediction, crop_info, original_shape):
    """
    将裁剪后的预测恢复到原始尺寸

    参数:
        prediction: [3, D_crop, H_crop, W_crop] 裁剪后的预测
        crop_info: 包含crop_start和original_shape的字典
        original_shape: (D, H, W) 原始尺寸

    返回:
        restored: [3, D, H, W] 恢复到原始尺寸的预测
    """
    # 创建原始尺寸的全零数组
    restored = np.zeros((3, *original_shape), dtype=np.float32)

    # 获取crop位置
    crop_start = crop_info.get('crop_start', None)

    if crop_start is None:
        # 如果没有crop_start信息，假设是中心crop
        d_start = (original_shape[0] - prediction.shape[1]) // 2
        h_start = (original_shape[1] - prediction.shape[2]) // 2
        w_start = (original_shape[2] - prediction.shape[3]) // 2
        crop_start = [max(0, d_start), max(0, h_start), max(0, w_start)]

    # 计算实际可以放置的尺寸
    d_size = min(prediction.shape[1], original_shape[0] - crop_start[0])
    h_size = min(prediction.shape[2], original_shape[1] - crop_start[1])
    w_size = min(prediction.shape[3], original_shape[2] - crop_start[2])

    # 将预测放回原始位置
    restored[:,
    crop_start[0]:crop_start[0] + d_size,
    crop_start[1]:crop_start[1] + h_size,
    crop_start[2]:crop_start[2] + w_size] = prediction[:,
                                            :d_size,
                                            :h_size,
                                            :w_size]

    return restored


def whole_volume_inference(model, image, crop_size=(224, 224, 128),
                           device='cuda', threshold=0.5):
    """
    整个体积的推理（使用滑动窗口 + 恢复原始尺寸）

    这个函数不做任何裁剪，直接处理整个体积
    """
    C, D, H, W = image.shape

    # 如果图像小于crop_size，直接推理
    if D <= crop_size[0] and H <= crop_size[1] and W <= crop_size[2]:
        # Padding到crop_size
        padded = np.zeros((C, *crop_size), dtype=np.float32)
        padded[:, :D, :H, :W] = image

        model.eval()
        with torch.no_grad():
            img_tensor = torch.FloatTensor(padded).unsqueeze(0).to(device)
            output = model(img_tensor)
            output = torch.sigmoid(output)
            pred = output.squeeze(0).cpu().numpy()

        # 裁剪回原始尺寸
        pred = pred[:, :D, :H, :W]

        # 强制层次关系
        pred = enforce_brats_hierarchy(pred, threshold)

        return pred

    # 否则使用滑动窗口
    return sliding_window_inference(
        model, image, crop_size,
        overlap=0.5, device=device, threshold=threshold
    )
def save_brats_prediction(pred_probs, save_path, threshold=0.5):
    """
    保存预测结果为BraTS格式的NIfTI文件
    标签映射: 0-背景, 1-坏死(NCR), 2-水肿(ED), 4-增强(ET)
    """
    try:
        # 二值化
        pred_binary = (pred_probs > threshold).astype(np.float32)

        # 强制执行层次关系
        pred_binary = enforce_brats_hierarchy(pred_binary)

        # 转换为BraTS标签格式
        wt, tc, et = pred_binary[0], pred_binary[1], pred_binary[2]

        output = np.zeros(wt.shape, dtype=np.uint8)

        # BraTS标签格式
        output[wt == 1] = 2  # 水肿区域
        output[tc == 1] = 1  # 肿瘤核心（坏死）
        output[et == 1] = 4  # 增强肿瘤

        # 确保层次关系（再次检查）
        # 增强肿瘤应该在肿瘤核心内
        output[(et == 1) & (tc == 0)] = 4  # 应该不会发生

        # 创建NIfTI图像
        nii_img = nib.Nifti1Image(output, affine=np.eye(4))
        nib.save(nii_img, save_path)

        return True
    except Exception as e:
        print(f"警告: 保存NIfTI失败: {e}")
        return False


# def test_single_task(maml, config, task_name, output_base_dir):
#     """
#     测试单个任务（支持快速适应）
#     """
#     print(f"\n{'='*70}")
#     print(f"测试任务: {task_name}")
#     print('='*70)
#
#     # 创建测试数据集
#     test_dataset = BraTSDataset(
#         data_root=config['data']['data_root'],
#         task_name=task_name,
#         mode='test',
#         crop_size=tuple(config['data']['crop_size']),
#         crop_strategy='smart_random',
#         normalize=True,
#         augment_type='none'
#     )
#
#     if len(test_dataset) == 0:
#         print(f"⚠️  {task_name} 没有测试数据")
#         return None
#
#     print(f"测试样本数: {len(test_dataset)}")
#
#     # 创建输出目录
#     task_output_dir = Path(output_base_dir) / task_name
#     task_output_dir.mkdir(parents=True, exist_ok=True)
#
#     vis_dir = task_output_dir / 'visualizations'
#     pred_dir = task_output_dir / 'predictions'
#
#     if config['testing']['visualization']:
#         vis_dir.mkdir(exist_ok=True)
#     if config['testing']['save_predictions']:
#         pred_dir.mkdir(exist_ok=True)
#
#     # ========== 快速适应阶段 ==========
#     inference_model = maml.model
#     adaptation_info = {"adapted": False, "k_shot": 0, "steps": 0}
#
#     if config['testing']['enable_adaptation']:
#         k_shot = config['testing']['adaptation_k_shot']
#         inner_steps = config['testing']['adaptation_inner_steps']
#
#         # 创建适应数据集（从测试集中采样）
#         adapt_dataset = BraTSDataset(
#             data_root=config['data']['data_root'],
#             task_name=task_name,
#             mode='test',
#             crop_size=tuple(config['data']['crop_size']),
#             crop_strategy='smart_random',
#             normalize=True,
#             augment_type='none'
#         )
#
#         # 快速适应
#         adapted_model = fast_adaptation(
#             maml, adapt_dataset,
#             k_shot=k_shot,
#             inner_steps=inner_steps
#         )
#
#         inference_model = adapted_model
#         adaptation_info = {"adapted": True, "k_shot": k_shot, "steps": inner_steps}
#
#     # ========== 推理配置 ==========
#     inference_mode = config['testing']['inference']['mode']
#     window_size = tuple(config['data']['crop_size'])
#     overlap = config['testing']['inference']['overlap']
#     threshold = config['testing']['threshold']
#
#     # ========== 测试每个样本 ==========
#     all_metrics = []
#     processing_times = []
#     failed_samples = []
#
#     for idx in tqdm(range(len(test_dataset)), desc=f'测试 {task_name}'):
#         sample = test_dataset[idx]
#
#         image = sample['image'].numpy()  # [4, D, H, W] - 裁剪后的
#         label = sample['label'].numpy()  # [3, D, H, W] - 裁剪后的
#         patient_id = sample['patient_id']
#         crop_info = sample['crop_info']
#         original_shape = sample['original_shape']  # ⚠️ 获取原始尺寸
#
#         try:
#             start_time = time.time()
#
#             # 推理（在裁剪后的图像上）
#             if inference_mode == 'sliding_window':
#                 prediction_cropped = sliding_window_inference(
#                     inference_model, image, window_size,
#                     overlap, maml.device, threshold
#                 )
#             else:
#                 prediction_cropped = center_crop_inference(
#                     inference_model, image, window_size,
#                     maml.device, threshold
#                 )
#
#             # ⚠️ 恢复到原始尺寸
#             prediction = restore_to_original_size(
#                 prediction_cropped,
#                 crop_info,
#                 original_shape
#             )
#
#             processing_time = time.time() - start_time
#
#             # ⚠️ 注意：这里的label也需要恢复到原始尺寸来计算指标
#             # 但如果原始标签太大，可以只在crop区域计算
#             # 这里我们简化处理，只在crop区域计算指标
#
#             # 计算指标（使用裁剪区域）
#             pred_logits = torch.FloatTensor(prediction_cropped)
#             target_tensor = torch.FloatTensor(label).unsqueeze(0)
#
#             metrics = compute_brats_metrics(
#                 pred_logits.unsqueeze(0),
#                 target_tensor,
#                 threshold
#             )
#
#             metrics['patient_id'] = patient_id
#             metrics['processing_time'] = processing_time
#             all_metrics.append(metrics)
#
#             # 保存预测结果
#             if config['testing']['save_predictions']:
#                 pred_path = pred_dir / f"{patient_id}_pred.nii.gz"
#                 save_brats_prediction(prediction, str(pred_path), threshold)  # 使用恢复后的
#
#                 # 可视化（使用裁剪区域，因为label也是裁剪的）
#             if config['testing']['visualization']:
#                 vis_path = vis_dir / f"{patient_id}_result.png"
#                 visualize_result_3d(
#                     image, label, prediction_cropped,  # 使用裁剪版本
#                     str(vis_path), patient_id, metrics
#                 )
#
#             # 打印单个样本结果
#             if config['testing']['verbose']:
#                 print(f"\n  {patient_id}:")
#                 print(f"    WT Dice: {metrics['WT']['dice']:.4f}")
#                 print(f"    TC Dice: {metrics['TC']['dice']:.4f}")
#                 print(f"    ET Dice: {metrics['ET']['dice']:.4f}")
#                 print(f"    Mean Dice: {metrics['mean']['dice']:.4f}")
#                 print(f"    Time: {processing_time:.2f}s")
#
#         except Exception as e:
#             print(f"\n⚠️  处理 {patient_id} 时出错: {e}")
#             failed_samples.append(patient_id)
#             continue
#
#     # ========== 汇总结果 ==========
#     if len(all_metrics) == 0:
#         print(f"❌  {task_name} 没有成功处理的样本")
#         return None
#
#     # 计算平均指标
#     summary = {
#         'task_name': task_name,
#         'num_samples': len(all_metrics),
#         'failed_samples': failed_samples,
#         'adaptation': adaptation_info,
#         'avg_processing_time': float(np.mean(processing_times)),
#         'metrics': {}
#     }
#
#     class_names = ['WT', 'TC', 'ET']
#     for name in class_names:
#         dice_values = [m[name]['dice'] for m in all_metrics]
#         summary['metrics'][name] = {
#             'dice_mean': float(np.mean(dice_values)),
#             'dice_std': float(np.std(dice_values)),
#             'dice_min': float(np.min(dice_values)),
#             'dice_max': float(np.max(dice_values)),
#             'sensitivity_mean': float(np.mean([m[name]['sensitivity'] for m in all_metrics])),
#             'specificity_mean': float(np.mean([m[name]['specificity'] for m in all_metrics]))
#         }
#
#     # 总体平均
#     summary['metrics']['overall'] = {
#         'dice_mean': float(np.mean([summary['metrics'][n]['dice_mean'] for n in class_names])),
#         'dice_std': float(np.mean([summary['metrics'][n]['dice_std'] for n in class_names]))
#     }
#
#     # 打印结果
#     print(f"\n{'='*70}")
#     print(f"{task_name} 测试结果汇总")
#     print('='*70)
#     print(f"成功处理: {len(all_metrics)}/{len(test_dataset)} 个样本")
#     if failed_samples:
#         print(f"失败样本: {', '.join(failed_samples)}")
#
#     print(f"\n平均处理时间: {summary['avg_processing_time']:.2f}s")
#
#     print(f"\n{'指标':<15} {'WT':>10} {'TC':>10} {'ET':>10}")
#     print("-" * 55)
#
#     for metric in ['dice_mean', 'dice_std', 'sensitivity_mean', 'specificity_mean']:
#         metric_name = metric.replace('_mean', '').replace('_std', ' std').title()
#         values = [summary['metrics'][n][metric] for n in class_names]
#         print(f"{metric_name:<15} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")
#
#     print(f"\n总体平均Dice: {summary['metrics']['overall']['dice_mean']:.4f}")
#     print('='*70)
#
#     # 保存详细结果
#     detailed_results = {
#         'summary': summary,
#         'all_metrics': all_metrics,
#         'config': config['testing']
#     }
#
#     result_file = task_output_dir / f"results_{task_name}.json"
#     with open(result_file, 'w') as f:
#         json.dump(detailed_results, f, indent=2)
#
#     print(f"详细结果已保存到: {result_file}")
#
#     return summary

def test_single_task(maml, config, task_name, output_base_dir):
    """
    测试单个任务（支持快速适应）- 修正版，分别记录适应样本和测试样本结果
    """
    print(f"\n{'=' * 70}")
    print(f"测试任务: {task_name}")
    print('=' * 70)

    # 创建测试数据集
    test_dataset = BraTSDataset(
        data_root=config['data']['data_root'],
        task_name=task_name,
        mode='test',
        crop_size=tuple(config['data']['crop_size']),
        crop_strategy='smart_random',
        normalize=True,
        augment_type='none'
    )

    if len(test_dataset) == 0:
        print(f"⚠️  {task_name} 没有测试数据")
        return None

    print(f"测试样本数: {len(test_dataset)}")

    # 创建输出目录
    task_output_dir = Path(output_base_dir) / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    # ========== 数据分割：适应样本 vs 测试样本 ==========
    total_samples = len(test_dataset)
    k_shot = config['testing']['adaptation_k_shot']

    # 如果样本数不足以分割，调整k_shot
    if k_shot >= total_samples:
        print(f"⚠️  k_shot({k_shot}) >= 总样本数({total_samples})，使用所有样本进行适应")
        k_shot = max(1, total_samples // 2)  # 至少保留一半样本用于测试

    # 固定随机种子以确保可重复性
    import random
    random_seed = config['testing'].get('random_seed', 42)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # 优先选择有肿瘤的样本作为适应样本
    print("  扫描样本，查找有肿瘤的样本...")
    tumor_indices = []
    normal_indices = []

    for idx in range(total_samples):
        sample = test_dataset[idx]
        if sample['label'].sum() > 0:  # 有肿瘤
            tumor_indices.append(idx)
        else:
            normal_indices.append(idx)

    print(f"  有肿瘤样本: {len(tumor_indices)}个, 正常样本: {len(normal_indices)}个")

    # 选择适应样本
    adapt_indices = []
    if len(tumor_indices) >= k_shot:
        adapt_indices = random.sample(tumor_indices, k_shot)
    else:
        adapt_indices = tumor_indices.copy()
        remaining = k_shot - len(tumor_indices)
        if remaining > 0 and len(normal_indices) >= remaining:
            adapt_indices.extend(random.sample(normal_indices, remaining))
        else:
            # 如果样本不足，使用所有样本
            adapt_indices = list(range(min(k_shot, total_samples)))

    # 测试样本是剩余的样本
    all_indices = set(range(total_samples))
    adapt_set = set(adapt_indices)
    test_eval_indices = list(all_indices - adapt_set)

    print(f"  适应样本: {len(adapt_indices)}个 (索引: {sorted(adapt_indices)})")
    print(f"  测试样本: {len(test_eval_indices)}个 (索引: {sorted(test_eval_indices)})")

    # 创建数据集包装器
    class IndexedDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, indices):
            self.original_dataset = original_dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            original_idx = self.indices[idx]
            sample = self.original_dataset[original_idx]
            sample['original_idx'] = original_idx
            return sample

    adapt_dataset = IndexedDataset(test_dataset, adapt_indices)
    test_eval_dataset = IndexedDataset(test_dataset, test_eval_indices)

    # ========== 创建可视化目录 ==========
    vis_dir = task_output_dir / 'visualizations'
    pred_dir = task_output_dir / 'predictions'

    if config['testing']['visualization']:
        vis_dir.mkdir(exist_ok=True)
        # 为适应样本和测试样本分别创建子目录
        vis_adapt_dir = vis_dir / 'adaptation_samples'
        vis_test_dir = vis_dir / 'test_samples'
        vis_adapt_dir.mkdir(exist_ok=True)
        vis_test_dir.mkdir(exist_ok=True)

    if config['testing']['save_predictions']:
        pred_dir.mkdir(exist_ok=True)
        pred_adapt_dir = pred_dir / 'adaptation_samples'
        pred_test_dir = pred_dir / 'test_samples'
        pred_adapt_dir.mkdir(exist_ok=True)
        pred_test_dir.mkdir(exist_ok=True)

    # ========== 快速适应阶段 ==========
    inference_model = maml.model
    adaptation_info = {"adapted": False, "k_shot": 0, "steps": 0, "adapt_indices": []}

    if config['testing']['enable_adaptation']:
        k_shot_actual = len(adapt_dataset)
        inner_steps = config['testing']['adaptation_inner_steps']

        print(f"\n🚀 快速适应阶段")
        print(f"  使用 {k_shot_actual} 个样本进行 {inner_steps} 步适应")
        print(f"  适应样本索引: {sorted(adapt_indices)}")

        # 快速适应
        adapted_model = fast_adaptation(
            maml, adapt_dataset,
            k_shot=k_shot_actual,
            inner_steps=inner_steps
        )

        inference_model = adapted_model
        adaptation_info = {
            "adapted": True,
            "k_shot": k_shot_actual,
            "steps": inner_steps,
            "adapt_indices": adapt_indices
        }

    # ========== 推理配置 ==========
    inference_mode = config['testing']['inference']['mode']
    window_size = tuple(config['data']['crop_size'])
    overlap = config['testing']['inference']['overlap']
    threshold = config['testing']['threshold']

    # ========== 分别测试适应样本和测试样本 ==========
    adapt_metrics = []
    test_metrics = []
    all_metrics = []  # 保持兼容性，所有样本的结果
    processing_times = []
    failed_samples = []

    print(f"\n📊 测试阶段")
    print(f"  1. 测试适应样本 ({len(adapt_dataset)}个)")

    # 测试适应样本
    for idx in tqdm(range(len(adapt_dataset)), desc='测试适应样本'):
        sample = adapt_dataset[idx]
        original_idx = sample['original_idx']

        image = sample['image'].numpy()
        label = sample['label'].numpy()
        patient_id = sample['patient_id']
        crop_info = sample['crop_info']
        original_shape = sample['original_shape']

        try:
            start_time = time.time()

            # 推理
            if inference_mode == 'sliding_window':
                prediction_cropped = sliding_window_inference(
                    inference_model, image, window_size,
                    overlap, maml.device, threshold
                )
            else:
                prediction_cropped = center_crop_inference(
                    inference_model, image, window_size,
                    maml.device, threshold
                )

            # 恢复到原始尺寸
            prediction = restore_to_original_size(
                prediction_cropped,
                crop_info,
                original_shape
            )

            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # 计算指标
            pred_logits = torch.FloatTensor(prediction_cropped)
            target_tensor = torch.FloatTensor(label).unsqueeze(0)

            metrics = compute_brats_metrics(
                pred_logits.unsqueeze(0),
                target_tensor,
                threshold
            )

            metrics['patient_id'] = patient_id
            metrics['original_idx'] = original_idx
            metrics['sample_type'] = 'adaptation'  # 标记为适应样本
            metrics['processing_time'] = processing_time

            adapt_metrics.append(metrics)
            all_metrics.append(metrics)  # 添加到总结果

            # 保存预测结果
            if config['testing']['save_predictions']:
                pred_path = pred_adapt_dir / f"{patient_id}_pred.nii.gz"
                save_brats_prediction(prediction, str(pred_path), threshold)

            # 可视化
            if config['testing']['visualization']:
                vis_path = vis_adapt_dir / f"{patient_id}_result.png"
                visualize_result_3d(
                    image, label, prediction_cropped,
                    str(vis_path), patient_id, metrics
                )

        except Exception as e:
            print(f"\n⚠️  处理适应样本 {patient_id} 时出错: {e}")
            failed_samples.append((patient_id, 'adaptation'))
            continue

    print(f"\n  2. 测试剩余样本 ({len(test_eval_dataset)}个)")

    # 测试剩余样本
    for idx in tqdm(range(len(test_eval_dataset)), desc='测试剩余样本'):
        sample = test_eval_dataset[idx]
        original_idx = sample['original_idx']

        image = sample['image'].numpy()
        label = sample['label'].numpy()
        patient_id = sample['patient_id']
        crop_info = sample['crop_info']
        original_shape = sample['original_shape']

        try:
            start_time = time.time()

            # 推理
            if inference_mode == 'sliding_window':
                prediction_cropped = sliding_window_inference(
                    inference_model, image, window_size,
                    overlap, maml.device, threshold
                )
            else:
                prediction_cropped = center_crop_inference(
                    inference_model, image, window_size,
                    maml.device, threshold
                )

            # 恢复到原始尺寸
            prediction = restore_to_original_size(
                prediction_cropped,
                crop_info,
                original_shape
            )

            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # 计算指标
            pred_logits = torch.FloatTensor(prediction_cropped)
            target_tensor = torch.FloatTensor(label).unsqueeze(0)

            metrics = compute_brats_metrics(
                pred_logits.unsqueeze(0),
                target_tensor,
                threshold
            )

            metrics['patient_id'] = patient_id
            metrics['original_idx'] = original_idx
            metrics['sample_type'] = 'test'  # 标记为测试样本
            metrics['processing_time'] = processing_time

            test_metrics.append(metrics)
            all_metrics.append(metrics)  # 添加到总结果

            # 保存预测结果
            if config['testing']['save_predictions']:
                pred_path = pred_test_dir / f"{patient_id}_pred.nii.gz"
                save_brats_prediction(prediction, str(pred_path), threshold)

            # 可视化
            if config['testing']['visualization']:
                vis_path = vis_test_dir / f"{patient_id}_result.png"
                visualize_result_3d(
                    image, label, prediction_cropped,
                    str(vis_path), patient_id, metrics
                )

        except Exception as e:
            print(f"\n⚠️  处理测试样本 {patient_id} 时出错: {e}")
            failed_samples.append((patient_id, 'test'))
            continue

    # ========== 汇总结果 ==========
    if len(all_metrics) == 0:
        print(f"❌  {task_name} 没有成功处理的样本")
        return None

    # 计算总体平均指标（保持兼容性）
    summary = {
        'task_name': task_name,
        'total_samples': total_samples,
        'adaptation_samples': len(adapt_dataset),
        'test_samples': len(test_eval_dataset),
        'successful_adaptation': len(adapt_metrics),
        'successful_test': len(test_metrics),
        'failed_samples': failed_samples,
        'adaptation_info': adaptation_info,
        'avg_processing_time': float(np.mean(processing_times)),
        'metrics': {},  # 总体指标
        'adaptation_metrics': {},  # 适应样本指标
        'test_metrics': {}  # 测试样本指标
    }

    # ========== 计算适应样本指标 ==========
    class_names = ['WT', 'TC', 'ET']

    # 适应样本指标
    if len(adapt_metrics) > 0:
        for name in class_names:
            dice_values = [m[name]['dice'] for m in adapt_metrics if not np.isnan(m[name]['dice'])]

            if len(dice_values) > 0:
                summary['adaptation_metrics'][name] = {
                    'dice_mean': float(np.mean(dice_values)),
                    'dice_std': float(np.std(dice_values)),
                    'dice_min': float(np.min(dice_values)),
                    'dice_max': float(np.max(dice_values)),
                    'sensitivity_mean': float(np.nanmean([m[name]['sensitivity'] for m in adapt_metrics])),
                    'specificity_mean': float(np.nanmean([m[name]['specificity'] for m in adapt_metrics])),
                    'num_samples': len(dice_values)
                }
            else:
                summary['adaptation_metrics'][name] = {
                    'dice_mean': float('nan'),
                    'dice_std': float('nan'),
                    'dice_min': float('nan'),
                    'dice_max': float('nan'),
                    'sensitivity_mean': float('nan'),
                    'specificity_mean': float('nan'),
                    'num_samples': len(adapt_metrics)
                }

        # 适应样本总体平均
        valid_dice_means = [summary['adaptation_metrics'][n]['dice_mean']
                            for n in class_names if not np.isnan(summary['adaptation_metrics'][n]['dice_mean'])]

        summary['adaptation_metrics']['overall'] = {
            'dice_mean': float(np.mean(valid_dice_means)) if valid_dice_means else float('nan'),
            'dice_std': float(np.std(valid_dice_means)) if valid_dice_means else float('nan')
        }

    # ========== 计算测试样本指标 ==========
    if len(test_metrics) > 0:
        for name in class_names:
            dice_values = [m[name]['dice'] for m in test_metrics if not np.isnan(m[name]['dice'])]

            if len(dice_values) > 0:
                summary['test_metrics'][name] = {
                    'dice_mean': float(np.mean(dice_values)),
                    'dice_std': float(np.std(dice_values)),
                    'dice_min': float(np.min(dice_values)),
                    'dice_max': float(np.max(dice_values)),
                    'sensitivity_mean': float(np.nanmean([m[name]['sensitivity'] for m in test_metrics])),
                    'specificity_mean': float(np.nanmean([m[name]['specificity'] for m in test_metrics])),
                    'num_samples': len(dice_values)
                }
            else:
                summary['test_metrics'][name] = {
                    'dice_mean': float('nan'),
                    'dice_std': float('nan'),
                    'dice_min': float('nan'),
                    'dice_max': float('nan'),
                    'sensitivity_mean': float('nan'),
                    'specificity_mean': float('nan'),
                    'num_samples': len(test_metrics)
                }

        # 测试样本总体平均
        valid_dice_means = [summary['test_metrics'][n]['dice_mean']
                            for n in class_names if not np.isnan(summary['test_metrics'][n]['dice_mean'])]

        summary['test_metrics']['overall'] = {
            'dice_mean': float(np.mean(valid_dice_means)) if valid_dice_means else float('nan'),
            'dice_std': float(np.std(valid_dice_means)) if valid_dice_means else float('nan')
        }

    # ========== 计算总体指标（兼容性） ==========
    for name in class_names:
        dice_values = [m[name]['dice'] for m in all_metrics if not np.isnan(m[name]['dice'])]

        if len(dice_values) > 0:
            summary['metrics'][name] = {
                'dice_mean': float(np.mean(dice_values)),
                'dice_std': float(np.std(dice_values)),
                'dice_min': float(np.min(dice_values)),
                'dice_max': float(np.max(dice_values)),
                'sensitivity_mean': float(np.nanmean([m[name]['sensitivity'] for m in all_metrics])),
                'specificity_mean': float(np.nanmean([m[name]['specificity'] for m in all_metrics]))
            }
        else:
            summary['metrics'][name] = {
                'dice_mean': float('nan'),
                'dice_std': float('nan'),
                'dice_min': float('nan'),
                'dice_max': float('nan'),
                'sensitivity_mean': float('nan'),
                'specificity_mean': float('nan')
            }

    # 总体平均
    valid_dice_means = [summary['metrics'][n]['dice_mean']
                        for n in class_names if not np.isnan(summary['metrics'][n]['dice_mean'])]

    summary['metrics']['overall'] = {
        'dice_mean': float(np.mean(valid_dice_means)) if valid_dice_means else float('nan'),
        'dice_std': float(np.std(valid_dice_means)) if valid_dice_means else float('nan')
    }

    # ========== 打印结果 ==========
    print(f"\n{'=' * 70}")
    print(f"{task_name} 测试结果汇总")
    print('=' * 70)
    print(f"总样本数: {total_samples}")
    print(f"适应样本: {len(adapt_dataset)}个 (索引: {sorted(adapt_indices)})")
    print(f"测试样本: {len(test_eval_dataset)}个")
    print(f"成功测试: {len(all_metrics)}/{total_samples}个")

    if failed_samples:
        print(f"失败样本: {failed_samples}")

    # 1. 打印适应样本结果
    print(f"\n📊 适应样本结果 ({len(adapt_metrics)}个):")
    if len(adapt_metrics) > 0:
        print(f"{'指标':<15} {'WT':>10} {'TC':>10} {'ET':>10}")
        print("-" * 55)

        for metric in ['dice_mean', 'dice_std', 'sensitivity_mean', 'specificity_mean']:
            metric_name = metric.replace('_mean', '').replace('_std', ' std').title()
            values = []
            for name in class_names:
                val = summary['adaptation_metrics'][name][metric]
                values.append(f"{val:.4f}" if not np.isnan(val) else "nan")

            print(f"{metric_name:<15} {values[0]:>10} {values[1]:>10} {values[2]:>10}")

        if not np.isnan(summary['adaptation_metrics']['overall']['dice_mean']):
            print(f"\n适应样本平均Dice: {summary['adaptation_metrics']['overall']['dice_mean']:.4f}")
    else:
        print("  无适应样本结果")

    # 2. 打印测试样本结果
    print(f"\n📊 测试样本结果 ({len(test_metrics)}个):")
    if len(test_metrics) > 0:
        print(f"{'指标':<15} {'WT':>10} {'TC':>10} {'ET':>10}")
        print("-" * 55)

        for metric in ['dice_mean', 'dice_std', 'sensitivity_mean', 'specificity_mean']:
            metric_name = metric.replace('_mean', '').replace('_std', ' std').title()
            values = []
            for name in class_names:
                val = summary['test_metrics'][name][metric]
                values.append(f"{val:.4f}" if not np.isnan(val) else "nan")

            print(f"{metric_name:<15} {values[0]:>10} {values[1]:>10} {values[2]:>10}")

        if not np.isnan(summary['test_metrics']['overall']['dice_mean']):
            print(f"\n测试样本平均Dice: {summary['test_metrics']['overall']['dice_mean']:.4f}")
    else:
        print("  无测试样本结果")

    # 3. 打印总体结果（保持原有格式）
    print(f"\n📊 总体结果 ({len(all_metrics)}个):")
    print(f"{'指标':<15} {'WT':>10} {'TC':>10} {'ET':>10}")
    print("-" * 55)

    for metric in ['dice_mean', 'dice_std', 'sensitivity_mean', 'specificity_mean']:
        metric_name = metric.replace('_mean', '').replace('_std', ' std').title()
        values = [summary['metrics'][n][metric] for n in class_names]
        print(f"{metric_name:<15} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")

    if not np.isnan(summary['metrics']['overall']['dice_mean']):
        print(f"\n总体平均Dice: {summary['metrics']['overall']['dice_mean']:.4f}")

    print('=' * 70)

    # ========== 保存详细结果 ==========
    detailed_results = {
        'summary': summary,
        'all_metrics': all_metrics,  # 所有样本
        'adaptation_metrics': adapt_metrics,  # 适应样本详细结果
        'test_metrics': test_metrics,  # 测试样本详细结果
        'adaptation_indices': adapt_indices,
        'test_indices': test_eval_indices,
        'config': config['testing']
    }

    result_file = task_output_dir / f"results_{task_name}.json"
    with open(result_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    print(f"详细结果已保存到: {result_file}")

    # 也保存单独的结果文件
    adapt_result_file = task_output_dir / f"adaptation_results_{task_name}.json"
    test_result_file = task_output_dir / f"test_results_{task_name}.json"

    with open(adapt_result_file, 'w') as f:
        json.dump({
            'summary': summary['adaptation_metrics'],
            'metrics': adapt_metrics,
            'indices': adapt_indices
        }, f, indent=2)

    with open(test_result_file, 'w') as f:
        json.dump({
            'summary': summary['test_metrics'],
            'metrics': test_metrics,
            'indices': test_eval_indices
        }, f, indent=2)

    print(f"适应样本结果: {adapt_result_file}")
    print(f"测试样本结果: {test_result_file}")

    return summary
def compare_centers(all_results):
    """
    比较不同中心的性能
    """
    print(f"\n{'='*70}")
    print("多中心性能比较")
    print('='*70)

    # 提取数据
    centers = list(all_results.keys())

    print(f"{'中心':<20} {'样本数':<8} {'WT Dice':<10} {'TC Dice':<10} {'ET Dice':<10} {'平均Dice':<10} {'适应':<10}")
    print("-" * 80)

    for center in centers:
        result = all_results[center]
        adapted = "是" if result['adaptation']['adapted'] else "否"

        print(f"{center:<20} {result['num_samples']:<8} "
              f"{result['metrics']['WT']['dice_mean']:<10.4f} "
              f"{result['metrics']['TC']['dice_mean']:<10.4f} "
              f"{result['metrics']['ET']['dice_mean']:<10.4f} "
              f"{result['metrics']['overall']['dice_mean']:<10.4f} "
              f"{adapted:<10}")

    # 计算所有中心的平均
    overall_wt = np.mean([r['metrics']['WT']['dice_mean'] for r in all_results.values()])
    overall_tc = np.mean([r['metrics']['TC']['dice_mean'] for r in all_results.values()])
    overall_et = np.mean([r['metrics']['ET']['dice_mean'] for r in all_results.values()])
    overall_mean = np.mean([r['metrics']['overall']['dice_mean'] for r in all_results.values()])

    print("-" * 80)
    print(f"{'所有中心平均':<20} {'-':<8} "
          f"{overall_wt:<10.4f} "
          f"{overall_tc:<10.4f} "
          f"{overall_et:<10.4f} "
          f"{overall_mean:<10.4f} "
          f"{'-':<10}")

    print('='*70)


def test(config):
    """
    主测试函数 - 所有参数从配置文件获取
    """
    print("\n" + "=" * 70)
    print("BraTS多中心MAML元测试")
    print("=" * 70)

    # 验证配置
    validate_config(config)

    # 从配置中获取checkpoint路径
    checkpoint_path = config['testing']['checkpoint']
    print(f"检查点路径: {checkpoint_path}")

    # 验证检查点文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    # 设置设备
    device = torch.device(config['hardware']['device']
                         if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 加载模型
    print("\n加载模型...")
    model = ResUNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels']
    )

    maml = FirstOrderMAML(
        model=model,
        inner_lr=config['maml']['inner_lr'],
        outer_lr=config['maml']['outer_lr'],
        inner_steps=config['maml']['inner_steps'],
        device=device
    )

    # 加载checkpoint
    checkpoint = maml.load_checkpoint(checkpoint_path)
    print(f"✓ 模型加载成功")
    print(f"  训练轮数: {checkpoint.get('epoch', 'Unknown')}")
    if 'metrics' in checkpoint:
        print(f"  训练Dice: {checkpoint['metrics'].get('dice_mean', 'Unknown'):.4f}")

    # 创建输出目录
    output_base_dir = Path(config['testing']['output_dir'])
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # 保存测试配置
    config_save_path = output_base_dir / "test_config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # 测试每个任务
    all_results = {}

    for task_name in config['testing']['test_tasks']:
        try:
            result = test_single_task(
                maml, config, task_name,
                output_base_dir
            )

            if result is not None:
                all_results[task_name] = result

        except Exception as e:
            print(f"\n❌ {task_name} 测试失败: {e}")
            import traceback
            traceback.print_exc()

    # 比较不同中心
    if len(all_results) > 1:
        compare_centers(all_results)

    # 保存总体结果
    if all_results:
        overall_results = {
            'config': config['testing'],
            'checkpoint': checkpoint_path,
            'results': all_results,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        overall_file = output_base_dir / "overall_results.json"
        with open(overall_file, 'w') as f:
            json.dump(overall_results, f, indent=2)

        print(f"\n总体结果已保存到: {overall_file}")

        # 打印关键结论
        adapted_centers = [c for c, r in all_results.items() if r['adaptation']['adapted']]
        non_adapted = [c for c, r in all_results.items() if not r['adaptation']['adapted']]

        print(f"\n📊 关键结论:")
        print(f"  1. 测试了 {len(all_results)} 个中心")
        print(f"  2. {len(adapted_centers)} 个中心使用了快速适应")
        print(f"  3. 最佳性能中心: {max(all_results.items(), key=lambda x: x[1]['metrics']['overall']['dice_mean'])[0]}")
        print(f"  4. 最差性能中心: {min(all_results.items(), key=lambda x: x[1]['metrics']['overall']['dice_mean'])[0]}")
        print(f"  5. 所有中心平均Dice: {np.mean([r['metrics']['overall']['dice_mean'] for r in all_results.values()]):.4f}")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)

    return all_results


def main():
    """主函数 - 从命令行获取配置文件路径"""
    parser = argparse.ArgumentParser(description='BraTS多中心MAML元测试')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径（默认: config.yaml）')

    args = parser.parse_args()

    # 加载配置
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        print("请创建一个配置文件或指定正确的路径")
        print("示例配置文件结构:")
        print("""
data:
  data_root: "data"
  crop_size: [224, 224, 128]

model:
  in_channels: 4
  out_channels: 3
  base_channels: 16

maml:
  inner_lr: 0.01
  outer_lr: 0.001
  inner_steps: 5

hardware:
  device: "cuda"

testing:
  checkpoint: "checkpoints/best_model.pth"
  test_tasks: ["BraTS_Center1", "BraTS_Center2"]
  output_dir: "test_results"
  enable_adaptation: true
  adaptation_k_shot: 3
  adaptation_inner_steps: 10
  inference:
    mode: "sliding_window"
    overlap: 0.5
  threshold: 0.5
  save_predictions: true
  visualization: true
  num_visualize: 5
        """)
        sys.exit(1)

    config = load_config(config_path)

    # 打印配置
    print("\n" + "=" * 70)
    print("测试配置:")
    print("=" * 70)
    print(f"配置文件: {config_path}")
    print(f"检查点: {config['testing']['checkpoint']}")
    print(f"测试任务: {config['testing']['test_tasks']}")
    print(f"输出目录: {config['testing']['output_dir']}")
    print(f"快速适应: {'启用' if config['testing']['enable_adaptation'] else '禁用'}")
    if config['testing']['enable_adaptation']:
        print(f"  适应样本数: {config['testing']['adaptation_k_shot']}")
        print(f"  适应步数: {config['testing']['adaptation_inner_steps']}")
    print(f"推理模式: {config['testing']['inference']['mode']}")
    print(f"阈值: {config['testing']['threshold']}")
    print("=" * 70)

    # 开始测试
    try:
        results = test(config)

        if results:
            print("\n✅ 测试成功完成!")
        else:
            print("\n⚠️  测试完成但无有效结果")

    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")

    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()