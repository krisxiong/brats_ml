"""
实用工具函数
包含数据预处理、可视化、评估等功能
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import json


# ============ 诊断工具 ============

def diagnose_labels():
    """
    诊断标签处理是否正确
    这是解决WT Dice过低问题的关键
    """
    print("=" * 70)
    print("标签诊断工具")
    print("=" * 70)

    try:
        from dataloader import BraTSDataset

        dataset = BraTSDataset(
            data_root='data',
            task_name='BraTS_Task1',
            mode='train',
            crop_size=(160, 160, 128)
        )

        # 加载多个样本检查
        print(f"\n检查 {min(5, len(dataset))} 个样本...")

        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            label = sample['label'].numpy()  # [3, D, H, W]

            wt = label[0]  # 整个肿瘤
            tc = label[1]  # 肿瘤核心
            et = label[2]  # 增强肿瘤

            # 计算非零像素数
            wt_count = (wt > 0).sum()
            tc_count = (tc > 0).sum()
            et_count = (et > 0).sum()

            print(f"\n样本 {i}:")
            print(f"  WT 非零像素: {wt_count:,}")
            print(f"  TC 非零像素: {tc_count:,}")
            print(f"  ET 非零像素: {et_count:,}")

            # 验证关系
            wt_ge_tc = wt_count >= tc_count
            tc_ge_et = tc_count >= et_count

            if wt_ge_tc and tc_ge_et:
                print(f"  ✓ 关系正确: WT >= TC >= ET")
            else:
                print(f"  ✗ 关系错误!")
                print(f"    WT >= TC? {wt_ge_tc}")
                print(f"    TC >= ET? {tc_ge_et}")

            # 检查是否有重叠
            # WT应该包含TC，TC应该包含ET
            tc_in_wt = ((tc > 0) & (wt > 0)).sum() == tc_count
            et_in_tc = ((et > 0) & (tc > 0)).sum() == et_count

            if tc_in_wt:
                print(f"  ✓ TC完全在WT内")
            else:
                print(f"  ✗ TC不在WT内！")

            if et_in_tc:
                print(f"  ✓ ET完全在TC内")
            else:
                print(f"  ✗ ET不在TC内！")

        print("\n" + "=" * 70)
        print("诊断建议:")
        print("如果看到'关系错误'或'不在内'，说明标签处理有问题")
        print("这会导致WT的Dice异常低！")
        print("=" * 70)

    except Exception as e:
        print(f"诊断失败: {e}")
        import traceback
        traceback.print_exc()


# ============ 数据预处理工具 ============

def check_data_structure(data_root):
    """
    检查数据目录结构是否正确

    参数:
        data_root: 数据根目录

    返回:
        report: 检查报告
    """
    data_root = Path(data_root)
    report = {
        'valid': True,
        'tasks': {},
        'warnings': []
    }

    if not data_root.exists():
        report['valid'] = False
        report['warnings'].append(f"数据目录不存在: {data_root}")
        return report

    # 检查每个任务
    for task_dir in data_root.iterdir():
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name
        task_info = {
            'train': 0,
            'test': 0,
            'valid': True
        }

        # 检查train和test目录
        for split in ['train', 'test']:
            split_dir = task_dir / split
            if split_dir.exists():
                patients = [d for d in split_dir.iterdir() if d.is_dir()]
                task_info[split] = len(patients)

                # 检查第一个患者的文件
                if patients:
                    patient_dir = patients[0]
                    required_files = ['t1', 't1ce', 't2', 'flair', 'seg']

                    for file_type in required_files:
                        # 尝试不同的文件命名
                        found = False
                        for pattern in [f"*{file_type}*.nii.gz", f"{file_type}.nii.gz"]:
                            if list(patient_dir.glob(pattern)):
                                found = True
                                break

                        if not found and file_type != 'seg':
                            task_info['valid'] = False
                            report['warnings'].append(
                                f"任务 {task_name}/{split}: 缺少 {file_type} 文件"
                            )

        report['tasks'][task_name] = task_info

        if not task_info['valid']:
            report['valid'] = False

    return report


def print_data_report(report):
    """
    打印数据检查报告
    """
    print("=" * 60)
    print("数据结构检查报告")
    print("=" * 60)

    if report['valid']:
        print("✓ 数据结构有效")
    else:
        print("✗ 数据结构有问题")

    print("\n任务统计:")
    for task_name, task_info in report['tasks'].items():
        status = "✓" if task_info['valid'] else "✗"
        print(f"\n{status} {task_name}:")
        print(f"    训练集: {task_info['train']} 个患者")
        print(f"    测试集: {task_info['test']} 个患者")

    if report['warnings']:
        print("\n⚠️  警告:")
        for warning in report['warnings']:
            print(f"    - {warning}")

    print("=" * 60)


# ============ 可视化工具 ============

def visualize_sample(image, label, save_path=None):
    """
    可视化单个样本

    参数:
        image: [4, D, H, W] - 四个模态
        label: [3, D, H, W] - 三个分割目标
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # 选择中间切片
    slice_idx = image.shape[1] // 2

    # 显示4个模态
    modalities = ['t1n', 't1c', 't2w', 't2f']
    for i, mod in enumerate(modalities):
        axes[0, i].imshow(image[i, slice_idx], cmap='gray')
        axes[0, i].set_title(mod)
        axes[0, i].axis('off')

    # 显示3个分割目标
    targets = ['WT', 'TC', 'ET']
    for i, target in enumerate(targets):
        axes[1, i].imshow(label[i, slice_idx], cmap='jet')
        axes[1, i].set_title(f'{target} Label')
        axes[1, i].axis('off')

    # 显示合并的标签
    combined = label.argmax(axis=0)[slice_idx]
    axes[1, 3].imshow(combined, cmap='jet')
    axes[1, 3].set_title('Combined')
    axes[1, 3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(history_file, save_path=None):
    """
    绘制训练历史曲线

    参数:
        history_file: 训练历史JSON文件路径
        save_path: 保存路径
    """
    with open(history_file, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice曲线
    axes[1].plot(history['train_dice'], label='Train Dice', linewidth=2)
    if 'val_dice' in history:
        axes[1].plot(history['val_dice'], label='Val Dice', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Score', fontsize=12)
    axes[1].set_title('Dice Score', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def compare_predictions(image, gt, pred1, pred2, names=['Model 1', 'Model 2'],
                       save_path=None):
    """
    比较两个模型的预测结果
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    slice_idx = image.shape[1] // 2

    # 第一行: 图像和真实标签
    axes[0, 0].imshow(image[0, slice_idx], cmap='gray')
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt.argmax(axis=0)[slice_idx], cmap='jet')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')

    axes[0, 2].axis('off')

    # 第二行: 两个预测
    axes[1, 0].imshow(pred1.argmax(axis=0)[slice_idx], cmap='jet')
    axes[1, 0].set_title(names[0])
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pred2.argmax(axis=0)[slice_idx], cmap='jet')
    axes[1, 1].set_title(names[1])
    axes[1, 1].axis('off')

    # 差异图
    diff = (pred1.argmax(axis=0) != pred2.argmax(axis=0)).astype(float)
    axes[1, 2].imshow(diff[slice_idx], cmap='hot')
    axes[1, 2].set_title('Difference')
    axes[1, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


# ============ 评估工具 ============

def calculate_detailed_metrics(pred, target):
    """
    计算详细的评估指标

    返回:
        metrics: 包含多个指标的字典
    """
    pred_binary = (pred > 0.5).float()

    metrics = {}

    # 对每个类别计算
    for c in range(pred.shape[0]):
        pred_c = pred_binary[c]
        target_c = target[c]

        # True Positives, False Positives, False Negatives
        tp = (pred_c * target_c).sum().item()
        fp = (pred_c * (1 - target_c)).sum().item()
        fn = ((1 - pred_c) * target_c).sum().item()
        tn = ((1 - pred_c) * (1 - target_c)).sum().item()

        # Dice Score
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

        # IoU (Jaccard)
        iou = tp / (tp + fp + fn + 1e-8)

        # Sensitivity (Recall)
        sensitivity = tp / (tp + fn + 1e-8)

        # Specificity
        specificity = tn / (tn + fp + 1e-8)

        # Precision
        precision = tp / (tp + fp + 1e-8)

        # F1 Score
        f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)

        metrics[f'class_{c}'] = {
            'dice': dice,
            'iou': iou,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1
        }

    return metrics


def save_metrics_report(metrics_list, class_names, save_path):
    """
    保存详细的评估报告
    """
    # 计算平均值
    avg_metrics = {}
    for class_idx, class_name in enumerate(class_names):
        class_key = f'class_{class_idx}'
        avg_metrics[class_name] = {}

        for metric_name in ['dice', 'iou', 'sensitivity', 'specificity',
                           'precision', 'f1']:
            values = [m[class_key][metric_name] for m in metrics_list]
            avg_metrics[class_name][metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

    # 保存为JSON
    with open(save_path, 'w') as f:
        json.dump(avg_metrics, f, indent=2)

    # 打印报告
    print("\n" + "=" * 60)
    print("详细评估报告")
    print("=" * 60)

    for class_name, metrics in avg_metrics.items():
        print(f"\n{class_name}:")
        for metric_name, stats in metrics.items():
            print(f"  {metric_name:12s}: "
                  f"{stats['mean']:.4f} ± {stats['std']:.4f} "
                  f"(范围: {stats['min']:.4f} - {stats['max']:.4f})")

    print("=" * 60)
    print(f"报告已保存到: {save_path}")


# ============ 模型分析工具 ============

def count_parameters(model):
    """
    统计模型参数量
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型参数统计:")
    print(f"  总参数: {total:,}")
    print(f"  可训练参数: {trainable:,}")
    print(f"  不可训练参数: {total - trainable:,}")

    # 计算模型大小(MB)
    param_size = sum(p.numel() * p.element_size()
                    for p in model.parameters()) / (1024 ** 2)
    buffer_size = sum(b.numel() * b.element_size()
                     for b in model.buffers()) / (1024 ** 2)

    print(f"  参数大小: {param_size:.2f} MB")
    print(f"  缓冲区大小: {buffer_size:.2f} MB")
    print(f"  总大小: {param_size + buffer_size:.2f} MB")

    return {
        'total': total,
        'trainable': trainable,
        'size_mb': param_size + buffer_size
    }


def analyze_model_layers(model):
    """
    分析模型各层的参数量
    """
    print("\n各层参数统计:")
    print("-" * 60)
    print(f"{'层名称':<40} {'参数量':>15}")
    print("-" * 60)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:<40} {param.numel():>15,}")

    print("-" * 60)


# ============ 主函数 ============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='实用工具脚本')
    parser.add_argument('--action', type=str, required=True,
                      choices=['check_data', 'visualize', 'analyze_model', 'diagnose_labels'],
                      help='要执行的操作')
    parser.add_argument('--data_root', type=str, default='data',
                      help='数据根目录')
    parser.add_argument('--checkpoint', type=str,
                      help='模型检查点路径')

    args = parser.parse_args()

    if args.action == 'check_data':
        # 检查数据结构
        report = check_data_structure(args.data_root)
        print_data_report(report)

    elif args.action == 'diagnose_labels':
        # 诊断标签问题 - 新增
        diagnose_labels()

    elif args.action == 'analyze_model':
        # 分析模型
        if not args.checkpoint:
            print("错误: 需要提供 --checkpoint 参数")
        else:
            from model import ResUNet

            model = ResUNet(in_channels=4, out_channels=3,
                          base_channels=32)

            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])

            count_parameters(model)
            analyze_model_layers(model)

    else:
        print(f"操作 {args.action} 尚未实现")