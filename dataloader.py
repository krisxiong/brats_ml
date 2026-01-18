"""
dataloader.py - BraTS 3D数据加载器
支持智能随机crop策略 + 真正的元学习任务采样
"""
import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import random
from pathlib import Path
from scipy import ndimage
import json


class BraTSDataset(Dataset):
    """
    BraTS 3D数据集
    关键特性：
    1. 智能随机crop（确保包含肿瘤但位置随机）
    2. 支持大crop尺寸（224×224×128）
    3. 不使用任何resize操作
    4. 保护医学影像完整性
    """
    def __init__(self, data_root, task_name, mode='train',
                 crop_size=(224, 224, 128),
                 crop_strategy='hybrid',  # 'hybrid', 'smart_random', 'random'
                 normalize=True,
                 augment_type='none'):
        """
        参数:
            crop_size: 裁剪大小，推荐 (224, 224, 128)
            crop_strategy:
                - 'hybrid': 70%智能crop + 30%随机crop（推荐）
                - 'smart_random': 100%智能crop
                - 'random': 100%随机crop
            augment_type:
                - 'none': 不增强（inner-loop用）
                - 'domain': domain shift增强（outer-loop用）
        """
        self.data_root = Path(data_root)
        self.task_name = task_name
        self.mode = mode
        self.crop_size = crop_size
        self.crop_strategy = crop_strategy
        self.normalize = normalize
        self.augment_type = augment_type

        # 加载任务路径和元数据
        self.task_path = self.data_root / task_name / mode
        self.metadata = self._load_metadata()

        # 获取所有患者ID
        self.patient_ids = self._get_patient_ids()

        if not self.patient_ids:
            raise ValueError(f"在 {self.task_path} 下没有找到任何患者数据")

        print(f"\n{'='*60}")
        print(f"任务: {task_name} ({mode})")
        print(f"{'='*60}")
        print(f"  患者数量: {len(self.patient_ids)}")
        print(f"  裁剪大小: {crop_size}")
        print(f"  裁剪策略: {crop_strategy}")
        print(f"  增强类型: {augment_type}")
        if self.metadata:
            print(f"  中心: {self.metadata.get('center', 'Unknown')}")
            print(f"  扫描仪: {self.metadata.get('scanner', 'Unknown')}")
        print('='*60)

    def _load_metadata(self):
        """加载任务元数据"""
        metadata_file = self.data_root / self.task_name / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _get_patient_ids(self):
        """获取所有患者ID"""
        patient_ids = []
        if self.task_path.exists():
            patient_ids = [d.name for d in self.task_path.iterdir()
                          if d.is_dir()]
        return sorted(patient_ids)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        """加载数据并记录原始尺寸"""
        patient_id = self.patient_ids[idx]
        patient_path = self.task_path / patient_id

        # 加载模态和标签
        modalities = ['t1n', 't1c', 't2w', 't2f']
        images = []
        for modality in modalities:
            img = self._load_modality(patient_path, patient_id, modality)
            images.append(img)

        label = self._load_label(patient_path, patient_id)

        # ⚠️ 记录原始尺寸
        original_shape = images[0].shape

        images = np.stack(images, axis=0)

        if self.normalize:
            images = self._normalize(images)

        label = self._process_label(label)

        # Crop并记录位置
        images, label, crop_info = self._apply_crop_strategy(
            images, label, self.crop_size
        )

        # ⚠️ 添加原始尺寸信息
        crop_info['original_shape'] = original_shape

        if self.augment_type == 'domain':
            images = self._domain_augmentation(images)

        return {
            'image': torch.FloatTensor(images),
            'label': torch.FloatTensor(label),
            'patient_id': patient_id,
            'crop_info': crop_info,
            'metadata': self.metadata,
            'original_shape': original_shape  # ⚠️ 新增
        }

    def _apply_crop_strategy(self, images, label, crop_size):
        """
        应用crop策略
        """
        if self.crop_strategy == 'hybrid':
            # 70%智能crop，30%随机crop
            if random.random() < 0.7:
                return self._smart_random_crop(images, label, crop_size)
            else:
                return self._pure_random_crop(images, label, crop_size)

        elif self.crop_strategy == 'smart_random':
            return self._smart_random_crop(images, label, crop_size)

        elif self.crop_strategy == 'random':
            return self._pure_random_crop(images, label, crop_size)

        else:
            raise ValueError(f"Unknown crop strategy: {self.crop_strategy}")

    def _smart_random_crop(self, images, label, crop_size):
        """
        ============ 智能随机Crop ============
        关键：确保crop包含肿瘤，但肿瘤位置是随机的

        这避免了模型学习"肿瘤总在中心"的虚假模式
        """
        current_size = images.shape[1:]

        # 找到肿瘤区域
        tumor_mask = (label.sum(axis=0) > 0)
        tumor_coords = np.where(tumor_mask)

        has_tumor = len(tumor_coords[0]) > 0

        if not has_tumor:
            # 没有肿瘤，使用随机crop
            return self._pure_random_crop(images, label, crop_size)

        # 计算肿瘤bounding box
        tumor_min = [int(coords.min()) for coords in tumor_coords]
        tumor_max = [int(coords.max()) for coords in tumor_coords]

        # 计算crop起始位置的有效范围
        # 确保crop与肿瘤有交集，但肿瘤可以在crop内的任意位置
        crop_starts = []

        for dim in range(3):
            # crop的起始位置范围：
            # 最小：肿瘤右边界 - crop_size（肿瘤在crop右侧）
            # 最大：肿瘤左边界（肿瘤在crop左侧）
            min_start = max(0, tumor_max[dim] - crop_size[dim])
            max_start = min(current_size[dim] - crop_size[dim], tumor_min[dim])

            # 确保范围有效
            if max_start < min_start:
                # 肿瘤太大或接近边界，居中处理
                start = max(0, (tumor_min[dim] + tumor_max[dim] - crop_size[dim]) // 2)
                start = min(start, current_size[dim] - crop_size[dim])
            else:
                # 在有效范围内随机选择
                start = random.randint(min_start, max_start)

            crop_starts.append(start)

        # 执行crop
        images_cropped = images[:,
                                crop_starts[0]:crop_starts[0]+crop_size[0],
                                crop_starts[1]:crop_starts[1]+crop_size[1],
                                crop_starts[2]:crop_starts[2]+crop_size[2]]

        label_cropped = label[:,
                              crop_starts[0]:crop_starts[0]+crop_size[0],
                              crop_starts[1]:crop_starts[1]+crop_size[1],
                              crop_starts[2]:crop_starts[2]+crop_size[2]]

        # Padding（如果需要）
        if images_cropped.shape[1:] != crop_size:
            images_cropped = self._pad_to_size(images_cropped, crop_size)
            label_cropped = self._pad_to_size(label_cropped, crop_size)

        crop_info = {
            'strategy': 'smart_random',
            'has_tumor': True,
            'tumor_bbox': (tumor_min, tumor_max),
            'crop_start': crop_starts
        }

        return images_cropped, label_cropped, crop_info

    def _pure_random_crop(self, images, label, crop_size):
        """纯随机crop"""
        current_size = images.shape[1:]

        # 随机选择起始位置
        crop_starts = []
        for dim in range(3):
            max_start = max(0, current_size[dim] - crop_size[dim])
            start = random.randint(0, max_start) if max_start > 0 else 0
            crop_starts.append(start)

        # 执行crop
        images_cropped = images[:,
                                crop_starts[0]:crop_starts[0]+crop_size[0],
                                crop_starts[1]:crop_starts[1]+crop_size[1],
                                crop_starts[2]:crop_starts[2]+crop_size[2]]

        label_cropped = label[:,
                              crop_starts[0]:crop_starts[0]+crop_size[0],
                              crop_starts[1]:crop_starts[1]+crop_size[1],
                              crop_starts[2]:crop_starts[2]+crop_size[2]]

        # Padding
        if images_cropped.shape[1:] != crop_size:
            images_cropped = self._pad_to_size(images_cropped, crop_size)
            label_cropped = self._pad_to_size(label_cropped, crop_size)

        crop_info = {
            'strategy': 'random',
            'has_tumor': (label_cropped.sum() > 0),
            'crop_start': crop_starts
        }

        return images_cropped, label_cropped, crop_info

    def _pad_to_size(self, data, target_size):
        """Padding到目标大小"""
        current_size = data.shape[1:]
        pad_width = [
            (0, 0),
            (0, max(0, target_size[0] - current_size[0])),
            (0, max(0, target_size[1] - current_size[1])),
            (0, max(0, target_size[2] - current_size[2]))
        ]
        return np.pad(data, pad_width, mode='constant', constant_values=0)

    def _domain_augmentation(self, images):
        """Domain shift增强（模拟不同扫描仪）"""
        # 亮度变化
        brightness = random.uniform(0.85, 1.15)
        images = images * brightness

        # 对比度变化
        contrast = random.uniform(0.9, 1.1)
        mean_val = images.mean()
        images = (images - mean_val) * contrast + mean_val

        # 噪声
        if random.random() > 0.5:
            noise_std = random.uniform(0, 0.03)
            noise = np.random.normal(0, noise_std, images.shape)
            images = images + noise

        return images

    def _load_modality(self, patient_path, patient_id, modality):
        """加载单个模态"""
        possible_names = [
            f"{patient_id}-{modality}.nii.gz",
            f"{patient_id}-{modality}.nii",
            f"{patient_id}_{modality}.nii.gz",
            f"{modality}.nii.gz",
        ]

        for name in possible_names:
            img_file = patient_path / name
            if img_file.exists():
                return nib.load(str(img_file)).get_fdata()

        raise FileNotFoundError(
            f"找不到 {patient_id}/{modality}\n尝试了: {possible_names}"
        )

    def _load_label(self, patient_path, patient_id):
        """加载标签"""
        possible_names = [
            f"{patient_id}-seg.nii.gz",
            f"{patient_id}-seg.nii",
            f"{patient_id}_seg.nii.gz",
            f"seg.nii.gz",
        ]

        for name in possible_names:
            seg_file = patient_path / name
            if seg_file.exists():
                return nib.load(str(seg_file)).get_fdata()

        if self.mode == 'test':
            # 测试集可能没有标签
            return np.zeros((155, 240, 240))

        raise FileNotFoundError(f"找不到 {patient_id} 的标签")

    def _normalize(self, images):
        """Z-score归一化"""
        for i in range(len(images)):
            img = images[i]
            mask = img > 0
            if mask.any():
                mean = img[mask].mean()
                std = img[mask].std()
                images[i] = np.where(mask, (img - mean) / (std + 1e-8), 0)
        return images

    def _process_label(self, label):
        """
        处理 BraTS 2024 标签
        输出 3 类: WT, TC, ET
        保证层次关系: WT ⊇ TC ⊇ ET
        """

        label = label.astype(np.int32)

        # BraTS 2024 标签
        ncr = (label == 1).astype(np.float32)  # 坏死
        ed = (label == 2).astype(np.float32)  # 水肿
        cc = (label == 3).astype(np.float32)  # Core Component (新增)
        et = (label == 4).astype(np.float32)  # 增强

        # Tumor Core = NCR + CC + ET
        tc = np.logical_or.reduce([ncr, cc, et]).astype(np.float32)

        # Whole Tumor = TC + ED
        wt = np.logical_or(tc, ed).astype(np.float32)

        return np.stack([wt, tc, et], axis=0)


class MetaTaskSampler:
    """
    真正的元学习任务采样器
    任务 = 不同的医疗中心/扫描仪/协议
    """
    def __init__(self, data_root, k_shot=2, k_query=2,
                 crop_size=(224, 224, 128),
                 crop_strategy='hybrid'):
        """
        参数:
            data_root: 数据根目录
            k_shot: support样本数
            k_query: query样本数
        """
        self.data_root = Path(data_root)
        self.k_shot = k_shot
        self.k_query = k_query
        self.crop_size = crop_size
        self.crop_strategy = crop_strategy

        # 发现所有任务
        self.tasks = self._discover_tasks()

        # 为每个任务创建数据集
        self.datasets = {}
        for task_name in self.tasks:
            try:
                # Support: 不增强（inner-loop）
                support_ds = BraTSDataset(
                    data_root=data_root,
                    task_name=task_name,
                    mode='train',
                    crop_size=crop_size,
                    crop_strategy=crop_strategy,
                    augment_type='none'
                )

                # Query: domain增强（outer-loop）
                query_ds = BraTSDataset(
                    data_root=data_root,
                    task_name=task_name,
                    mode='train',
                    crop_size=crop_size,
                    crop_strategy=crop_strategy,
                    augment_type='domain'
                )

                if len(support_ds) > 0:
                    self.datasets[task_name] = {
                        'support': support_ds,
                        'query': query_ds
                    }

            except Exception as e:
                print(f"⚠️  跳过任务 {task_name}: {e}")

        if not self.datasets:
            raise ValueError("没有成功加载任何任务！")

        print(f"\n{'='*60}")
        print("元学习任务采样器")
        print('='*60)
        print(f"  任务数量: {len(self.datasets)}")
        print(f"  K-shot: {k_shot}, K-query: {k_query}")
        print(f"  任务列表: {list(self.datasets.keys())}")
        print('='*60)

    def _discover_tasks(self):
        """发现所有任务"""
        tasks = []
        for task_dir in self.data_root.iterdir():
            if task_dir.is_dir() and (task_dir / 'train').exists():
                tasks.append(task_dir.name)

        if not tasks:
            raise ValueError(
                f"在 {self.data_root} 下没有找到任何任务！\n"
                f"期望结构：\n"
                f"  data/\n"
                f"  ├── Center_A/train/\n"
                f"  └── Center_B/train/"
            )

        return sorted(tasks)

    def sample_task(self):
        """采样一个任务"""
        # 随机选择一个任务
        task_name = random.choice(list(self.datasets.keys()))
        support_ds = self.datasets[task_name]['support']
        query_ds = self.datasets[task_name]['query']

        k_total = self.k_shot + self.k_query

        # 采样患者索引
        if len(support_ds) < k_total:
            indices = random.choices(range(len(support_ds)), k=k_total)
        else:
            indices = random.sample(range(len(support_ds)), k=k_total)

        # 分配给support和query
        support_indices = indices[:self.k_shot]
        query_indices = indices[self.k_shot:]

        # 加载数据
        support_samples = [support_ds[i] for i in support_indices]
        query_samples = [query_ds[i] for i in query_indices]

        # 堆叠为batch
        support_x = torch.stack([s['image'] for s in support_samples], dim=0)
        support_y = torch.stack([s['label'] for s in support_samples], dim=0)
        query_x = torch.stack([s['image'] for s in query_samples], dim=0)
        query_y = torch.stack([s['label'] for s in query_samples], dim=0)

        return {
            'support_x': support_x,
            'support_y': support_y,
            'query_x': query_x,
            'query_y': query_y,
            'task_name': task_name,
            'task_metadata': support_ds.metadata
        }

    def create_batch(self, batch_size):
        """创建任务批次"""
        return [self.sample_task() for _ in range(batch_size)]


# ============ 测试代码 ============
if __name__ == "__main__":
    print("=" * 70)
    print("测试数据加载器")
    print("=" * 70)

    # 测试单个数据集
    try:
        dataset = BraTSDataset(
            data_root='data',
            task_name='BraTS_Task1',
            mode='train',
            crop_size=(224, 224, 128),
            crop_strategy='hybrid'
        )

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n✓ 单个数据集测试通过")
            print(f"  图像: {sample['image'].shape}")
            print(f"  标签: {sample['label'].shape}")
            print(f"  Crop策略: {sample['crop_info']['strategy']}")
            print(f"  包含肿瘤: {sample['crop_info']['has_tumor']}")

    except Exception as e:
        print(f"✗ 单个数据集测试失败: {e}")

    # 测试元任务采样器
    try:
        sampler = MetaTaskSampler(
            data_root='data',
            k_shot=2,
            k_query=2,
            crop_size=(224, 224, 128),
            crop_strategy='hybrid'
        )

        task = sampler.sample_task()
        print(f"\n✓ 元任务采样器测试通过")
        print(f"  任务: {task['task_name']}")
        print(f"  Support: {task['support_x'].shape}")
        print(f"  Query: {task['query_x'].shape}")

    except Exception as e:
        print(f"✗ 元任务采样器测试失败: {e}")

    print("\n" + "=" * 70)