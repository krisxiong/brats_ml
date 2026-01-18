# BraTS 元学习分割项目

基于MAML的脑肿瘤3D分割项目

## 功能特性

- ✅ 支持多种元学习算法 (MAML, First-Order MAML)
- ✅ 智能数据增强和裁剪策略
- ✅ 完善的实验管理系统
- ✅ 支持测试时快速适应

## 项目结构

```
brats-meta-learning/
├── src/              # 源代码
├── configs/          # 配置文件
├── scripts/          # 运行脚本
├── data/             # 数据目录
└── outputs/          # 输出目录
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将BraTS数据放入 `data/processed/` 目录:

```
data/processed/
├── BraTS_Task1/
│   └── train/
│       ├── patient_001/
│       ├── patient_002/
│       └── ...
└── BraTS_Task2/
    └── train/
```

### 3. 训练模型

```bash
# 使用默认配置
python scripts/train.py --config configs/maml.yaml

# 调试模式(快速验证)
python scripts/train.py --config configs/debug.yaml
```

### 4. 测试模型

```bash
python scripts/test.py \
    --config configs/maml.yaml \
    --checkpoint outputs/checkpoints/best_model.pth
```

## 配置说明

主要配置文件在 `configs/` 目录:

- `base.yaml` - 基础配置
- `maml.yaml` - MAML训练配置
- `debug.yaml` - 调试配置(小数据集)

## 许可证

MIT License
