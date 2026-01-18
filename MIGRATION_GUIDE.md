# 代码迁移指南

本文档指导如何将原始代码迁移到新结构

## 迁移步骤

### 1. dataloader.py → src/data/

- `BraTSDataset` 类 → `src/data/dataset.py`
- `MetaTaskSampler` 类 → `src/data/sampler.py`
- 裁剪函数 → `src/utils/crop.py`
- 归一化函数 → `src/data/preprocessing.py`

### 2. model.py → src/models/

- 整个文件 → `src/models/resunet.py`

### 3. maml.py → src/meta_learning/

- `FirstOrderMAML` 类 → `src/meta_learning/fomaml.py`
- `PrototypicalNetwork` 类 → `src/meta_learning/protonet.py` (可选)

### 4. train.py → scripts/

- 主要逻辑 → `scripts/train.py`
- 训练函数 → `src/training/trainer.py` (可选)

### 5. utils.py → src/utils/

- 可视化函数 → `src/utils/visualization.py`
- 其他工具函数 → `src/utils/` 下相应文件

### 6. config.yaml

- 保持不变,放在 `configs/` 目录

## 修改import语句

原始:
```python
from dataloader import BraTSDataset
from model import ResUNet
from maml import FirstOrderMAML
```

新的:
```python
from src.data.dataset import BraTSDataset
from src.models.resunet import ResUNet
from src.meta_learning.fomaml import FirstOrderMAML
```

## 快速迁移命令

```bash
# 1. 复制dataset
cp dataloader.py src/data/dataset_temp.py
# 手动提取BraTSDataset类到src/data/dataset.py

# 2. 复制模型
cp model.py src/models/resunet.py

# 3. 复制MAML
cp maml.py src/meta_learning/fomaml.py

# 4. 复制训练脚本
cp train.py scripts/train.py

# 5. 修改所有import语句
```

## 注意事项

1. 所有相对import需要改为绝对import
2. 添加必要的 `__init__.py` 文件
3. 测试每个模块是否可以独立导入
4. 更新配置文件路径

## 验证迁移

```bash
# 测试import
python -c "from src.data.dataset import BraTSDataset; print('OK')"
python -c "from src.models.resunet import ResUNet; print('OK')"
python -c "from src.meta_learning.fomaml import FirstOrderMAML; print('OK')"

# 运行调试配置
python scripts/train.py --config configs/debug.yaml
```
