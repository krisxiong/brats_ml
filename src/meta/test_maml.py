#!/usr/bin/env python3
"""
测试MAML模块
"""

import sys
import torch
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入重构后的模块
from src.meta.maml import FirstOrderMAML


def test_maml_functionality():
    """测试MAML功能"""
    print("=" * 70)
    print("测试改进的MAML")
    print("=" * 70)

    try:
        # 创建模拟模型
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv3d(4, 3, kernel_size=3, padding=1)

            def forward(self, x):
                return torch.sigmoid(self.conv(x))

        model = MockModel()

        # 创建First-Order MAML
        maml = FirstOrderMAML(
            model=model,
            inner_lr=0.01,
            outer_lr=0.001,
            inner_steps=3,
            device='cpu',
            l2_reg=0.001
        )

        # 模拟任务批次
        print("\n创建测试数据...")
        task_batch = []
        for i in range(2):
            task = {
                'support_x': torch.randn(2, 4, 32, 64, 64),
                'support_y': torch.randint(0, 2, (2, 3, 32, 64, 64)).float(),
                'query_x': torch.randn(2, 4, 32, 64, 64),
                'query_y': torch.randint(0, 2, (2, 3, 32, 64, 64)).float(),
                'task_name': f'Task_{i}'
            }
            task_batch.append(task)

        # 测试训练步骤
        print("\n执行元训练步骤...")
        meta_loss, metrics = maml.meta_train_step(task_batch)

        print(f"✓ 元损失: {meta_loss:.4f}")
        print(f"✓ Query Dice: {metrics['dice_mean']:.4f}")
        print(f"✓ Support Dice: {metrics['support_dice']:.4f}")
        print(f"\n详细指标:")
        print(f"  WT: {metrics['dice_wt']:.4f}")
        print(f"  TC: {metrics['dice_tc']:.4f}")
        print(f"  ET: {metrics['dice_et']:.4f}")

        print("\n✓ First-Order MAML测试通过!")
        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    success = test_maml_functionality()

    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    if success:
        print("✅ 所有测试通过！")
        return 0
    else:
        print("❌ 测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())