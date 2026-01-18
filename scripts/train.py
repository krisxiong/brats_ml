#!/usr/bin/env python3
"""
MAML训练脚本 - 规范化版本
支持配置文件、验证、早停、checkpoint保存
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from src.utils.config_loader import load_config, print_config
from src.utils.setup_directories import setup_directories

from src.training.meta_trainer import MAMLTrainer
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MAML训练脚本')

    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='配置文件路径 (默认: configs/default.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的checkpoint路径')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录 (覆盖配置文件中的设置)')
    parser.add_argument('--data-root', type=str, default=None,
                       help='数据根目录 (覆盖配置文件中的设置)')
    parser.add_argument('--device', type=str, default=None,
                       help='训练设备 (如: cuda:0, cpu)')

    return parser.parse_args()


def update_config_from_args(config, args):
    """根据命令行参数更新配置"""
    if args.output_dir:
        config['checkpoint']['save_dir'] = args.output_dir
        config['logging']['log_dir'] = args.output_dir
        config['testing']['output_dir'] = args.output_dir

    if args.data_root:
        config['data']['data_root'] = args.data_root

    return config


def main():
    """主函数"""
    args = parse_args()

    print("=" * 70)
    print("BraTS MAML训练脚本")
    print("=" * 70)

    try:
        # 加载配置
        print(f"\n加载配置文件: {args.config}")
        config = load_config(args.config)

        # 根据命令行参数更新配置
        config = update_config_from_args(config, args)

        # 打印配置
        print_config(config)

        # 创建目录
        setup_directories(config)

        # 创建训练器
        trainer = MAMLTrainer(config, device=args.device)

        # 恢复训练（如果指定）
        if args.resume:
            print(f"\n恢复训练: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # 开始训练
        print("\n" + "=" * 70)
        print("开始训练...")
        print("=" * 70)

        history = trainer.train()

        # 训练完成
        print("\n" + "=" * 70)
        print("✅ 训练成功完成!")
        print("=" * 70)

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        return 130  # SIGINT退出码

    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())