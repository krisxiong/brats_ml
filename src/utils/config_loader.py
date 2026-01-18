"""
配置文件加载模块
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def print_config(config: Dict[str, Any]):
    """
    打印配置信息

    Args:
        config: 配置字典
    """
    print("\n" + "=" * 70)
    print("训练配置")
    print("=" * 70)

    sections = [
        ('数据', 'data'),
        ('模型', 'model'),
        ('MAML', 'maml'),
        ('元学习', 'meta_learning'),
        ('训练', 'training'),
        ('验证', 'validation'),
        ('检查点', 'checkpoint'),
        ('日志', 'logging')
    ]

    for section_name, section_key in sections:
        if section_key in config:
            print(f"\n{section_name}:")
            _print_section(config[section_key], indent=2)

    print("=" * 70)


def _print_section(section: Dict[str, Any], indent: int = 0):
    """递归打印配置部分"""
    indent_str = " " * indent

    for key, value in section.items():
        if isinstance(value, dict):
            print(f"{indent_str}{key}:")
            _print_section(value, indent + 2)
        else:
            print(f"{indent_str}{key}: {value}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并配置（深度合并）

    Args:
        base_config: 基础配置
        override_config: 覆盖配置

    Returns:
        合并后的配置
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged