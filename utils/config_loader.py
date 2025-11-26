"""
配置文件加载工具
用于从UI传递参数到训练脚本
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def load_training_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载训练配置

    Parameters:
    -----------
    config_path : str, optional
        配置文件路径。如果为None，则从命令行参数读取

    Returns:
    --------
    config : dict
        配置字典，包含 model_name 和 model_params
    """
    # 如果没有指定配置路径，尝试从命令行参数读取
    if config_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default=None,
                          help='Path to training config file')
        args, _ = parser.parse_known_args()
        config_path = args.config

    # 如果仍然没有配置文件，返回空配置
    if config_path is None:
        print("[配置] 未指定配置文件，使用默认参数")
        return {'model_name': 'Unknown', 'model_params': {}}

    config_file = Path(config_path)

    if not config_file.exists():
        print(f"[配置] 配置文件不存在: {config_path}，使用默认参数")
        return {'model_name': 'Unknown', 'model_params': {}}

    # 读取配置文件
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print(f"[配置] 成功加载配置文件: {config_path}")
        print(f"[配置] 模型: {config.get('model_name', 'Unknown')}")

        if 'model_params' in config and config['model_params']:
            print(f"[配置] 用户自定义参数:")
            for key, value in config['model_params'].items():
                print(f"  - {key}: {value}")
        else:
            print(f"[配置] 使用默认参数")

        return config

    except Exception as e:
        print(f"[配置] 读取配置文件失败: {e}，使用默认参数")
        return {'model_name': 'Unknown', 'model_params': {}}


def get_param(config: Dict[str, Any], param_name: str, default_value: Any) -> Any:
    """
    从配置中获取参数值

    Parameters:
    -----------
    config : dict
        配置字典
    param_name : str
        参数名称
    default_value : Any
        默认值

    Returns:
    --------
    value : Any
        参数值（如果配置中存在）或默认值
    """
    if 'model_params' in config and param_name in config['model_params']:
        return config['model_params'][param_name]
    return default_value


def update_params_from_config(config: Dict[str, Any], default_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    用配置文件中的参数更新默认参数

    Parameters:
    -----------
    config : dict
        配置字典
    default_params : dict
        默认参数字典

    Returns:
    --------
    updated_params : dict
        更新后的参数字典
    """
    params = default_params.copy()

    if 'model_params' in config and config['model_params']:
        for key, value in config['model_params'].items():
            if key in params:
                params[key] = value
                print(f"[参数更新] {key}: {params[key]} (来自配置)")

    return params
