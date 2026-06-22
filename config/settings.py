import argparse
from core.utils.util import read_config


def load_config():
    """加载配置文件"""
    parser = argparse.ArgumentParser(description="Server configuration")
    parser.add_argument("--config_path", type=str, default="config.yaml")
    args = parser.parse_args()
    return read_config(args.config_path)
