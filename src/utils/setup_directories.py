
import os
def setup_directories(config):
    """创建必要的目录"""
    dirs = [
        config['checkpoint']['save_dir'],
        config['logging']['log_dir'],
        config['testing']['output_dir']
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
