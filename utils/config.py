# Multimodal Transfer Learning / Image Captioning
# File: utils/config.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: Configuration loading functions/classes.
# Created: 2025-05-05
# Updated: 2025-05-05

import yaml
from pathlib import Path

def load_config(config_path):
    """
    Loads a YAML configuration file and returns its contents as a dictionary.
    Args:
        config_path (str or Path): Path to the YAML config file.
    Returns:
        dict: Configuration dictionary, or None if loading fails.
    """
    try:
        config_path = Path(config_path)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"[config.py] Failed to load config: {e}")
        return None

# Basic pass statement or main block for runnable scripts
if __name__ == "__main__":
    pass
