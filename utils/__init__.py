# Multimodal Transfer Learning / Image Captioning
# File: utils/__init__.py
# Description: Package initialization for utils module.
# Created: 2025-05-06
# Updated: 2025-05-06

# Import the configured logger instance from logging.py
# The setup_logging() function in logging.py runs automatically on this import.
from .logging import logger

# Import the get_device function from device_setup.py
from .device_setup import get_device

# Import the load_config function from config.py
from .config import load_config

# Import utility functions from run_utils.py
from .run_utils import (
    format_num_words, # Keep only one
    save_metrics,     # New function
    plot_metrics      # New function
)

# Import the tokenizer utility functions from tokenizer_utils.py
# from .tokenizer_utils import labels_to_sequence, sequence_to_labels

# Define what gets imported with 'from utils import *'
__all__ = ['logger', 'get_device', 'load_config',
           'format_num_words', 'save_metrics', 'plot_metrics']

# Optional: Log that the package is being initialized
# Note: logger might already be configured here due to import above
logger.debug("Utils package initialized (__init__.py).")