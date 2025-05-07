# Utilities module
# This file marks the directory as a Python package

# Import the load_config function from config.py
from .config import load_config

# Import the get_device function from device_setup.py
from .device_setup import get_device

# Import the configured logger instance from logging.py
# The setup_logging() function in logging.py runs automatically on this import.
from .logging import logger

# Import utility functions from run_utils.py
from .run_utils import format_num_words  # Keep only one
from .run_utils import plot_metrics  # New function
from .run_utils import save_metrics  # New function

# Import the tokenizer utility functions from tokenizer_utils.py
# from .tokenizer_utils import labels_to_sequence, sequence_to_labels

# Define what gets imported with 'from utils import *'
__all__ = [
    "logger",
    "get_device",
    "load_config",
    "format_num_words",
    "save_metrics",
    "plot_metrics",
]

# Optional: Log that the package is being initialized
# Note: logger might already be configured here due to import above
logger.debug("Utils package initialized (__init__.py).")
