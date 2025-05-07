# Multimodal Transfer Learning / Image Captioning
# File: utils/logging.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: Logging setup for the project with colored console output.
# Created: 2025-05-05
# Updated: 2025-05-06

import logging
import multiprocessing
import os
import sys
from logging.handlers import RotatingFileHandler

# --- 👇 Import coloredlogs ---
try:
    import coloredlogs

    COLOREDLOGS_AVAILABLE = True
except ImportError:
    COLOREDLOGS_AVAILABLE = False
    print("⚠️ 'coloredlogs' library not found.")
    print("   Console output will not be colored.")
    print("   Install with: pip install coloredlogs")
# --- End import ---

# --- Config ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FILE_ENABLED = os.environ.get("LOG_FILE_ENABLED", "true").lower() in (
    "true",
    "yes",
    "1",
)
LOG_CONSOLE_ENABLED = os.environ.get(
    "LOG_CONSOLE_ENABLED", "true"
).lower() in ("true", "yes", "1")
LOGS_DIR = os.environ.get("LOGS_DIR", "logs")
LOG_FILE_NAME = os.environ.get("LOG_FILE_NAME", "mnist_vit_train.log")
LOG_MAX_BYTES = int(os.environ.get("LOG_MAX_BYTES", 10 * 1024 * 1024))
LOG_BACKUP_COUNT = int(os.environ.get("LOG_BACKUP_COUNT", 5))
LOG_FORMAT = os.environ.get(
    "LOG_FORMAT",
    "%(asctime)s | %(name)s | %(levelname)-8s | "
    "[%(filename)s:%(lineno)d] | %(message)s",
)
FIELD_STYLES = (
    coloredlogs.DEFAULT_FIELD_STYLES if COLOREDLOGS_AVAILABLE else {}
)
if COLOREDLOGS_AVAILABLE:
    FIELD_STYLES["levelname"] = {"color": "white", "bold": True}
    FIELD_STYLES["name"] = {"color": "blue"}

LEVEL_STYLES = (
    coloredlogs.DEFAULT_LEVEL_STYLES if COLOREDLOGS_AVAILABLE else {}
)
if COLOREDLOGS_AVAILABLE:
    LEVEL_STYLES["info"] = {"color": "green"}
    LEVEL_STYLES["warning"] = {"color": "yellow"}
    LEVEL_STYLES["error"] = {"color": "red", "bold": True}
    LEVEL_STYLES["critical"] = {
        "color": "red",
        "bold": True,
        "background": "white",
    }
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGER_NAME = "Dropout Disco"
# --- End Config ---

logger = logging.getLogger(LOGGER_NAME)
_logging_initialized = False

# Configure a basic root logger to avoid "No handlers could be found for logger" warnings
# This ensures that any logging calls before our setup_logging() will at least go somewhere
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    force=True,  # Override any existing basicConfig
)


def setup_logging(log_dir=LOGS_DIR, log_file=LOG_FILE_NAME):
    """
    Configures the project logger with colored console output and
    rotating file logging.

    Args:
        log_dir (str): Directory where log files are stored.
        log_file (str): Name of the log file.

    This function:
        - Sets up logging level and handlers.
        - Adds a rotating file handler if enabled.
        - Adds a colored console handler if enabled and coloredlogs is
          available.
        - Ensures handlers are not duplicated on repeated calls.
        - Prevents log propagation to the root logger.
    """
    global _logging_initialized
    if _logging_initialized:
        return

    print(f"⚙️  Configuring {LOGGER_NAME} logging...")
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(level)

    if logger.hasHandlers():
        print("  Clearing existing handlers...")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
    logger.propagate = False

    print(f"  Logger '{LOGGER_NAME}' level set to: {LOG_LEVEL}")

    # --- File Handler (No Color) ---
    if LOG_FILE_ENABLED:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, log_file)
            file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
            fh = RotatingFileHandler(
                log_path,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            fh.setLevel(level)
            fh.setFormatter(file_formatter)
            logger.addHandler(fh)
            print(f"  ✅ File handler added: {log_path}")
        except Exception as e:
            print(f"  ❌ ERROR setting up file log: {e}")

    # --- Console Handler (WITH Color if available) ---
    if LOG_CONSOLE_ENABLED:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)

        if COLOREDLOGS_AVAILABLE:
            console_formatter = coloredlogs.ColoredFormatter(
                fmt=LOG_FORMAT,
                datefmt=DATE_FORMAT,
                level_styles=LEVEL_STYLES,
                field_styles=FIELD_STYLES,
            )
            print("  🎨 Applying colored formatter to console handler.")
        else:
            console_formatter = logging.Formatter(
                LOG_FORMAT, datefmt=DATE_FORMAT
            )
            print("  Falling back to standard console formatter.")

        ch.setFormatter(console_formatter)
        logger.addHandler(ch)
        print("  ✅ Console handler added.")

    if logger.hasHandlers():
        logger.info("🎉 Logging system initialized!")
    else:
        print(f"⚠️ Warning: No handlers configured for {LOGGER_NAME}.")
    _logging_initialized = True

    # Configure other common module loggers to use our settings
    # This helps reduce common "using basic setup" messages
    _configure_common_loggers()

    return logger


def _configure_common_loggers():
    """
    Configure common third-party and project-specific loggers to use our settings.
    This reduces duplicate log messages and ensures consistent formatting.
    """
    common_loggers = [
        "TokenizerUtils",
        "transformers",
        "datasets",
        "PIL",
        "huggingface_hub",
    ]

    for logger_name in common_loggers:
        module_logger = logging.getLogger(logger_name)
        # Remove any existing handlers
        for handler in module_logger.handlers[:]:
            module_logger.removeHandler(handler)
        # Inherit from our main logger
        module_logger.handlers = logger.handlers.copy()
        module_logger.setLevel(logger.level)
        module_logger.propagate = False


def get_logger():
    """
    Returns the configured logger instance. Ensures setup_logging has been called.
    For use when importing the logger from other modules.

    Returns:
        logging.Logger: The configured logger instance
    """
    global _logging_initialized
    if not _logging_initialized:
        setup_logging()
    return logger


# Initialize logging for the main process only
if (
    multiprocessing.current_process().name == "MainProcess"
    and not _logging_initialized
):
    setup_logging()

if __name__ == "__main__":
    if multiprocessing.current_process().name == "MainProcess":
        logger.info("Logging module test (MainProcess). INFO")
        logger.warning("Logging module test (MainProcess). WARNING")
        logger.error("Logging module test (MainProcess). ERROR")
    else:
        proc_name = multiprocessing.current_process().name
        print(
            f"Logging module test (Worker: {proc_name}). "
            "Logger setup skipped."
        )
