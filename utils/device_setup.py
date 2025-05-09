# Multimodal Transfer Learning / Image Captioning
# File: utils/device_setup.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton) # Corrected Team Name
# Description: Detects and sets the appropriate PyTorch device.
# Created: 2025-05-05
# Updated: 2025-05-09 # Updated Date

import os  # Import os for environment variable access

import torch

# Import the logger instance directly from the logging module
from .logging import logger

# --- Recommended Environment Variables (Informational) ---
# Consider setting these externally for stability:
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

_pytorch_device_logged_once = (
    False  # Add this global-like flag at module level
)


def get_device() -> torch.device:
    """
    Detects and returns the most suitable PyTorch device.

    Checks for Apple Silicon MPS (Metal Performance Shaders) support,
    then CUDA, then falls back to CPU if neither is available.
    Logs the selected device.

    Returns:
        torch.device: The selected PyTorch device object.
    """
    global _pytorch_device_logged_once
    selected_device_str = ""  # To hold string representation for logging

    logger.debug("⚙️  Checking for available PyTorch hardware accelerators...")

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        selected_device = torch.device("mps")
        selected_device_str = "MPS"
        logger.info("✅ Apple MPS device found and available. Selecting MPS.")
        # Log MPS fallback status (optional, but good for debugging)
        mps_fallback_enabled = (
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"
        )
        logger.debug(
            f"  -> MPS fallback to CPU is {'ENABLED' if mps_fallback_enabled else 'DISABLED'} "
            f"(PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'Not Set')})."
        )
    elif torch.cuda.is_available():
        selected_device = torch.device("cuda")
        selected_device_str = (
            f"CUDA ({torch.cuda.get_device_name(0)})"  # Get specific GPU name
        )
        logger.info(f"✅ {selected_device_str} device found. Selecting CUDA.")
    else:
        selected_device = torch.device("cpu")
        selected_device_str = "CPU"
        logger.warning(
            f"⚠️ MPS and CUDA not available. Falling back to {selected_device_str}."
        )
        if not (
            torch.backends.mps.is_available() and torch.backends.mps.is_built()
        ):
            logger.debug(
                f"   (MPS status: available={torch.backends.mps.is_available()}, "
                f"built={hasattr(torch.backends, 'mps') and torch.backends.mps.is_built()})"
            )
        if not torch.cuda.is_available():
            logger.debug("   (CUDA status: not available)")

    # Only log the final selection once per script run for PyTorch device
    if not _pytorch_device_logged_once:
        logger.info(
            f"✨ PyTorch selected compute device: {selected_device_str}"
        )
        _pytorch_device_logged_once = True
    elif selected_device_str != "CPU":
        logger.debug(f"PyTorch using device: {selected_device_str}")
    return selected_device


# --- Example Usage (for testing this script directly) ---
if __name__ == "__main__":
    logger.info("🚀 Running PyTorch device setup check directly...")
    device = get_device()
    logger.info(f"🔍 Device object returned: {device}")

    try:
        logger.debug(
            f"Attempting to create PyTorch test tensor on device '{device}'..."
        )
        x = torch.randn(3, 3, device=device)
        logger.info(
            f"✅ Successfully created PyTorch test tensor on device '{device}'."
        )
        logger.debug(f"Test tensor value:\n{x}")
        if device.type == "mps":  # Check for MPS specifically
            logger.debug(f"  Tensor backend device check for MPS: {x.device}")
    except Exception as e:
        logger.error(
            f"❌ Failed to create PyTorch test tensor on device '{device}': {e}",
            exc_info=True,
        )
