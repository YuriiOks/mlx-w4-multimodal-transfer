# Multimodal Transfer Learning / Image Captioning
# File: src/training/pytorch/checkpoint.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: PyTorch save/load checkpoint functions (incl. config).
# Created: 2025-05-05
# Updated: 2025-05-07

import os
import pickle

# --- Add project root for logger ---
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from utils import logger

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- End imports ---

CHECKPOINT_STATE_FILENAME = "training_state_pt.pkl"
MODEL_WEIGHTS_FILENAME = "model_weights_pt.pth"
BEST_MODEL_WEIGHTS_FILENAME = "best_model_weights_pt.pth"


def save_checkpoint_pt(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    metrics_history: Dict[str, List[float]],
    save_dir: Path,
    is_best_model: bool = False,
    model_config_to_save: Optional[Dict] = None,
    dataset_config_to_save: Optional[Dict] = None,
    tokenizer_config_to_save: Optional[Dict] = None,
):
    """
    Save a PyTorch checkpoint including model weights, optimizer, scheduler,
    metrics, and configs.

    Args:
        model (nn.Module): The PyTorch model to save.
        optimizer (optim.Optimizer): The optimizer instance.
        scheduler (Optional[_LRScheduler]): Learning rate scheduler.
        epoch (int): Last completed epoch (0-indexed).
        metrics_history (Dict[str, List[float]]): Metrics history.
        save_dir (Path): Directory to save checkpoint files.
        is_best_model (bool): If True, also save as best model.
        model_config_to_save (Optional[Dict]): Model config for reload.
        dataset_config_to_save (Optional[Dict]): Dataset config for reload.
        tokenizer_config_to_save (Optional[Dict]): Tokenizer config.
    """
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        model_weights_path = save_dir / MODEL_WEIGHTS_FILENAME
        torch.save(model.state_dict(), model_weights_path)

        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
            if scheduler
            else None,
            "epoch": epoch + 1,
            "metrics_history": metrics_history,
            "model_config": model_config_to_save or {},
            "dataset_config": dataset_config_to_save or {},
            "tokenizer_config": tokenizer_config_to_save or {},
        }
        state_path = save_dir / CHECKPOINT_STATE_FILENAME
        with open(state_path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"💾 Saved checkpoint to {state_path} (epoch {epoch+1})")

        if is_best_model:
            best_model_path = save_dir.parent / BEST_MODEL_WEIGHTS_FILENAME
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"🏆 New best PyTorch model weights saved to {best_model_path}"
            )

    except Exception as e:
        logger.error(
            f"❌ Failed to save PyTorch checkpoint: {e}", exc_info=True
        )


def load_checkpoint_pt(
    save_dir: Path, device: torch.device
) -> Tuple[
    Optional[Dict],
    Optional[Dict],
    Optional[Dict],
    int,
    Dict,
    Optional[Dict],
    Optional[Dict],
    Optional[Dict],
]:
    """
    Load a PyTorch checkpoint from disk.

    Args:
        save_dir (Path): Directory containing checkpoint files.
        device (torch.device): Device to map model weights to.

    Returns:
        Tuple:
            model_state_dict (Optional[Dict]): Model weights.
            optimizer_state_dict (Optional[Dict]): Optimizer state.
            scheduler_state_dict (Optional[Dict]): Scheduler state.
            start_epoch (int): Epoch to resume from.
            metrics_history (Dict): Loaded metrics history.
            loaded_model_config (Optional[Dict]): Model config.
            loaded_dataset_config (Optional[Dict]): Dataset config.
            loaded_tokenizer_config (Optional[Dict]): Tokenizer config.
    """
    model_weights_path = save_dir / MODEL_WEIGHTS_FILENAME
    state_path = save_dir / CHECKPOINT_STATE_FILENAME

    model_sd = None
    optimizer_sd = None
    scheduler_sd = None
    start_epoch = 0
    metrics_hist = {
        "avg_train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }
    loaded_model_cfg = None
    loaded_dataset_cfg = None
    loaded_tokenizer_cfg = None

    if model_weights_path.exists() and state_path.exists():
        logger.info(
            f"♻️ Attempting to load PyTorch checkpoint from {save_dir}"
        )
        try:
            model_sd = torch.load(model_weights_path, map_location=device)
            with open(state_path, "rb") as f:
                state = pickle.load(f)

            optimizer_sd = state.get("optimizer_state_dict")
            scheduler_sd = state.get("scheduler_state_dict")
            start_epoch = state.get("epoch", 0)
            loaded_metrics = state.get("metrics_history", {})
            for key in metrics_hist:
                metrics_hist[key] = loaded_metrics.get(key, [])

            loaded_model_cfg = state.get("model_config")
            loaded_dataset_cfg = state.get("dataset_config")
            loaded_tokenizer_cfg = state.get("tokenizer_config")
            logger.info(
                f"✅ PT Checkpoint loaded. Will resume from epoch {start_epoch}."
            )
            logger.info(
                f"   Loaded model config: {'Yes' if loaded_model_cfg else 'No'}"
            )
            logger.info(
                f"   Loaded dataset config: {'Yes' if loaded_dataset_cfg else 'No'}"
            )
            logger.info(
                f"   Loaded tokenizer config: {'Yes' if loaded_tokenizer_cfg else 'No'}"
            )

        except Exception as e:
            logger.error(
                f"❌ Failed loading PT checkpoint: {e}. Will start fresh.",
                exc_info=True,
            )
            model_sd = None
            optimizer_sd = None
            scheduler_sd = None
            start_epoch = 0
            metrics_hist = {k: [] for k in metrics_hist}
            loaded_model_cfg = None
            loaded_dataset_cfg = None
            loaded_tokenizer_cfg = None
    else:
        logger.info(
            "No PyTorch checkpoint found. Starting training from scratch."
        )

    return (
        model_sd,
        optimizer_sd,
        scheduler_sd,
        start_epoch,
        metrics_hist,
        loaded_model_cfg,
        loaded_dataset_cfg,
        loaded_tokenizer_cfg,
    )


# --- Test Block ---
if __name__ == "__main__":
    """
    Test block for checkpoint save/load utilities.
    """
    logger.info("🧪 Testing PyTorch Checkpoint Utils...")
    # Create dummy objects for testing
    dummy_model = nn.Linear(10, 2)
    dummy_optimizer = optim.Adam(dummy_model.parameters())
    dummy_scheduler = optim.lr_scheduler.StepLR(dummy_optimizer, step_size=1)
    dummy_epoch = 5
    dummy_history = {
        "avg_train_loss": [0.1, 0.05],
        "val_accuracy": [90.0, 95.0],
    }
    dummy_save_dir = Path("./temp_pt_checkpoint_test")
    dummy_model_cfg = {"embed_dim": 256, "depth": 3}
    dummy_dataset_cfg = {"image_size": 224, "max_seq_len": 50}
    dummy_tokenizer_cfg = {"vocab_size": 1000}

    logger.info(f"Saving checkpoint to {dummy_save_dir}...")
    save_checkpoint_pt(
        dummy_model,
        dummy_optimizer,
        dummy_scheduler,
        dummy_epoch,
        dummy_history,
        dummy_save_dir,
        is_best_model=True,
        model_config_to_save=dummy_model_cfg,
        dataset_config_to_save=dummy_dataset_cfg,
        tokenizer_config_to_save=dummy_tokenizer_cfg,
    )

    logger.info(f"Loading checkpoint from {dummy_save_dir}...")
    (
        model_sd,
        opt_sd,
        sched_sd,
        epoch,
        hist,
        m_cfg,
        ds_cfg,
        tk_cfg,
    ) = load_checkpoint_pt(dummy_save_dir, torch.device("cpu"))

    if model_sd is not None and opt_sd is not None:
        logger.info(f"Loaded successfully. Resuming from epoch: {epoch}")
        logger.info(f"Loaded history: {hist}")
        logger.info(f"Loaded model_cfg: {m_cfg}")
        logger.info(f"Loaded dataset_cfg: {ds_cfg}")
        logger.info(f"Loaded tokenizer_cfg: {tk_cfg}")
        assert epoch == dummy_epoch + 1
        assert m_cfg == dummy_model_cfg
        assert ds_cfg == dummy_dataset_cfg
        assert tk_cfg == dummy_tokenizer_cfg
        logger.info("✅ Checkpoint save/load values match.")
    else:
        logger.error("❌ Checkpoint loading failed in test.")

    # Clean up
    import shutil

    if dummy_save_dir.exists():
        shutil.rmtree(dummy_save_dir)
    best_model_path = dummy_save_dir.parent / BEST_MODEL_WEIGHTS_FILENAME
    if best_model_path.exists():
        os.remove(best_model_path)
    logger.info("✅ PyTorch Checkpoint Utils test finished.")
