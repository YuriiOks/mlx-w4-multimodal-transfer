# Multimodal Transfer Learning / Image Captioning
# File: src/training/mlx/checkpoint.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: MLX save/load checkpoint functions (incl. config).
# Created: 2025-05-05
# Updated: 2025-05-05


import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils import logger

CHECKPOINT_STATE_FILENAME = "training_state_mlx.pkl"
MODEL_WEIGHTS_FILENAME = "model_weights_mlx.npz"
BEST_MODEL_WEIGHTS_FILENAME = "best_model_weights_mlx.npz"


def save_checkpoint_mlx(
    model,
    optimizer,
    epoch: int,
    metrics_history: Dict[str, List[float]],
    save_dir: Path,
    is_best_model: bool = False,
    model_config_to_save: Optional[Dict] = None,
    dataset_config_to_save: Optional[Dict] = None,
    tokenizer_config_to_save: Optional[Dict] = None,
):
    """
    Save an MLX checkpoint including model weights, optimizer state,
    metrics, and configs.

    Args:
        model: The model to save.
        optimizer: The optimizer (not currently saved).
        epoch (int): Current epoch number.
        metrics_history (Dict[str, List[float]]): Training/validation metrics.
        save_dir (Path): Directory to save checkpoint files.
        is_best_model (bool): If True, also saves as best model.
        model_config_to_save (Optional[Dict]): Model config to save.
        dataset_config_to_save (Optional[Dict]): Dataset config to save.
        tokenizer_config_to_save (Optional[Dict]): Tokenizer config to save.
    """
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        model_weights_path = save_dir / MODEL_WEIGHTS_FILENAME
        # Save model weights as numpy arrays
        np.savez(model_weights_path, **model.state_dict())
        state = {
            "epoch": epoch + 1,
            "metrics_history": metrics_history,
            "model_config": model_config_to_save or {},
            "dataset_config": dataset_config_to_save or {},
            "tokenizer_config": tokenizer_config_to_save or {},
            # Optionally add optimizer state if available
        }
        state_path = save_dir / CHECKPOINT_STATE_FILENAME
        with open(state_path, "wb") as f:
            pickle.dump(state, f)
        logger.info(
            f"💾 Saved MLX checkpoint to {state_path} (epoch {epoch+1})"
        )
        if is_best_model:
            best_model_path = save_dir.parent / BEST_MODEL_WEIGHTS_FILENAME
            np.savez(best_model_path, **model.state_dict())
            logger.info(
                f"🏆 New best MLX model weights saved to {best_model_path}"
            )
    except Exception as e:
        logger.error(f"❌ Failed to save MLX checkpoint: {e}", exc_info=True)


def load_checkpoint_mlx(
    model,
    save_dir: Path,
) -> Tuple[int, Dict, Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Load an MLX checkpoint from disk and restore model weights.

    Args:
        model: The model to load weights into.
        save_dir (Path): Directory containing checkpoint files.

    Returns:
        Tuple containing:
            - start_epoch (int): Epoch to resume from.
            - metrics_history (Dict): Loaded metrics history.
            - model_config (Optional[Dict]): Loaded model config.
            - dataset_config (Optional[Dict]): Loaded dataset config.
            - tokenizer_config (Optional[Dict]): Loaded tokenizer config.
    """
    model_weights_path = save_dir / MODEL_WEIGHTS_FILENAME
    state_path = save_dir / CHECKPOINT_STATE_FILENAME
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
        logger.info(f"♻️ Attempting to load MLX checkpoint from {save_dir}")
        try:
            weights = np.load(model_weights_path)
            model.load_state_dict({k: weights[k] for k in weights})
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            start_epoch = state.get("epoch", 0)
            loaded_metrics = state.get("metrics_history", {})
            for key in metrics_hist:
                metrics_hist[key] = loaded_metrics.get(key, [])
            loaded_model_cfg = state.get("model_config")
            loaded_dataset_cfg = state.get("dataset_config")
            loaded_tokenizer_cfg = state.get("tokenizer_config")
            logger.info(
                f"✅ MLX Checkpoint loaded. Will resume from epoch {start_epoch}."
            )
        except Exception as e:
            logger.error(
                f"❌ Failed loading MLX checkpoint: {e}. Will start fresh.",
                exc_info=True,
            )
    else:
        logger.info("No MLX checkpoint found. Starting training from scratch.")
    return (
        start_epoch,
        metrics_hist,
        loaded_model_cfg,
        loaded_dataset_cfg,
        loaded_tokenizer_cfg,
    )


if __name__ == "__main__":
    pass
