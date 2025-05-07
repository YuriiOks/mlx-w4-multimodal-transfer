# Multimodal Transfer Learning / Image Captioning
# File: src/training/pytorch/trainer.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: PyTorch training and evaluation loops.
# Created: 2025-05-07
# Updated: 2025-05-07

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.constants import PAD_TOKEN_ID

# --- 👇 Import checkpoint functions ---
from src.training.pytorch.checkpoint import (
    save_checkpoint_pt,  # load_checkpoint_pt is used in main script
)
from utils import logger

# --- Add project root ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


try:
    import wandb
except ImportError:
    logger.warning("⚠️ wandb not installed.")
    wandb = None


def train_epoch_pt(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_num: int,
    total_epochs: int,
    wandb_run: Optional[Any] = None,
    log_frequency: int = 100,
    gradient_clipping: Optional[float] = 1.0,
    phase: int = 1,
    is_encoder_decoder: bool = False,
) -> float:
    """
    Runs one training epoch for the given model and dataloader.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to use.
        epoch_num (int): Current epoch number.
        total_epochs (int): Total number of epochs.
        wandb_run (Optional[Any]): Weights & Biases run object.
        log_frequency (int): Logging frequency for W&B.
        gradient_clipping (Optional[float]): Max norm for gradients.
        phase (int): Training phase (1, 2, or 3).
        is_encoder_decoder (bool): If model is encoder-decoder.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    if num_batches == 0:
        logger.warning("Train dataloader empty.")
        return 0.0

    data_iterator = tqdm(
        dataloader, desc=f"Epoch {epoch_num+1}/{total_epochs} Train", ncols=76
    )
    for batch_idx, batch_data in enumerate(data_iterator):
        images = batch_data["pixel_values"].to(device)
        input_ids = batch_data["input_ids"].to(device)

        optimizer.zero_grad()

        if is_encoder_decoder:
            decoder_input_ids = input_ids[:, :-1]
            outputs = model(images, decoder_input_ids)
            targets = input_ids[:, 1:]
            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1)
            )
        else:
            outputs = model(images)
            if phase == 1:
                loss = criterion(outputs, input_ids)
            elif phase == 2:
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[-1]),
                    input_ids.reshape(-1),
                )
            else:
                loss = torch.tensor(0.0, device=device)
                logger.error("Unsupported phase in train_epoch_pt")

        loss.backward()
        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clipping
            )
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        data_iterator.set_postfix(loss=f"{batch_loss:.4f}")
        if wandb_run and batch_idx % log_frequency == 0:
            global_step = epoch_num * num_batches + batch_idx
            wandb_run.log(
                {
                    "batch_loss": batch_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_fractional": epoch_num + (batch_idx / num_batches),
                    "global_step": global_step,
                }
            )

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate_model_pt(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    phase: int = 1,
    is_encoder_decoder: bool = False,
    pad_token_id: int = PAD_TOKEN_ID,
) -> Dict[str, float]:
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use.
        phase (int): Training phase (1, 2, or 3).
        is_encoder_decoder (bool): If model is encoder-decoder.
        pad_token_id (int): Padding token id.

    Returns:
        Dict[str, float]: Dictionary with val_loss and val_accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_valid_tokens = 0
    num_batches = len(dataloader)
    if num_batches == 0:
        return {"val_loss": 0.0, "val_accuracy": 0.0}

    logger.info(f"🧪 Starting PyTorch evaluation (Phase {phase})...")
    with torch.no_grad():
        data_iterator = tqdm(dataloader, desc="Evaluating", ncols=76)
        for batch_data in data_iterator:
            images = batch_data["pixel_values"].to(device)
            input_ids = batch_data["input_ids"].to(device)

            if is_encoder_decoder:
                decoder_input_ids = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                outputs = model(images, decoder_input_ids)
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1)
                )
                preds = torch.argmax(outputs, dim=-1)
                mask = targets != pad_token_id
                correct_tokens = ((preds == targets) & mask).sum().item()
                valid_tokens = mask.sum().item()
                total_correct += correct_tokens
                total_valid_tokens += valid_tokens
                current_acc = (
                    (correct_tokens / valid_tokens * 100.0)
                    if valid_tokens > 0
                    else 0.0
                )
                valid_tokens_batch = valid_tokens
            else:
                outputs = model(images)
                if phase == 1:
                    loss = criterion(outputs, input_ids)
                    preds = torch.argmax(outputs, dim=1)
                    correct_batch = (preds == input_ids).sum().item()
                    valid_tokens_batch = input_ids.size(0)
                elif phase == 2:
                    loss = criterion(
                        outputs.reshape(-1, outputs.shape[-1]),
                        input_ids.reshape(-1),
                    )
                    preds = torch.argmax(outputs, dim=2)
                    correct_batch = (preds == input_ids).sum().item()
                    valid_tokens_batch = input_ids.numel()
                else:
                    loss = torch.tensor(0.0)
                    correct_batch = 0
                    valid_tokens_batch = 0
                total_correct += correct_batch
                total_valid_tokens += valid_tokens_batch
                current_acc = (
                    (correct_batch / valid_tokens_batch * 100.0)
                    if valid_tokens_batch > 0
                    else 0.0
                )

            total_loss += loss.item() * valid_tokens_batch
            data_iterator.set_postfix(
                loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%"
            )

    avg_loss = (
        total_loss / total_valid_tokens if total_valid_tokens > 0 else 0.0
    )
    avg_accuracy = (
        (total_correct / total_valid_tokens) * 100.0
        if total_valid_tokens > 0
        else 0.0
    )
    logger.info(
        f"🧪 PT Eval finished. Loss: {avg_loss:.4f}, "
        f"Acc: {avg_accuracy:.2f}% ({total_correct}/{total_valid_tokens})"
    )
    return {"val_loss": avg_loss, "val_accuracy": avg_accuracy}


def train_model_pt(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    optimizer: optim.Optimizer,
    device: torch.device,
    target_epochs: int,
    start_epoch: int,
    metrics_history: Dict[str, List[float]],
    model_save_dir: Path,
    config: Dict,
    phase: int,
    run_name: str,
    wandb_run: Optional[Any] = None,
    lr_scheduler: Optional[_LRScheduler] = None,
    save_every: int = 1,
) -> Dict[str, List[float]]:
    """
    Main training loop for the model.

    Args:
        model (nn.Module): Model to train.
        train_dataloader (DataLoader): Training data loader.
        val_dataloader (Optional[DataLoader]): Validation data loader.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to use.
        target_epochs (int): Total epochs to train.
        start_epoch (int): Starting epoch (for resuming).
        metrics_history (Dict[str, List[float]]): Metrics history.
        model_save_dir (Path): Directory to save checkpoints.
        config (Dict): Full config for params.
        phase (int): Training phase.
        run_name (str): Name of the run.
        wandb_run (Optional[Any]): W&B run object.
        lr_scheduler (Optional[_LRScheduler]): LR scheduler.
        save_every (int): Save checkpoint every N epochs.

    Returns:
        Dict[str, List[float]]: Updated metrics history.
    """
    logger.info(
        f"🚀 Starting/Resuming PyTorch Training: Run='{run_name}', Phase={phase}"
    )
    logger.info(
        f"   Target Epochs: {target_epochs}, Starting from Epoch: {start_epoch}"
    )
    model.to(device)

    model_cfg_to_save = config.get("model", {})
    dataset_cfg_to_save = config.get("dataset", {})
    tokenizer_cfg_to_save = config.get("tokenizer", {})
    pad_token_id = tokenizer_cfg_to_save.get("pad_token_id", PAD_TOKEN_ID)
    is_enc_dec = phase == 3

    if phase == 3:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    best_val_accuracy = 0.0
    if metrics_history.get("val_accuracy"):
        valid_accs = [
            acc
            for acc in metrics_history["val_accuracy"]
            if not (isinstance(acc, float) and math.isnan(acc))
        ]
        if valid_accs:
            best_val_accuracy = max(valid_accs)

    for epoch in range(start_epoch, target_epochs):
        avg_train_loss = train_epoch_pt(
            model,
            train_dataloader,
            criterion,
            optimizer,
            device,
            epoch,
            target_epochs,
            wandb_run,
            phase=phase,
            is_encoder_decoder=is_enc_dec,
        )
        metrics_history["avg_train_loss"].append(avg_train_loss)

        val_metrics = {}
        if val_dataloader:
            val_metrics = evaluate_model_pt(
                model,
                val_dataloader,
                criterion,
                device,
                phase,
                is_enc_dec,
                pad_token_id,
            )
            metrics_history["val_loss"].append(
                val_metrics.get("val_loss", float("nan"))
            )
            metrics_history["val_accuracy"].append(
                val_metrics.get("val_accuracy", float("nan"))
            )
        else:
            metrics_history["val_loss"].append(float("nan"))
            metrics_history["val_accuracy"].append(float("nan"))

        current_lr = optimizer.param_groups[0]["lr"]
        metrics_history["learning_rate"].append(current_lr)
        if lr_scheduler:
            if isinstance(
                lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                lr_scheduler.step(val_metrics.get("val_loss", float("inf")))
            else:
                lr_scheduler.step()

        if wandb_run:
            log_data = {
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "learning_rate": current_lr,
                **val_metrics,
            }
            wandb_run.log(log_data)

        current_val_acc = val_metrics.get("val_accuracy", -1.0)
        is_best = current_val_acc > best_val_accuracy
        if is_best:
            best_val_accuracy = current_val_acc
            logger.info(
                f"🏆 New best val_accuracy: {best_val_accuracy:.2f}% "
                f"at Epoch {epoch+1}"
            )

        if (
            (epoch + 1) % save_every == 0
            or epoch == target_epochs - 1
            or is_best
        ):
            save_checkpoint_pt(
                model,
                optimizer,
                lr_scheduler,
                epoch,
                metrics_history,
                model_save_dir,
                is_best,
                model_cfg_to_save,
                dataset_cfg_to_save,
                tokenizer_cfg_to_save,
                phase,
            )
    logger.info("🏁 PyTorch Training finished for this run.")
    return metrics_history


if __name__ == "__main__":
    """
    Minimal test block for syntax checking.
    """
    logger.info(
        "🧪 Basic Pytorch Trainer function calls (no actual training)..."
    )
    logger.info("✅ PyTorch Trainer test block finished (syntax checks only).")
