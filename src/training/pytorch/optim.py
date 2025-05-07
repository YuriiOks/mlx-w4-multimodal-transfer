# Multimodal Transfer Learning / Image Captioning
# File: src/training/pytorch/optim.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: PyTorch optimizer and learning rate scheduler setup.
# Created: 2025-05-07
# Updated: 2025-05-07

import os

# --- Add project root for logger ---
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch.optim as optim
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler  # For type hinting
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)

from utils import logger

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- End imports ---


def create_optimizer_pt(
    model: Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    **kwargs,
) -> optim.Optimizer:
    """
    Create a PyTorch optimizer for the given model.

    Args:
        model (Module): The PyTorch model.
        optimizer_name (str): Name of the optimizer ("adamw", "adam", "sgd").
        lr (float): Learning rate.
        weight_decay (float): Weight decay (L2 penalty).
        **kwargs: Additional optimizer-specific parameters.

    Returns:
        optim.Optimizer: Instantiated optimizer.
    """
    logger.info(
        f"Creating PyTorch Optimizer: {optimizer_name} with "
        f"LR={lr}, WD={weight_decay}"
    )
    if optimizer_name.lower() == "adamw":
        return optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs
        )
    elif optimizer_name.lower() == "adam":
        return optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs
        )
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.9),
            **kwargs,
        )
    else:
        logger.warning(
            f"Unsupported optimizer '{optimizer_name}'. "
            "Defaulting to AdamW."
        )
        return optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )


def create_scheduler_pt(
    optimizer: optim.Optimizer,
    scheduler_name: Optional[str],
    total_epochs: int,
    base_lr: float,
    scheduler_params: Optional[Dict[str, Any]] = None,
) -> Optional[_LRScheduler]:
    """
    Create a PyTorch learning rate scheduler.

    Args:
        optimizer (optim.Optimizer): The optimizer to schedule.
        scheduler_name (Optional[str]): Name of the scheduler.
        total_epochs (int): Total number of epochs (for T_max).
        base_lr (float): Base learning rate (for eta_min).
        scheduler_params (Optional[Dict[str, Any]]): Scheduler parameters.

    Returns:
        Optional[_LRScheduler]: Instantiated scheduler or None.
    """
    if scheduler_params is None:
        scheduler_params = {}
    if not scheduler_name:
        logger.info("No LR scheduler specified.")
        return None

    logger.info(f"Creating PyTorch LR Scheduler: {scheduler_name}")
    if scheduler_name.lower() == "cosineannealinglr":
        t_max = scheduler_params.get("T_max", total_epochs)
        eta_min = scheduler_params.get("eta_min", base_lr * 0.01)
        logger.info(
            f"  CosineAnnealingLR params: T_max={t_max}, eta_min={eta_min}"
        )
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    elif scheduler_name.lower() == "reducelronplateau":
        params = {
            "mode": scheduler_params.get("mode", "min"),
            "factor": scheduler_params.get("factor", 0.1),
            "patience": scheduler_params.get("patience", 10),
            "verbose": scheduler_params.get("verbose", True),
        }
        logger.info(f"  ReduceLROnPlateau params: {params}")
        return ReduceLROnPlateau(optimizer, **params)
    elif scheduler_name.lower() == "steplr":
        params = {
            "step_size": scheduler_params.get("step_size", 30),
            "gamma": scheduler_params.get("gamma", 0.1),
        }
        logger.info(f"  StepLR params: {params}")
        return StepLR(optimizer, **params)
    else:
        logger.warning(
            f"Unsupported scheduler '{scheduler_name}'. "
            "No scheduler will be used."
        )
        return None


if __name__ == "__main__":
    """
    Test block for optimizer and scheduler creation utilities.
    """
    import torch.nn as nn

    logger.info("🧪 Testing PyTorch Optimizer/Scheduler Utils...")
    dummy_model = nn.Linear(10, 2)
    opt_cfg = {"optimizer": "AdamW", "base_lr": 1e-3, "weight_decay": 1e-2}
    sched_cfg = {
        "scheduler": "CosineAnnealingLR",
        "scheduler_params": {"T_max": 10, "eta_min": 1e-5},
    }

    optimizer = create_optimizer_pt(
        dummy_model,
        opt_cfg["optimizer"],
        opt_cfg["base_lr"],
        opt_cfg["weight_decay"],
    )
    assert isinstance(optimizer, optim.AdamW)
    logger.info(f"✅ Optimizer created: {type(optimizer)}")

    scheduler = create_scheduler_pt(
        optimizer,
        sched_cfg["scheduler"],
        10,
        opt_cfg["base_lr"],
        sched_cfg["scheduler_params"],
    )
    assert isinstance(scheduler, CosineAnnealingLR)
    logger.info(f"✅ Scheduler created: {type(scheduler)}")
    logger.info("✅ PyTorch Optim/Scheduler utils seem OK.")
