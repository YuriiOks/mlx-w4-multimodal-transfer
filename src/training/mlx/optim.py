# Multimodal Transfer Learning / Image Captioning
# File: src/training/mlx/optim.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: (Optional) MLX optimizer/scheduler setup helpers.
# Created: 2025-05-05
# Updated: 2025-05-05

import os
import sys  # Added for sys.path manipulation
from pathlib import Path

import mlx.optimizers as mx_optim

from utils import logger

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent.parent  # Go up three levels
if str(project_root) not in sys.path:
    print(f"🚂 [trainer_mlx.py] Adding project root: {project_root}")
    sys.path.insert(0, str(project_root))


def create_optimizer_mlx(
    model,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    **kwargs,
):
    """
    Create an MLX optimizer for the given model.

    Args:
        model: The MLX model whose parameters will be optimized.
        optimizer_name (str): Name of the optimizer ("adamw", "adam", "sgd").
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 penalty).
        **kwargs: Additional optimizer-specific parameters.

    Returns:
        Optimizer: Instantiated MLX optimizer.
    """
    logger.info(
        f"Creating MLX Optimizer: {optimizer_name} with LR={lr}, "
        f"WD={weight_decay}"
    )

    # Create the optimizer without any parameter handling
    if optimizer_name.lower() == "adamw":
        logger.info("Creating AdamW optimizer")
        optimizer = mx_optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adam":
        logger.info("Creating Adam optimizer")
        optimizer = mx_optim.Adam(learning_rate=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        momentum = kwargs.get("momentum", 0.0)
        logger.info(f"Creating SGD optimizer with momentum={momentum}")
        optimizer = mx_optim.SGD(
            learning_rate=lr, weight_decay=weight_decay, momentum=momentum
        )
    else:
        logger.warning(
            f"Unsupported optimizer '{optimizer_name}'. Defaulting to AdamW."
        )
        optimizer = mx_optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    # # Override the apply_gradients method to be more robust
    # original_apply_gradients = optimizer.apply_gradients

    # def safe_apply_gradients(gradients, parameters):
    #     try:
    #         # Try the original method first
    #         return original_apply_gradients(gradients, parameters)
    #     except TypeError as e:
    #         if "'str' object does not support item assignment" in str(e):
    #             logger.warning("⚠️ Optimizer state error detected, using fallback update")
    #             # Simple fallback: just apply a basic gradient step
    #             updated_params = {}
    #             for k, v in gradients.items():
    #                 if k in parameters:
    #                     param = parameters[k]
    #                     # Simple SGD-like update
    #                     updated_params[k] = param - lr * v
    #             return updated_params
    #         else:
    #             # Re-raise if it's not the specific error we're handling
    #             raise e

    # # Monkey-patch the apply_gradients method
    # optimizer.apply_gradients = safe_apply_gradients

    return optimizer


def create_scheduler_mlx(
    optimizer,
    scheduler_name: str,
    total_epochs: int,
    base_lr: float,
    scheduler_params: dict = None,
):
    """
    Create an MLX learning rate scheduler.

    Args:
        optimizer: The MLX optimizer to schedule (not used, for API compat).
        scheduler_name (str): Name of the scheduler ("cosineannealinglr",
            "steplr").
        total_epochs (int): Total number of epochs for training.
        base_lr (float): Base learning rate.
        scheduler_params (dict, optional): Additional scheduler parameters.

    Returns:
        Callable or None: Learning rate schedule function or None if not used.
    """
    import mlx.optimizers as mx_optim

    if scheduler_params is None:
        scheduler_params = {}
    if not scheduler_name:
        logger.info("No LR scheduler specified.")
        return None
    logger.info(f"Creating MLX LR Scheduler: {scheduler_name}")
    if scheduler_name.lower() == "cosineannealinglr":
        t_max = scheduler_params.get("T_max", total_epochs)
        eta_min = scheduler_params.get("eta_min", base_lr * 0.01)
        logger.info(
            f"  CosineAnnealingLR params: T_max={t_max}, eta_min={eta_min}"
        )
        # MLX cosine_decay: init, decay_steps, end
        return mx_optim.cosine_decay(
            init=base_lr, decay_steps=t_max, end=eta_min
        )
    elif scheduler_name.lower() == "steplr":
        step_size = scheduler_params.get("step_size", 30)
        gamma = scheduler_params.get("gamma", 0.1)
        logger.info(f"  StepLR params: step_size={step_size}, gamma={gamma}")
        # MLX step_decay: init, decay_steps, decay_factor
        return mx_optim.step_decay(
            init=base_lr, decay_steps=step_size, decay_factor=gamma
        )
    else:
        logger.warning(
            f"Unsupported scheduler '{scheduler_name}'. "
            f"No scheduler will be used."
        )
        return None


if __name__ == "__main__":
    """
    Main entry point for the optimizer module.
    Currently, this block does not execute any code.
    """
    pass
