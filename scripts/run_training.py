# Multimodal Transfer Learning / Image Captioning
# File: scripts/run_training.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: Main unified script to run training for PyTorch or MLX.
# Created: 2025-05-07
# Updated: 2025-05-07

import argparse
import os
import sys
from pathlib import Path

import torch  # Added to fix F821 errors for torch usage

from src.common.tokenizer import init_tokenizer

# --- Project-specific imports ---
from utils import load_config, logger, plot_metrics, save_metrics

# --- Add project root to sys.path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    print(f"🚀 [run_training.py] Adding project root: {project_root}")
    sys.path.insert(0, project_root)

# W&B Import
try:
    import wandb
except ImportError:
    logger.warning("⚠️ wandb not installed. Experiment tracking disabled.")
    wandb = None


def parse_args(config: dict):
    """
    Parse command-line arguments and merge with config defaults.

    Args:
        config (dict): Configuration loaded from file.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train Image Captioning Model."
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="pytorch",
        choices=["pytorch", "mlx"],
        help="ML framework to use (pytorch or mlx).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="config.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.get("training", {}).get("epochs", 10),
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.get("training", {}).get("batch_size", 32),
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.get("training", {}).get("base_lr", 1e-4),
        help="Base learning rate.",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=config.get("training", {}).get("weight_decay", 0.01),
        help="Weight decay.",
    )
    parser.add_argument(
        "--model-save-dir",
        type=str,
        default=config.get("paths", {}).get(
            "model_save_dir", "models/image_captioning"
        ),
        help="Base dir for models.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=config.get("wandb", {}).get(
            "project_name", "image-captioning"
        ),
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=config.get("wandb", {}).get("entity_name", None),
        help="W&B entity.",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None, help="Custom W&B run name."
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        default=False,
        help="Disable W&B logging.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() // 2 if os.cpu_count() else 0,
        help="DataLoader workers (PyTorch only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.get("training", {}).get("seed", 42),
        help="Random seed.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from latest checkpoint.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run in debug mode (small data, few epochs).",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable half-precision (float16) for model weights (for compatibility)",
    )

    args = parser.parse_args()

    # If debug mode, override some settings for quick runs
    if args.debug:
        logger.info("🐛 Running in DEBUG mode.")
        args.epochs = 2
        args.batch_size = min(args.batch_size, 8)
        args.no_wandb = True

    logger.info(f"--- Effective Configuration ({args.framework.upper()}) ---")
    for arg, value in vars(args).items():
        logger.info(f"  --{arg.replace('_', '-'): <20}: {value}")
    logger.info("------------------------------------")
    return args


def main():
    """
    Main entry point for training script.
    Loads config, parses args, sets up environment, and runs training.
    """
    # --- Simpler approach for now: Get config_path from initial minimal parse ---
    initial_parser = argparse.ArgumentParser(
        add_help=False
    )  # Don't interfere with main parser help
    initial_parser.add_argument(
        "--config-path", type=str, default="config.yaml"
    )
    config_arg, _ = initial_parser.parse_known_args()

    config_from_file = load_config(
        config_path=config_arg.config_path
    )  # Pass the path
    if config_from_file is None:
        logger.error(
            f"❌ Config file '{config_arg.config_path}' failed to load! Exiting."
        )
        return
    args = parse_args(config_from_file)  # Now parse fully

    # --- Get Full Config for Modules (move this up before W&B logic) ---
    full_config = config_from_file

    # --- W&B Init ---
    run = None
    framework_prefix = "PT" if args.framework == "pytorch" else "MLX"
    run_name_base = (
        f"{framework_prefix}_Cap_E{args.epochs}_LR{args.lr}_B{args.batch_size}"
    )
    if args.wandb_run_name:
        run_name = args.wandb_run_name
    else:
        run_name = run_name_base

    # --- W&B project name selection (debug print) ---
    wandb_project_name = args.wandb_project
    if args.framework == "mlx":
        wandb_project_name = full_config.get("wandb", {}).get(
            "project_name_mlx", wandb_project_name
        )
        print(f"[DEBUG] Using W&B project name for MLX: {wandb_project_name}")
    else:
        wandb_project_name = full_config.get("wandb", {}).get(
            "project_name_pt", wandb_project_name
        )
        print(
            f"[DEBUG] Using W&B project name for PyTorch: {wandb_project_name}"
        )

    if wandb is not None and not args.no_wandb:
        try:
            run = wandb.init(
                project=wandb_project_name,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args),
                resume="allow",
            )
            logger.info(f"📊 Initialized W&B run: {run.name} ({run.url})")
        except Exception as e:
            logger.error(f"❌ Failed W&B init: {e}", exc_info=True)
            run = None
    else:
        logger.info("📊 W&B logging disabled.")
        run_name = (
            f"{run_name_base}_local"
            if args.wandb_run_name is None
            else args.wandb_run_name
        )

    # --- Prepare Tokenizer (Common) ---
    tokenizer_cfg = full_config.get("tokenizer", {})
    hf_tokenizer_name = tokenizer_cfg.get("hf_name", "gpt2")
    tokenizer = init_tokenizer(hf_tokenizer_name)
    if tokenizer is None:
        logger.error("❌ Tokenizer failed to initialize!")
        return
    _VOCAB_SIZE = len(tokenizer)
    full_config["tokenizer"]["vocab_size"] = _VOCAB_SIZE

    # --- PyTorch Specific Path ---
    if args.framework == "pytorch":
        from datasets import load_dataset
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        from src.data.dataloader import get_dataloader
        from src.data.image_datasets import ImageCaptionDatasetPT
        from src.models.pytorch.caption_model import ImageCaptionerPT
        from src.training.pytorch.checkpoint import load_checkpoint_pt
        from src.training.pytorch.trainer import train_model_pt

        # --- Device selection for PyTorch ---
        pt_device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                else "cpu"
            )
        )

        logger.info("--- Loading PyTorch Data ---")
        dataset_params = full_config.get("dataset", {})
        train_transform = get_pytorch_image_transforms(
            image_size=dataset_params["image_size"], is_train=True
        )

        hf_dataset_name = dataset_params.get("name", "nlphuji/flickr30k")
        logger.info(f"Loading '{hf_dataset_name}' dataset...")
        try:
            logger.info(
                f"Attempting to load and process FULL dataset "
                f"'{hf_dataset_name}' for PyTorch..."
            )
            full_hf_dataset = load_dataset(
                hf_dataset_name,
                name=dataset_params.get("hf_config_name"),
                split="test",
                trust_remote_code=True,
            )
            logger.info(
                f"Loaded HF dataset with {len(full_hf_dataset)} samples."
            )

            full_caption_dataset = ImageCaptionDatasetPT(
                dataset_name=dataset_params.get("name"),
                raw_dataset_input=full_hf_dataset,
                split="all",
                tokenizer=tokenizer,
                image_transform=train_transform,
                max_seq_len=tokenizer_cfg.get("max_seq_len", 50),
                caption_col=dataset_params.get("caption_col", "caption"),
                image_col=dataset_params.get("image_col", "image"),
                max_samples=None,
            )

            if len(full_caption_dataset) == 0:
                raise ValueError("Full caption dataset resulted in 0 samples.")

            from torch.utils.data import random_split

            val_ratio = 0.1
            num_total = len(full_caption_dataset)
            val_len = int(val_ratio * num_total)
            train_len = num_total - val_len
            train_dataset, val_dataset = random_split(
                full_caption_dataset,
                [train_len, val_len],
                generator=torch.Generator().manual_seed(args.seed),
            )

            logger.info(
                f"Created train dataset with {len(train_dataset)} samples "
                f"and val dataset with {len(val_dataset)} samples."
            )

        except Exception as e:
            logger.error(
                f"❌ Error during PyTorch data loading/splitting: {e}",
                exc_info=True,
            )
            train_dataset, val_dataset = None, None

        if (
            not train_dataset
            or not val_dataset
            or len(train_dataset) == 0
            or len(val_dataset) == 0
        ):
            logger.error(
                "❌ PyTorch dataset loading resulted in empty dataset. Exiting."
            )
            if run:
                run.finish(exit_code=1)
                return
        train_dataloader = get_dataloader(
            train_dataset,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_dataloader = get_dataloader(
            val_dataset,
            args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        logger.info("--- Initializing PyTorch Model ---")
        model_params = full_config.get("model", {})
        encoder_params = full_config.get("encoder", {})
        model = ImageCaptionerPT(
            vision_encoder_name=encoder_params.get(
                "hf_name", DEFAULT_VIT_MODEL
            ),
            decoder_vocab_size=_VOCAB_SIZE,
            decoder_embed_dim=model_params.get("decoder_embed_dim", 768),
            decoder_num_heads=model_params.get("decoder_num_heads", 8),
            decoder_ffn_dim=model_params.get("decoder_embed_dim", 768)
            * int(model_params.get("decoder_ffn_dim_ratio", 4.0)),
            decoder_depth=model_params.get("decoder_depth", 6),
            max_seq_len=tokenizer_cfg.get("max_seq_len", 50),
            dropout=model_params.get("dropout", 0.1),
            freeze_encoder=encoder_params.get("freeze", True),
            device=pt_device,
        )

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )

        start_epoch = 0
        metrics_history = {
            "avg_train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }
        run_save_path = Path(args.model_save_dir) / run_name

        if args.resume:
            (
                model_sd,
                opt_sd,
                sched_sd,
                s_epoch,
                m_hist,
                loaded_model_cfg,
                loaded_ds_cfg,
                loaded_tk_cfg,
                loaded_phase,
            ) = load_checkpoint_pt(run_save_path, pt_device)

            if loaded_model_cfg and loaded_ds_cfg and loaded_tk_cfg:
                logger.info("Re-instantiating model from checkpoint config...")
                model = ImageCaptionerPT(
                    vision_encoder_name=full_config["encoder"]["hf_name"],
                    decoder_vocab_size=loaded_tk_cfg["vocab_size"],
                    decoder_embed_dim=loaded_model_cfg["decoder_embed_dim"],
                    max_seq_len=loaded_tk_cfg["max_seq_len"],
                    device=pt_device,
                )
            model.to(pt_device)
            if model_sd:
                model.load_state_dict(model_sd)
            if opt_sd:
                optimizer.load_state_dict(opt_sd)
            if sched_sd and scheduler:
                scheduler.load_state_dict(sched_sd)
            start_epoch = s_epoch
            metrics_history = m_hist

        model.to(pt_device)
        logger.info(
            f"PyTorch Model Parameters: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable"
        )

        logger.info("--- Starting PyTorch Training ---")
        train_model_pt(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            device=pt_device,
            target_epochs=args.epochs,
            start_epoch=start_epoch,
            metrics_history=metrics_history,
            model_save_dir=run_save_path,
            config=full_config,
            run_name=run_name,
            wandb_run=run,
            lr_scheduler=scheduler,
            save_every=full_config.get("training", {}).get("save_every", 1),
        )

    elif args.framework == "mlx":
        from src.data.dataloader import get_dataloader
        from src.data.image_datasets import ImageCaptionDatasetPT
        from src.models.mlx.caption_model import ImageCaptionerMLX
        from src.training.mlx.checkpoint import load_checkpoint_mlx
        from src.training.mlx.optim import (
            create_optimizer_mlx,
            create_scheduler_mlx,
        )
        from src.training.mlx.trainer import train_model_mlx

        logger.info("--- Loading MLX Data ---")
        dataset_params = full_config.get("dataset", {})
        train_transform = get_pytorch_image_transforms(
            image_size=dataset_params["image_size"], is_train=True
        )
        hf_dataset_name = dataset_params.get("name", "nlphuji/flickr30k")
        logger.info(f"Loading '{hf_dataset_name}' dataset...")
        try:
            from datasets import load_dataset

            full_hf_dataset = load_dataset(
                hf_dataset_name,
                name=dataset_params.get("hf_config_name"),
                split="test",
                trust_remote_code=True,
            )
            logger.info(
                f"Loaded HF dataset with {len(full_hf_dataset)} samples."
            )
            full_caption_dataset = ImageCaptionDatasetPT(
                dataset_name=dataset_params.get("name"),
                raw_dataset_input=full_hf_dataset,
                split="all",
                tokenizer=tokenizer,
                image_transform=train_transform,
                max_seq_len=tokenizer_cfg.get("max_seq_len", 50),
                caption_col=dataset_params.get("caption_col", "caption"),
                image_col=dataset_params.get("image_col", "image"),
                max_samples=None,
            )
            if len(full_caption_dataset) == 0:
                raise ValueError("Full caption dataset resulted in 0 samples.")
            from torch.utils.data import random_split

            val_ratio = 0.1
            num_total = len(full_caption_dataset)
            val_len = int(val_ratio * num_total)
            train_len = num_total - val_len
            train_dataset, val_dataset = random_split(
                full_caption_dataset,
                [train_len, val_len],
                generator=None,
            )
            logger.info(
                f"Created train dataset with {len(train_dataset)} samples "
                f"and val dataset with {len(val_dataset)} samples."
            )
        except Exception as e:
            logger.error(
                f"❌ Error during MLX data loading/splitting: {e}",
                exc_info=True,
            )
            train_dataset, val_dataset = None, None
        if (
            not train_dataset
            or not val_dataset
            or len(train_dataset) == 0
            or len(val_dataset) == 0
        ):
            logger.error(
                "❌ MLX dataset loading resulted in empty dataset. Exiting."
            )
            if run:
                run.finish(exit_code=1)
                return
        train_dataloader = get_dataloader(
            train_dataset,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        val_dataloader = get_dataloader(
            val_dataset,
            args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        logger.info("--- Initializing MLX Model ---")
        model_params = full_config.get("model", {})
        encoder_params = full_config.get("encoder", {})
        model = ImageCaptionerMLX(
            vision_encoder_name=encoder_params.get("hf_name", "mlx-vit-base"),
            decoder_vocab_size=_VOCAB_SIZE,
            decoder_embed_dim=model_params.get("decoder_embed_dim", 768),
            decoder_num_heads=model_params.get("decoder_num_heads", 8),
            decoder_ffn_dim=model_params.get("decoder_embed_dim", 768)
            * int(model_params.get("decoder_ffn_dim_ratio", 4.0)),
            decoder_depth=model_params.get("decoder_depth", 6),
            max_seq_len=tokenizer_cfg.get("max_seq_len", 50),
            dropout=model_params.get("dropout", 0.1),
            freeze_encoder=encoder_params.get("freeze", True),
        )
        optimizer = create_optimizer_mlx(
            model,
            optimizer_name=full_config.get("training", {}).get(
                "optimizer_name", "adamw"
            ),
            lr=args.lr,
            weight_decay=args.wd,
        )
        scheduler = create_scheduler_mlx(
            optimizer,
            scheduler_name=full_config.get("training", {}).get(
                "scheduler_name", "cosineannealinglr"
            ),
            total_epochs=args.epochs,
            base_lr=args.lr,
            scheduler_params=full_config.get("training", {}).get(
                "scheduler_params", {}
            ),
        )
        start_epoch = 0
        metrics_history = {
            "avg_train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }
        run_save_path = Path(args.model_save_dir) / run_name
        if args.resume:
            (
                start_epoch,
                metrics_history,
                loaded_model_cfg,
                loaded_ds_cfg,
                loaded_tk_cfg,
            ) = load_checkpoint_mlx(model, run_save_path)
        logger.info(
            f"MLX Model Parameters: {sum(p.size for p in model.parameters() if hasattr(p, 'size'))/1e6:.2f}M trainable"
        )
        logger.info("--- Starting MLX Training ---")
        train_model_mlx(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            target_epochs=args.epochs,
            start_epoch=start_epoch,
            metrics_history=metrics_history,
            model_save_dir=run_save_path,
            config=full_config,
            run_name=run_name,
            lr_scheduler=scheduler,
            save_every=full_config.get("training", {}).get("save_every", 1),
            pad_token_id=tokenizer_cfg.get("pad_token_id", 0),
            is_encoder_decoder=True,
        )

    else:
        logger.error(f"❌ Unsupported framework: {args.framework}")
        if run:
            run.finish(exit_code=1)
        return

    # --- Finalize (Common part, but W&B artifact logging is framework-specific) ---
    logger.info("--- Finalizing Run ---")
    metrics_file = save_metrics(metrics_history, run_save_path)
    plot_file = plot_metrics(metrics_history, run_save_path)

    if run:
        logger.info("☁️ Logging final artifacts to W&B...")
        try:
            artifact_name = f"caption_model_{framework_prefix}_final_{run.id}"
            final_artifact = wandb.Artifact(artifact_name, type="model")
            if args.framework == "pytorch":
                model_weights_file = (
                    run_save_path / "best_model_weights_pt.pth"
                )
                if not model_weights_file.exists():
                    model_weights_file = run_save_path / "model_weights_pt.pth"
            else:
                model_weights_file = (
                    run_save_path / "model_weights.safetensors"
                )

            if model_weights_file.exists():
                final_artifact.add_file(str(model_weights_file))
            else:
                logger.warning(
                    f"Model weights file {model_weights_file} not found for artifact."
                )

            if metrics_file and Path(metrics_file).exists():
                final_artifact.add_file(metrics_file)
            if plot_file and Path(plot_file).exists():
                final_artifact.add_file(plot_file)
            config_in_checkpoint = run_save_path / "model_config.yaml"
            if config_in_checkpoint.exists():
                final_artifact.add_file(str(config_in_checkpoint))
            else:
                final_artifact.add_file(args.config_path)

            run.log_artifact(final_artifact)
            logger.info("  Logged final model, results, and config artifact.")
        except Exception as e:
            logger.error(f"❌ Failed W&B artifact logging: {e}", exc_info=True)
        run.finish()
        logger.info("☁️ W&B run finished.")

    logger.info(f"✅ {args.framework.upper()} Training script completed.")


if __name__ == "__main__":
    from src.data.image_transforms import (
        get_image_transforms as get_pytorch_image_transforms,
    )
    from src.models.encoder_wrapper import DEFAULT_VIT_MODEL

    main()
