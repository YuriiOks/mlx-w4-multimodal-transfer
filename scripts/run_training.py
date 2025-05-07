# Multimodal Transfer Learning / Image Captioning
# File: scripts/run_training.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: Main training script for image captioning models.
# Created: 2025-05-05
# Updated: 2025-05-05

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train image captioning model"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    # TODO: Implement training logic
    print(f"Training with config: {args.config}")
    print(f"Output will be saved to: {args.output_dir}")
