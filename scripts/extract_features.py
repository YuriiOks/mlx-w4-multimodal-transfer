# Multimodal Transfer Learning / Image Captioning
# File: scripts/extract_features.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: Script to pre-compute image features for faster training and inference.
# Created: 2025-05-05
# Updated: 2025-05-05

import argparse
import os


def parse_args():
    """Parse command line arguments for feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract image features for faster training and inference"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing images to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="features",
        help="Directory to save extracted features",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing"
    )
    parser.add_argument(
        "--model", type=str, default="resnet50", help="CNN backbone to use"
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable half-precision (float16) for model weights (for compatibility)",
    )
    return parser.parse_args()


def extract_features(args):
    """
    Extract features from images using a pre-trained CNN backbone.

    Args:
        args: Command line arguments
    """
    print(f"Extracting features from images in {args.image_dir}")
    print(f"Using model: {args.model}")
    print(f"Output will be saved to {args.output_dir}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # TODO: Implement feature extraction using MLX
    # 1. Load pre-trained model
    # 2. Process images in batches
    # 3. Save extracted features


if __name__ == "__main__":
    args = parse_args()
    extract_features(args)
