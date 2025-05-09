# Multimodal Transfer Learning / Image Captioning
# File: src/data/image_transforms.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: Image transformations for preprocessing and augmentation.
# Created: 2025-05-06
# Updated: 2025-05-06

from PIL import Image  # For type hinting
from torchvision import transforms

# --- Default Image Preprocessing from common Vision Encoders ---
# Example: CLIP uses specific resize, center crop, and normalization
# ViT from HF transformers also uses a specific processor.


def get_image_transforms(
    image_size: int = 224,  # Typical for ViT/CLIP
    is_train: bool = False,  # Apply augmentation only for training
) -> transforms.Compose:
    """
    Returns a composition of torchvision transforms for image preprocessing.
    Uses typical settings for ViT/CLIP models.
    Includes basic augmentation if is_train is True.

    Args:
        image_size (int): Target size for the image (square).
        is_train (bool): If True, adds augmentation.

    Returns:
        transforms.Compose: The transform pipeline.
    """
    # These are common for CLIP/ViT. Adjust if using a different encoder.
    mean = [0.48145466, 0.4578275, 0.40821073]  # CLIP mean
    std = [0.26862954, 0.26130258, 0.27577711]  # CLIP std

    # Or ViT mean/std from HF processors:
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]

    transform_list = []
    if is_train:
        # Augmentation: RandomResizedCrop is very common and effective
        transform_list.extend(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),  # ViT scale range
                    ratio=(0.75, 1.333),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # Add more from your augmentation.py if desired, e.g., ColorJitter
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        )
    else:
        # Validation/Test: Resize and CenterCrop
        # CLIP often uses smaller resize then center crop
        transform_list.extend(
            [
                transforms.Resize(
                    image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(image_size),
            ]
        )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transforms.Compose(transform_list)


# --- Test Block ---
if __name__ == "__main__":
    # Add project root for logger if run directly
    import os
    import sys
    from pathlib import Path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = Path(script_dir).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from utils import logger

    logger.info("🧪 Testing Image Transforms...")
    train_transforms = get_image_transforms(image_size=224, is_train=True)
    val_transforms = get_image_transforms(image_size=224, is_train=False)
    logger.info(f"Train Transforms: {train_transforms}")
    logger.info(f"Validation Transforms: {val_transforms}")

    # Create a dummy PIL image to test
    try:
        dummy_pil_img = Image.new("RGB", (300, 400), color="red")
        transformed_train_img = train_transforms(dummy_pil_img)
        transformed_val_img = val_transforms(dummy_pil_img)
        logger.info(
            f"Dummy train img transformed shape: {transformed_train_img.shape}"
        )
        logger.info(
            f"Dummy val img transformed shape: {transformed_val_img.shape}"
        )
        assert transformed_train_img.shape == (3, 224, 224)
        assert transformed_val_img.shape == (3, 224, 224)
        logger.info("✅ Image transforms seem OK.")
    except Exception as e:
        logger.error(f"❌ Error testing transforms: {e}")
