# Multimodal Transfer Learning / Image Captioning
# File: src/data/dataloader.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: Creates PyTorch DataLoaders with custom collate_fn.
# Created: 2025-05-06
# Updated: 2025-05-06

import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# --- Add project root for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils import logger

try:
    from src.data.image_datasets import (  # Flag from image_datasets
        HF_DATASETS_AVAILABLE,
    )
except ImportError:
    HF_DATASETS_AVAILABLE = False
try:
    from src.common.tokenizer import (
        TOKENIZER_AVAILABLE as TOKENIZER_IMPORTED_OK_FOR_DL,  # Flag from tokenizer
    )
except ImportError:
    TOKENIZER_IMPORTED_OK_FOR_DL = False


def collate_fn_captioning(
    batch: List[Dict[str, Any]]
) -> Dict[str, torch.Tensor]:
    """
    Custom collate_fn for image captioning.
    Pads input_ids and attention_masks. Stacks pixel_values.

    Args:
        batch (List[Dict[str, Any]]): A list of dictionaries from the Dataset.
            Each dict: {"pixel_values": tensor, "input_ids": tensor, "attention_mask": tensor}

    Returns:
        Dict[str, torch.Tensor]: A batch ready for the model.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack(
        [item["input_ids"] for item in batch]
    )  # Already padded by tokenizer
    attention_masks = torch.stack(
        [item["attention_mask"] for item in batch]
    )  # Already padded

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
    }


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Creates a PyTorch DataLoader with the custom collate_fn."""
    logger.info(
        f"📦 Creating DataLoader: BS={batch_size}, Shuffle={shuffle}, "
        f"NW={num_workers}, PinMem={pin_memory}"
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_captioning,  # Use custom collate
    )


# Inside src/data/dataloader.py

# ... (Keep existing imports and functions) ...

# --- Test Block ---
if __name__ == "__main__":
    logger.info(
        "🧪 Testing DataLoader Utils with DUMMY DATA for Image Captioning..."
    )

    # --- Add project root for config/module imports if run directly ---
    # ... (sys.path logic as in the previous file) ...
    script_dir_dl = os.path.dirname(os.path.abspath(__file__))
    project_root_dl = Path(script_dir_dl).parent.parent
    if str(project_root_dl) not in sys.path:
        sys.path.insert(0, str(project_root_dl))
    # --- End Add project root ---

    can_test_dataloader = True
    test_tokenizer = None
    test_caption_ds = None
    max_len = 50
    img_size = 224

    try:
        from src.common.tokenizer import TOKENIZER_AVAILABLE, init_tokenizer
        from src.data.image_datasets import (  # Import necessary transform
            ImageCaptionDatasetPT,
            get_image_transforms,
        )
        from utils.config import load_config

        if not TOKENIZER_AVAILABLE:
            raise ImportError(
                "Tokenizer utils not available for dataloader test."
            )

        logger.info("Loading config for dataloader test...")
        cfg = load_config(project_root_dl / "config.yaml")
        if cfg is None:
            logger.error(
                "Failed to load config, cannot run dataloader test properly."
            )
            can_test_dataloader = False
        else:
            tokenizer_cfg = cfg.get("tokenizer", {})
            dataset_cfg = cfg.get("dataset", {})
            tok_name = tokenizer_cfg.get("hf_model_name", "gpt2")
            max_len = dataset_cfg.get("max_seq_len", 50)
            img_size = dataset_cfg.get("image_size", 224)

            logger.info("Initializing tokenizer for dataloader test...")
            test_tokenizer = init_tokenizer(tok_name)

    except Exception as e:
        logger.error(f"Error during dataloader test setup: {e}", exc_info=True)
        can_test_dataloader = False

    if can_test_dataloader and test_tokenizer:
        logger.info("Using dummy data for dataloader test.")
        # Create dummy data
        dummy_pil_images = [
            Image.new(
                "RGB",
                (img_size, img_size),
                color=(random.randint(0, 255), 0, 0),
            )
            for _ in range(8)
        ]
        dummy_captions = [
            f"Another dummy caption example {i}" for i in range(8)
        ]
        dummy_hf_ds_list = [
            {"image": img, "caption": [cap]}
            for img, cap in zip(dummy_pil_images, dummy_captions)
        ]

        class DummyHFDatasetWrapper:
            def __init__(self, data_list):
                self.data_list = data_list

            def __len__(self):
                return len(self.data_list)

            def __getitem__(self, idx):
                return self.data_list[idx]

        dummy_hf_dataset_obj = DummyHFDatasetWrapper(dummy_hf_ds_list)

        test_caption_ds = ImageCaptionDatasetPT(
            dataset_name="dummy_dataloader_test",
            raw_dataset_input=dummy_hf_dataset_obj,  # Pass the dummy structure
            split="train",
            tokenizer=test_tokenizer,
            image_transform=get_image_transforms(
                image_size=img_size, is_train=False
            ),  # Use imported transform
            max_seq_len=max_len,
            caption_col="caption",
            image_col="image",
            max_samples=None,  # Use all dummy samples
        )

        if test_caption_ds and len(test_caption_ds) > 0:
            logger.info(
                f"Dummy test dataset created with {len(test_caption_ds)} samples."
            )
            test_loader = get_dataloader(
                test_caption_ds, batch_size=4, shuffle=False
            )
            batch = next(iter(test_loader))
            logger.info(
                f"Batch pixel_values shape: {batch['pixel_values'].shape}"
            )
            logger.info(f"Batch input_ids shape: {batch['input_ids'].shape}")
            logger.info(
                f"Batch attention_mask shape: {batch['attention_mask'].shape}"
            )

            assert batch["pixel_values"].shape == (4, 3, img_size, img_size)
            assert batch["input_ids"].shape == (4, max_len)
            assert batch["attention_mask"].shape == (4, max_len)
            logger.info(
                "✅ DataLoader with collate_fn using dummy data seems OK."
            )
        else:
            logger.warning(
                "Dummy test dataset for dataloader was empty or failed to create."
            )
    else:
        logger.error(
            "Skipping dataloader test due to setup issues (config or tokenizer)."
        )
    logger.info("✅ DataLoader test block finished.")
