# Multimodal Transfer Learning / Image Captioning
# File: src/data/image_datasets.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: PyTorch Dataset classes for image captioning.
# Created: 2025-05-06
# Updated: 2025-05-06

import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

# --- Add project root for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.common.tokenizer import (  # Assuming AutoTokenizer imported there
    tokenize_captions,
)
from src.data.image_transforms import get_image_transforms
from utils import logger

try:
    from src.common.tokenizer import (
        TOKENIZER_AVAILABLE as TOKENIZER_IMPORTED_OK,  # Use an alias
    )
except ImportError:
    TOKENIZER_IMPORTED_OK = False  # Define if import fails

try:
    from datasets import load_dataset  # Hugging Face datasets library

    HF_DATASETS_AVAILABLE = True
except ImportError:
    logger.error(
        "❌ Hugging Face 'datasets' library not found. Install with 'pip install datasets'."
    )
    HF_DATASETS_AVAILABLE = False


class ImageCaptionDatasetPT(Dataset):
    """
    PyTorch Dataset for Image Captioning.
    Loads data using Hugging Face datasets (e.g., Flickr30k).

    Args:
        dataset_name (str): Name of the HF dataset (e.g., "flickr30k").
        dataset_config_name (Optional[str]): Specific config for HF dataset.
        split (str): "train", "validation", or "test".
        tokenizer (AutoTokenizer): Pre-initialized tokenizer.
        image_transform (Callable): Transform pipeline for images.
        max_seq_len (int): Max length for tokenized captions.
        caption_col (str): Name of the column containing captions list.
        image_col (str): Name of the column containing PIL Image objects.
        max_samples (Optional[int]): Max samples to use (for debugging).
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config_name: Optional[str] = None,
        split: str = "train",
        tokenizer: Any = None,  # Should be AutoTokenizer
        image_transform: Optional[Callable] = None,
        max_seq_len: int = 50,
        caption_col: str = "caption",  # Adjust if dataset has different key
        image_col: str = "image",  # Adjust if dataset has different key
        max_samples: Optional[int] = None,
        raw_dataset_input: Optional[Any] = None,  # <-- new argument
    ):
        """
        Initialize dataset.

        Args:
            dataset_name (str): Name of dataset on HF Hub or descriptive name for logging.
            split (str): Data split ('train', 'validation', 'test').
            tokenizer: HF tokenizer for captions.
            image_transform: PyTorch transforms for images.
            max_seq_len (int): Maximum caption length (including special tokens).
            caption_col (str): Name of caption column in dataset.
            image_col (str): Name of image column in dataset.
            max_samples (int, optional): Max samples to load (for debugging).
            raw_dataset_input: Optional pre-loaded dataset to use instead of loading from Hub.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.image_transform = image_transform or get_image_transforms(
            is_train=(split == "train")
        )
        self.max_seq_len = max_seq_len
        self.caption_col = caption_col
        self.image_col = image_col
        self.dataset_name = dataset_name  # For logging
        self.split = split  # For logging
        self.max_samples = max_samples
        self.data = []

        # Check if we're using a pre-loaded dataset
        if raw_dataset_input is not None:
            logger.info(
                f"💾 Using pre-loaded dataset with {len(raw_dataset_input)} samples for '{split}' split"
            )
            self.raw_dataset = raw_dataset_input
        else:
            # Load dataset from HF Hub
            if not HF_DATASETS_AVAILABLE:
                logger.error(
                    "Hugging Face 'datasets' library is not available but is required for loading HF datasets."
                )
                # self.data remains empty, len(self) will be 0
                return

            logger.info(
                f"💾 Loading HF dataset '{dataset_name}' (config: {dataset_config_name}) for split '{split}'..."
            )
            try:
                self.raw_dataset = load_dataset(
                    dataset_name,
                    name=dataset_config_name,
                    split=split,
                    trust_remote_code=True,
                )
                logger.info(
                    f"✅ Loaded {len(self.raw_dataset)} samples from '{dataset_name}' ({split})"
                )
            except Exception as e:
                logger.error(
                    f"❌ Failed to load or process HF dataset '{dataset_name}': {e}"
                )
                self.raw_dataset = []  # Empty list to prevent further errors
                return

        # Limit dataset size for debugging if needed
        if (
            max_samples
            and max_samples > 0
            and len(self.raw_dataset) > max_samples
        ):
            logger.info(
                f"🔍 Limiting dataset to {max_samples} samples (debug mode)"
            )
            if isinstance(self.raw_dataset, list):
                self.raw_dataset = self.raw_dataset[:max_samples]
            else:
                # For datasets.Dataset objects
                self.raw_dataset = self.raw_dataset.select(range(max_samples))

        # Process dataset
        self._process_dataset()

    def _process_dataset(self):
        """
        Process the raw dataset to extract image-caption pairs.
        Populates the self.data attribute with processed items.
        """
        if self.raw_dataset is None:
            logger.warning(
                "No raw dataset found. Ensure dataset is loaded before processing."
            )
            return

        logger.info(
            f"🔄 Processing {len(self.raw_dataset)} raw items to extract image-caption pairs..."
        )
        processed_count = 0
        for item in self.raw_dataset:
            img = item.get(self.image_col)

            # Be more flexible with caption key if it's a common alternative
            captions_data = item.get(self.caption_col)
            if captions_data is None:  # Try alternative if primary is missing
                if self.caption_col == "captions" and "caption" in item:
                    captions_data = item.get("caption")
                    logger.debug(
                        "Using alternative 'caption' column instead of 'captions'"
                    )
                elif self.caption_col == "caption" and "captions" in item:
                    captions_data = item.get("captions")
                    logger.debug(
                        "Using alternative 'captions' column instead of 'caption'"
                    )

            if img is None:
                logger.warning(
                    f"Skipping item: missing '{self.image_col}' key. Item: {item}"
                )
                continue
            if captions_data is None:
                logger.warning(
                    f"Skipping item: missing '{self.caption_col}' key. Item keys: {list(item.keys())}"
                )
                continue

            # Process the captions (handle both list and string formats)
            if not isinstance(captions_data, list):
                captions = [str(captions_data)]
            else:
                captions = [str(cap) for cap in captions_data]

            for cap_text in captions:
                self.data.append({"image": img, "caption_text": cap_text})
                processed_count += 1
                if (
                    self.max_samples is not None
                    and self.max_samples > 0
                    and processed_count >= self.max_samples
                ):
                    break
            if (
                self.max_samples is not None
                and self.max_samples > 0
                and processed_count >= self.max_samples
            ):
                break

        logger.info(
            f"✅ Processing complete. Total items (image-caption pairs): {len(self.data)}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        image_pil = item["image"]
        caption_text = item["caption_text"]

        # Apply image transformations
        image_tensor = self.image_transform(
            image_pil.convert("RGB")
        )  # Ensure RGB

        # Tokenize caption (single caption string)
        # tokenize_captions expects a list of strings
        tokenized_output = tokenize_captions(
            [caption_text], self.tokenizer, self.max_seq_len
        )
        # Extract the single tokenized sequence and mask
        input_ids = torch.tensor(tokenized_output["input_ids"][0])
        attention_mask = torch.tensor(tokenized_output["attention_mask"][0])

        return {
            "pixel_values": image_tensor,
            "input_ids": input_ids,  # For decoder target
            "attention_mask": attention_mask,  # For decoder target
        }


# --- Test Block ---
if __name__ == "__main__":
    logger.info("🧪 Testing ImageCaptionDatasetPT with DUMMY DATA...")

    # --- Add project root for config if run directly ---
    script_dir_ds = os.path.dirname(os.path.abspath(__file__))
    project_root_ds = Path(script_dir_ds).parent.parent
    if str(project_root_ds) not in sys.path:
        sys.path.insert(0, str(project_root_ds))
    # --- End Add project root ---

    # Import what's needed for the test, define fallbacks if necessary
    test_tokenizer = None
    max_len = 50
    img_size = 224

    try:
        from src.common.tokenizer import TOKENIZER_AVAILABLE, init_tokenizer
        from utils.config import load_config  # Ensure this is available

        if not TOKENIZER_AVAILABLE:
            raise ImportError("Tokenizer utils not available.")

        cfg = load_config(project_root_ds / "config.yaml") or {}
        tokenizer_name = cfg.get("tokenizer", {}).get("hf_model_name", "gpt2")
        # Use a phase-agnostic or specific max_len/img_size for testing
        max_len = cfg.get("dataset", {}).get("max_seq_len", 50)
        img_size = cfg.get("dataset", {}).get(
            "image_size", 224
        )  # Default if not phase specific

        test_tokenizer = init_tokenizer(tokenizer_name)

    except Exception as e:
        logger.error(
            f"Failed to load config or initialize tokenizer for test: {e}",
            exc_info=True,
        )
        # test_tokenizer will remain None

    if test_tokenizer:
        logger.info("Using dummy data for ImageCaptionDatasetPT test.")

        # Create dummy data
        dummy_pil_images = [
            Image.new(
                "RGB",
                (img_size, img_size),
                color=(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ),
            )
            for _ in range(8)
        ]
        dummy_captions = [
            f"This is a dummy caption number {i} for testing."
            for i in range(8)
        ]
        # Structure to mimic what ImageCaptionDatasetPT expects from a raw dataset
        dummy_hf_ds_list = [
            {"image": img, "caption": [cap]}
            for img, cap in zip(dummy_pil_images, dummy_captions)
        ]  # Note: 'caption' here to match default

        class DummyHFDatasetWrapper:  # Simple wrapper
            def __init__(self, data_list):
                self.data_list = data_list

            def __len__(self):
                return len(self.data_list)

            def __getitem__(self, idx):
                return self.data_list[idx]

        dummy_hf_dataset_obj = DummyHFDatasetWrapper(dummy_hf_ds_list)

        test_ds_instance = ImageCaptionDatasetPT(
            dataset_name="dummy_for_test",  # Indicate it's not a real HF dataset
            raw_dataset_input=dummy_hf_dataset_obj,  # Pass the dummy structure
            split="train",  # Split type is less critical for this dummy test
            tokenizer=test_tokenizer,
            image_transform=get_image_transforms(
                image_size=img_size, is_train=False
            ),
            max_seq_len=max_len,
            caption_col="caption",  # Matches dummy_hf_ds_list structure
            image_col="image",  # Matches dummy_hf_ds_list structure
            max_samples=None,  # Use all dummy samples
        )

        if test_ds_instance and len(test_ds_instance) > 0:
            sample = test_ds_instance[0]
            logger.info(
                f"Sample image tensor shape: {sample['pixel_values'].shape}"
            )
            logger.info(f"Sample input_ids shape: {sample['input_ids'].shape}")
            logger.info(
                f"Sample attention_mask shape: {sample['attention_mask'].shape}"
            )
            logger.info(f"Sample input_ids: {sample['input_ids'].tolist()}")
            logger.info(
                f"Decoded sample: {test_tokenizer.decode(sample['input_ids'])}"
            )
            assert sample["pixel_values"].shape == (3, img_size, img_size)
            assert sample["input_ids"].shape == (max_len,)
            logger.info("✅ ImageCaptionDatasetPT with dummy data seems OK.")
        else:
            logger.warning(
                "Test dataset with dummy data is empty or failed to create."
            )
    else:
        logger.error(
            "Tokenizer failed to initialize, cannot test ImageCaptionDatasetPT."
        )
    logger.info("✅ ImageCaptionDatasetPT test block finished.")
