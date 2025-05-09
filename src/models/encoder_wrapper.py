# Multimodal Transfer Learning / Image Captioning
# File: src/models/encoder_wrapper.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: Wraps pre-trained vision encoders (ViT, CLIP) from HF.
# Created: 2025-05-07
# Updated: 2025-05-07

import os

# --- Add project root for imports ---
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils import get_device as get_pytorch_device
from utils import logger

# --- Import MLX if available for conversion ---
try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    logger.warning(
        "⚠️ MLX not available. Feature conversion to MLX will fail."
    )
    MLX_AVAILABLE = False

# --- Import Transformers library ---
try:
    from transformers import AutoModel, AutoProcessor, BatchFeature

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.error(
        "❌ 'transformers' library not found. "
        "Install with 'pip install transformers'."
    )
    TRANSFORMERS_AVAILABLE = False

    # Dummy classes to prevent import errors elsewhere
    class AutoProcessor:
        @staticmethod
        def from_pretrained(name):
            return None

    class AutoModel:
        @staticmethod
        def from_pretrained(name):
            return None

    class BatchFeature(dict):
        pass


# --- Constants for common model names ---
DEFAULT_VIT_MODEL = "google/vit-base-patch16-224-in21k"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def load_vision_encoder_and_processor(
    model_name_or_path: str,
) -> Tuple[Optional[nn.Module], Optional[Any]]:
    """
    Loads a pre-trained vision model (encoder) and its processor from
    Hugging Face Transformers.

    Args:
        model_name_or_path (str): HF model identifier.

    Returns:
        Tuple[Optional[nn.Module], Optional[Any]]:
            - The pre-trained vision model (PyTorch nn.Module).
            - The associated image processor.
            Returns (None, None) on failure.
    """
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    logger.info(
        f"🧠 Loading vision encoder & processor: " f"'{model_name_or_path}'"
    )
    try:
        processor = AutoProcessor.from_pretrained(
            model_name_or_path, use_fast=True
        )
        # For CLIP, use vision_model part
        if "clip" in model_name_or_path.lower():
            model = AutoModel.from_pretrained(model_name_or_path).vision_model
        else:
            model = AutoModel.from_pretrained(model_name_or_path)
        logger.info("✅ Vision encoder and processor loaded successfully.")
        return model, processor
    except Exception as e:
        logger.error(
            f"❌ Failed to load model/processor "
            f"'{model_name_or_path}': {e}",
            exc_info=True,
        )
        return None, None


def get_image_features_pt(
    encoder: nn.Module,
    processor: Any,
    images: Any,  # Accept any type for robust checking
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Extracts image features using a pre-trained PyTorch vision encoder.

    Args:
        encoder (nn.Module): The pre-trained vision model.
        processor (Any): The Hugging Face image processor for the model.
        images (Any): A list of PIL Image objects.
        device (torch.device): The PyTorch device to run inference on.

    Returns:
        Optional[torch.Tensor]: Feature tensor (usually last_hidden_state)
            Shape: (batch_size, sequence_length, hidden_size)
            or None on failure.
    """
    # Robust initial checks
    if encoder is None:
        logger.warning("Encoder is None for PT feature extraction.")
        return None
    if processor is None:
        logger.warning("Processor is None for PT feature extraction.")
        return None
    is_valid_input_list = isinstance(images, list) and len(images) > 0
    if not is_valid_input_list:
        logger.warning(
            f"Input 'images' is not a non-empty list (Type: {type(images)}). Cannot extract PT features."
        )
        return None
    try:
        encoder.eval()
        encoder.to(device)
        # Preprocess images
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs.get("pixel_values")
        if (
            pixel_values is None
            or not isinstance(pixel_values, torch.Tensor)
            or pixel_values.shape[0] == 0
        ):
            logger.warning(
                "Processor did not return valid 'pixel_values' tensor."
            )
            return None
        # Move inputs to device
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = encoder(**inputs)
            # For ViT models: outputs.last_hidden_state
            # For CLIP vision model: outputs.last_hidden_state or outputs.pooler_output
            if hasattr(outputs, "last_hidden_state"):
                features = outputs.last_hidden_state
            elif (
                hasattr(outputs, "image_embeds")
                and "clip" in encoder.config._name_or_path.lower()
            ):
                logger.warning(
                    "CLIP vision_model output structure might vary. "
                    "Prioritizing last_hidden_state."
                )
                features = getattr(
                    outputs,
                    "last_hidden_state",
                    getattr(outputs, "image_embeds", None),
                )
                if features is None:
                    logger.error(
                        "Could not extract suitable features from "
                        "CLIP model output."
                    )
                    return None
            else:
                logger.error(
                    "Could not extract 'last_hidden_state' from "
                    "encoder output."
                )
                return None
        logger.debug(f"Extracted PT features shape: {features.shape}")
        return features
    except Exception as e:
        logger.error(
            "❌ Error during PyTorch image feature extraction: " f"{e}",
            exc_info=True,
        )
        return None


def convert_pt_features_to_mlx(
    pt_tensor: torch.Tensor,
) -> Optional["mx.array"]:
    """
    Converts a PyTorch tensor to an MLX array.

    Args:
        pt_tensor (torch.Tensor): The PyTorch tensor to convert.

    Returns:
        Optional[mx.array]: The converted MLX array, or None on failure.
    """
    if not MLX_AVAILABLE:
        logger.error("❌ MLX not available for tensor conversion.")
        return None
    if not isinstance(pt_tensor, torch.Tensor):
        logger.error("Input is not a PyTorch tensor.")
        return None
    try:
        np_array = pt_tensor.detach().cpu().numpy()
        mlx_array = mx.array(np_array)
        logger.debug(
            f"Converted PT tensor {pt_tensor.shape} to "
            f"MLX array {mlx_array.shape}"
        )
        return mlx_array
    except Exception as e:
        logger.error(
            "❌ Error converting PyTorch tensor to MLX: " f"{e}", exc_info=True
        )
        return None


def get_image_features_mx(
    encoder: Any,  # Can be PT nn.Module or MLX nn.Module
    processor: Any,
    images: List[Image.Image],  # Expect List[PIL] for processor
) -> Optional["mx.array"]:
    """
    Extracts image features using a PyTorch encoder & converting,
    OR using an MLX encoder directly.

    Args:
        encoder: The pre-trained vision model (PT or MLX).
        processor: The associated image processor.
        images (List[Image.Image]): List of PIL images.

    Returns:
        Optional[mx.array]: Feature tensor (last_hidden_state) as MLX array.
    """
    # Robust initial checks
    if encoder is None or processor is None or not images:
        logger.warning(
            "Encoder, processor, or image list is None/empty for MLX feature extraction."
        )
        return None

    # --- MLX-native encoder path ---
    if hasattr(encoder, "__module__") and (
        "mlx.nn" in str(type(encoder).__module__)
        or "src.models.mlx" in str(type(encoder).__module__)
    ):
        logger.debug("Using MLX-native encoder path...")
        try:
            inputs_np = processor(images=images, return_tensors="np")
            pixel_values_np = inputs_np.get("pixel_values")
            if pixel_values_np is None:
                logger.error("MLX processor did not return 'pixel_values'.")
                return None
            pixel_values_mx = mx.array(pixel_values_np)
            logger.debug(f"MLX pixel values shape: {pixel_values_mx.shape}")
            outputs = encoder(pixel_values_mx)
            # Extract last_hidden_state (assuming similar output structure)
            if hasattr(outputs, "last_hidden_state"):
                features_mlx = outputs.last_hidden_state
                logger.debug(
                    f"Extracted MLX features shape: {features_mlx.shape}"
                )
                return features_mlx
            else:
                logger.error(
                    "Could not extract 'last_hidden_state' from MLX encoder output."
                )
                return None
        except Exception as e:
            logger.error(
                f"❌ Error during MLX-native feature extraction: {e}",
                exc_info=True,
            )
            return None

    # --- PyTorch Encoder Path (keep as before) ---
    elif isinstance(encoder, nn.Module):
        logger.debug("Using PyTorch encoder for MLX feature path...")
        pt_device = get_pytorch_device()
        features_pt = get_image_features_pt(
            encoder, processor, images, pt_device
        )
        if features_pt is None:
            return None
        features_mlx = convert_pt_features_to_mlx(features_pt)
        if features_mlx is None:
            return None
        return features_mlx

    else:
        logger.error(
            f"Unsupported encoder type in get_image_features_mx: {type(encoder)}"
        )
        return None


# --- MLX Vision Encoder Loader ---
def load_mlx_vision_encoder_and_processor(model_name_or_path: str):
    """
    Loads an MLX-native vision encoder and a basic processor.
    Args:
        model_name_or_path (str): Model name (e.g., 'mlx-vit-base').
    Returns:
        Tuple[ViTMLX, Callable]: MLX vision encoder and processor.
    """
    from src.models.mlx.vision_encoder import ViTMLX

    if model_name_or_path.lower() in ["mlx-vit-base", "mlx-vit"]:
        encoder = ViTMLX()

        def processor(images, return_tensors="np"):
            # images: list of PIL.Image
            import numpy as np

            arrs = []
            for img in images:
                arr = (
                    np.array(img.resize((224, 224))).astype(np.float32) / 255.0
                )
                if arr.shape[-1] == 3:
                    arr = arr.transpose(2, 0, 1)  # HWC to CHW
                arrs.append(arr)
            arrs = np.stack(arrs)
            return {"pixel_values": arrs}

        return encoder, processor
    return None, None


# --- Test Block ---
if __name__ == "__main__":
    """
    Test block for encoder wrapper functionality.
    Loads ViT and CLIP models, extracts features, and tests MLX conversion.
    """
    logger.info("🧪 Testing Encoder Wrapper...")

    # Dummy device for PyTorch
    pt_device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"PyTorch using device: {pt_device}")

    # Test with ViT
    logger.info("\n--- Testing with ViT ---")
    vit_encoder, vit_processor = load_vision_encoder_and_processor(
        DEFAULT_VIT_MODEL
    )
    if vit_encoder and vit_processor:
        dummy_images_pil = [Image.new("RGB", (224, 224)) for _ in range(2)]
        features_pt_vit = get_image_features_pt(
            vit_encoder, vit_processor, dummy_images_pil, pt_device
        )
        if features_pt_vit is not None:
            logger.info(f"ViT PT features shape: {features_pt_vit.shape}")
            # Expected for ViT base: (batch_size, num_patches+1=197, hidden=768)
            assert features_pt_vit.shape[0] == 2
            assert features_pt_vit.shape[2] == 768
            logger.info("✅ ViT feature extraction (PT) seems OK.")
            features_mlx_vit = convert_pt_features_to_mlx(features_pt_vit)
            if features_mlx_vit is not None:
                logger.info(
                    f"ViT MLX features shape: {features_mlx_vit.shape}"
                )
                assert features_mlx_vit.shape == features_pt_vit.shape
                logger.info("✅ ViT PT to MLX conversion seems OK.")
        else:
            logger.error("❌ Failed to get ViT features.")
    else:
        logger.error("❌ Failed to load ViT model/processor.")

    # Test with CLIP
    logger.info("\n--- Testing with CLIP ---")
    clip_encoder, clip_processor = load_vision_encoder_and_processor(
        DEFAULT_CLIP_MODEL
    )
    if clip_encoder and clip_processor:
        dummy_images_pil = [Image.new("RGB", (224, 224)) for _ in range(2)]
        features_pt_clip = get_image_features_pt(
            clip_encoder, clip_processor, dummy_images_pil, pt_device
        )
        if features_pt_clip is not None:
            logger.info(f"CLIP PT features shape: {features_pt_clip.shape}")
            # Expected for CLIP ViT-B/32: (batch, num_patches+1=50, hidden=768)
            assert features_pt_clip.shape[0] == 2
            assert features_pt_clip.shape[2] == 768
            logger.info("✅ CLIP feature extraction (PT) seems OK.")
            features_mlx_clip = convert_pt_features_to_mlx(features_pt_clip)
            if features_mlx_clip is not None:
                logger.info(
                    f"CLIP MLX features shape: " f"{features_mlx_clip.shape}"
                )
                assert features_mlx_clip.shape == features_pt_clip.shape
                logger.info("✅ CLIP PT to MLX conversion seems OK.")
        else:
            logger.error("❌ Failed to get CLIP features.")
    else:
        logger.error("❌ Failed to load CLIP model/processor.")

    logger.info("\n✅ Encoder Wrapper test block finished.")
