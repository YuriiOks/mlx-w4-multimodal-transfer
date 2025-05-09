# Multimodal Transfer Learning / Image Captioning
# File: src/models/pytorch/caption_model.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: PyTorch Encoder-Decoder model for Image Captioning.
# Created: 2025-05-07
# Updated: 2025-05-07

import os

# --- Add project root for imports ---
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from src.models.encoder_wrapper import (
    DEFAULT_VIT_MODEL,
    get_image_features_pt,
    load_vision_encoder_and_processor,
)
from src.models.pytorch.modules import DecoderLayerPT
from utils import get_device, logger


class PositionalEncodingPT(nn.Module):
    """
    Fixed sinusoidal positional encoding for transformer models.

    Args:
        embed_dim (int): Embedding dimension.
        max_len (int): Maximum sequence length.
        dropout (float): Dropout probability.
    """

    def __init__(
        self, embed_dim: int, max_len: int = 512, dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2)
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape
                [batch_size, seq_len, embedding_dim].

        Returns:
            torch.Tensor: Tensor with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ImageCaptionerPT(nn.Module):
    """
    Image captioning model using vision encoder and transformer decoder.

    Args:
        vision_encoder_name (str): Name of vision encoder model.
        decoder_vocab_size (int): Decoder vocabulary size.
        decoder_embed_dim (int): Decoder embedding dimension.
        decoder_num_heads (int): Number of decoder attention heads.
        decoder_ffn_dim (int): Decoder feedforward network dimension.
        decoder_depth (int): Number of decoder layers.
        max_seq_len (int): Maximum sequence length.
        dropout (float): Dropout probability.
        freeze_encoder (bool): Whether to freeze vision encoder.
        device (Optional[torch.device]): Device to use.
    """

    def __init__(
        self,
        vision_encoder_name: str,
        decoder_vocab_size: int,
        decoder_embed_dim: int,
        decoder_num_heads: int,
        decoder_ffn_dim: int,
        decoder_depth: int,
        max_seq_len: int,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or get_device()
        (
            self.vision_encoder,
            self.image_processor,
        ) = load_vision_encoder_and_processor(vision_encoder_name)
        if self.vision_encoder is None or self.image_processor is None:
            raise RuntimeError(
                f"Failed to load vision encoder/processor: "
                f"{vision_encoder_name}"
            )
        self.vision_encoder.to(self.device)
        if freeze_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()
            logger.info(f"🧊 Vision encoder '{vision_encoder_name}' is FROZEN.")
        else:
            logger.info(
                f"🔥 Vision encoder '{vision_encoder_name}' is TRAINABLE."
            )
        self.decoder_embed = nn.Embedding(
            decoder_vocab_size, decoder_embed_dim
        )
        self.decoder_pos_embed = PositionalEncodingPT(
            decoder_embed_dim, max_seq_len, dropout
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayerPT(
                    embed_dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    ffn_dim=decoder_ffn_dim,
                    dropout=dropout,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.output_head = nn.Linear(decoder_embed_dim, decoder_vocab_size)
        self.max_seq_len = max_seq_len
        logger.info(
            f"🧠 ImageCaptionerPT initialized. Decoder depth: "
            f"{decoder_depth}"
        )

    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """
        Generate a causal mask for decoder self-attention.

        Args:
            size (int): Sequence length.

        Returns:
            torch.Tensor: Causal mask of shape (size, size).
        """
        mask = torch.triu(torch.ones(size, size) * float("-inf"), diagonal=1)
        return mask.to(self.device)

    def encode(self, images: List[Image.Image]) -> Optional[torch.Tensor]:
        """
        Encode a list of PIL images into feature tensors.

        Args:
            images (List[Image.Image]): List of images.

        Returns:
            Optional[torch.Tensor]: Encoded image features or None.
        """
        if not images:
            return None
        return get_image_features_pt(
            self.vision_encoder, self.image_processor, images, self.device
        )

    def decode(
        self,
        tgt_token_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_causal_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode target tokens using encoder memory.

        Args:
            tgt_token_ids (torch.Tensor): Target token ids (B, T_tgt).
            memory (torch.Tensor): Encoder memory (B, S_src, D_enc).
            tgt_causal_mask (Optional[torch.Tensor]): Causal mask (T_tgt, T_tgt).
            memory_key_padding_mask (Optional[torch.Tensor]):
                Encoder padding mask (B, S_src).

        Returns:
            torch.Tensor: Decoder output embeddings (B, T_tgt, D_dec).
        """
        tgt_embed = self.decoder_embed(tgt_token_ids)
        tgt_with_pos = self.decoder_pos_embed(tgt_embed)
        x = tgt_with_pos
        for layer in self.decoder_layers:
            x = layer(
                tgt=x,
                memory=memory,
                tgt_mask=tgt_causal_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return x

    def forward(
        self,
        images: List[Image.Image],
        tgt_token_ids: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training with teacher forcing.

        Args:
            images (List[Image.Image]): Batch of PIL images.
            tgt_token_ids (torch.Tensor): Target token sequences (B, T).
            memory_key_padding_mask (Optional[torch.Tensor]):
                Encoder output padding mask (B, S_src).

        Returns:
            torch.Tensor: Logits over vocab (B, T, VocabSize).
        """
        memory = self.encode(images)
        if memory is None:
            bs = tgt_token_ids.shape[0]
            seq_len = tgt_token_ids.shape[1]
            dummy_logits = torch.zeros(
                bs, seq_len, self.output_head.out_features, device=self.device
            )
            logger.error("Image encoding failed in forward pass!")
            return dummy_logits
        causal_mask = self._generate_causal_mask(tgt_token_ids.size(1))
        decoder_output_embeds = self.decode(
            tgt_token_ids,
            memory,
            tgt_causal_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.output_head(decoder_output_embeds)
        return logits


if __name__ == "__main__":
    logger.info("🧪 Testing PyTorch ImageCaptionerPT Model...")
    test_device = get_device()
    _B = 2
    _MAX_SEQ_LEN = 20
    _VOCAB_SIZE = 50261  # From GPT2 tokenizer (example)
    _ENC_FEATURE_DIM = 768  # Output from ViT/CLIP base encoder
    _DEC_EMBED_DIM = 768  # <--- MAKE THIS MATCH _ENC_FEATURE_DIM
    _DEC_HEADS = 8  # Match config
    _DEC_FFN_DIM = _DEC_EMBED_DIM * 4  # Match config (using ratio)
    _DEC_DEPTH = 3  # Match config
    _VISION_ENCODER_NAME = DEFAULT_VIT_MODEL

    dummy_pil_images_test = [Image.new("RGB", (224, 224)) for _ in range(_B)]
    dummy_tgt_ids_test = torch.randint(
        0, _VOCAB_SIZE, (_B, _MAX_SEQ_LEN - 1), dtype=torch.long
    ).to(test_device)
    try:
        logger.info("Instantiating ImageCaptionerPT...")
        caption_model = ImageCaptionerPT(
            vision_encoder_name=_VISION_ENCODER_NAME,
            decoder_vocab_size=_VOCAB_SIZE,
            decoder_embed_dim=_DEC_EMBED_DIM,  # Now 768
            decoder_num_heads=_DEC_HEADS,
            decoder_ffn_dim=_DEC_FFN_DIM,  # Based on 768 * 4
            decoder_depth=_DEC_DEPTH,
            max_seq_len=_MAX_SEQ_LEN,
            dropout=0.1,
            freeze_encoder=True,
            device=test_device,
        ).to(test_device)
        caption_model.eval()
        logger.info("Testing forward pass...")
        output_logits = caption_model(
            dummy_pil_images_test, dummy_tgt_ids_test
        )
        logger.info(f"Output logits shape: {output_logits.shape}")
        assert output_logits.shape == (_B, _MAX_SEQ_LEN - 1, _VOCAB_SIZE)
        logger.info("✅ ImageCaptionerPT forward pass test successful.")
    except Exception as e:
        logger.error(f"❌ ImageCaptionerPT test failed: {e}", exc_info=True)
    logger.info("\n✅ PyTorch Caption Model test block finished.")
