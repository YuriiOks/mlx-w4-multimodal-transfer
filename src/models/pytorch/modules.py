# Multimodal Transfer Learning / Image Captioning
# File: src/models/pytorch/modules.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: PyTorch Transformer Decoder building blocks.
# Created: 2025-05-07
# Updated: 2025-05-07

import os

# --- Add project root for logger ---
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(
    script_dir
).parent.parent.parent  # src/models/pytorch -> src/models -> src -> root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from utils import logger


# --- 1. Multi-Head Attention (PyTorch Wrapper) ---
class AttentionPT(nn.Module):
    """
    PyTorch Multi-Head Attention wrapper for self-attention (masked)
    and cross-attention. Uses nn.MultiheadAttention.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize AttentionPT module.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # Expects (Batch, Seq, Dim)
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention.

        Args:
            query (Tensor): Query tensor (B, T, D_q).
            key (Tensor): Key tensor (B, S, D_k).
            value (Tensor): Value tensor (B, S, D_v).
            attn_mask (Tensor, optional): Prevents attention to future tokens.
            key_padding_mask (Tensor, optional): Prevents attention to padded tokens.

        Returns:
            Tensor: Output tensor (B, T, D_q).
        """
        attn_output, _ = self.mha(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return attn_output


# --- 2. Feed-Forward Network (PyTorch) ---
class FeedForwardPT(nn.Module):
    """
    Standard two-layer feed-forward network for Transformers.
    """

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        """
        Initialize FeedForwardPT module.

        Args:
            embed_dim (int): Embedding dimension.
            ffn_dim (int): Feed-forward hidden dimension.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feed-forward network.

        Args:
            x (Tensor): Input tensor (B, T, D).

        Returns:
            Tensor: Output tensor (B, T, D).
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


# --- 3. Transformer Decoder Layer (PyTorch) ---
class DecoderLayerPT(nn.Module):
    """
    A single Transformer Decoder layer.
    Consists of masked self-attention, cross-attention, and a feed-forward network.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        """
        Initialize DecoderLayerPT module.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            ffn_dim (int): Feed-forward hidden dimension.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.masked_self_attn = AttentionPT(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = AttentionPT(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = FeedForwardPT(embed_dim, ffn_dim, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for a single Transformer decoder layer.

        Args:
            tgt (Tensor): Target sequence (decoder input) (B, T, D).
            memory (Tensor): Encoder output (B, S, D).
            tgt_mask (Tensor, optional): Causal mask for self-attention.
            memory_key_padding_mask (Tensor, optional): Padding mask for encoder memory.

        Returns:
            Tensor: Output tensor (B, T, D).
        """
        tgt_norm = self.norm1(tgt)
        self_attn_out = self.masked_self_attn(
            query=tgt_norm, key=tgt_norm, value=tgt_norm, attn_mask=tgt_mask
        )
        tgt = tgt + self.dropout1(self_attn_out)

        tgt_norm = self.norm2(tgt)
        cross_attn_out = self.cross_attn(
            query=tgt_norm,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(cross_attn_out)

        tgt_norm = self.norm3(tgt)
        ffn_out = self.ffn(tgt_norm)
        tgt = tgt + self.dropout3(ffn_out)

        return tgt


# --- Test Block ---
if __name__ == "__main__":
    logger.info("🧪 Testing PyTorch Decoder Modules...")
    device_test = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    B = 4  # Batch size
    T = 20  # Target sequence length (decoder input)
    S = 50  # Source sequence length (encoder output)
    D = 256  # Embedding dimension
    H = 8  # Number of heads
    FFN_D = D * 4  # Feed-forward hidden dimension

    # Dummy inputs
    dummy_tgt = torch.randn(B, T, D).to(device_test)
    dummy_memory = torch.randn(B, S, D).to(device_test)

    # Create causal mask for self-attention (tgt_mask)
    causal_mask_test = torch.triu(
        torch.ones(T, T) * float("-inf"), diagonal=1
    ).to(device_test)

    # Create dummy padding mask for encoder memory
    memory_padding_mask_test = torch.zeros(B, S, dtype=torch.bool).to(
        device_test
    )
    memory_padding_mask_test[:, -5:] = True

    logger.info(f"Dummy target shape: {dummy_tgt.shape}")
    logger.info(f"Dummy memory shape: {dummy_memory.shape}")
    logger.info(f"Causal mask shape: {causal_mask_test.shape}")
    logger.info(f"Memory padding mask shape: {memory_padding_mask_test.shape}")

    # Test DecoderLayerPT
    logger.info("\n--- Testing DecoderLayerPT ---")
    try:
        decoder_layer = DecoderLayerPT(D, H, FFN_D).to(device_test)
        decoder_layer.eval()
        output = decoder_layer(
            dummy_tgt,
            dummy_memory,
            tgt_mask=causal_mask_test,
            memory_key_padding_mask=memory_padding_mask_test,
        )
        logger.info(f"DecoderLayerPT output shape: {output.shape}")
        assert output.shape == (B, T, D)
        logger.info("✅ DecoderLayerPT test passed.")
    except Exception as e:
        logger.error(f"❌ DecoderLayerPT test failed: {e}", exc_info=True)

    # Test AttentionPT directly (Self-Attention example)
    logger.info("\n--- Testing AttentionPT (Self-Attention) ---")
    try:
        attn_self = AttentionPT(D, H).to(device_test)
        attn_self.eval()
        output_self = attn_self(
            dummy_tgt, dummy_tgt, dummy_tgt, attn_mask=causal_mask_test
        )
        logger.info(f"AttentionPT (Self) output shape: {output_self.shape}")
        assert output_self.shape == (B, T, D)
        logger.info("✅ AttentionPT (Self) test passed.")
    except Exception as e:
        logger.error(f"❌ AttentionPT (Self) test failed: {e}", exc_info=True)

    # Test AttentionPT directly (Cross-Attention example)
    logger.info("\n--- Testing AttentionPT (Cross-Attention) ---")
    try:
        attn_cross = AttentionPT(D, H).to(device_test)
        attn_cross.eval()
        output_cross = attn_cross(
            dummy_tgt,
            dummy_memory,
            dummy_memory,
            key_padding_mask=memory_padding_mask_test,
        )
        logger.info(f"AttentionPT (Cross) output shape: {output_cross.shape}")
        assert output_cross.shape == (B, T, D)
        logger.info("✅ AttentionPT (Cross) test passed.")
    except Exception as e:
        logger.error(f"❌ AttentionPT (Cross) test failed: {e}", exc_info=True)

    logger.info("\n✅ PyTorch Decoder Modules test block finished.")
