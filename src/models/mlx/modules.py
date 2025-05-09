# Multimodal Transfer Learning / Image Captioning
# File: src/models/mlx/modules.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: MLX Transformer Decoder building blocks.
# Created: 2025-05-05
# Updated: 2025-05-05

import os
import sys  # Added for sys.path manipulation
from pathlib import Path

import mlx.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent.parent  # Go up three levels
if str(project_root) not in sys.path:
    print(f"🚂 [trainer_mlx.py] Adding project root: {project_root}")
    sys.path.insert(0, str(project_root))


class AttentionMLX(nn.Module):
    """
    MLX Multi-Head Attention wrapper for self-attention (masked)
    and cross-attention. Minimal implementation for testing.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiHeadAttention(embed_dim, num_heads)

    def __call__(
        self, query, key, value, attn_mask=None, key_padding_mask=None
    ):
        # MLX MultiHeadAttention expects [seq_len, batch, embed_dim]
        # attn_mask: [seq_len, seq_len] or None
        # key_padding_mask: [batch, seq_len] or None
        # For simplicity, ignore attn_mask and key_padding_mask in this minimal version
        return self.mha(query, key, value)


class FeedForwardMLX(nn.Module):
    """
    Standard two-layer feed-forward network for Transformers (MLX).
    """

    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def __call__(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class DecoderLayerMLX(nn.Module):
    """
    A single Transformer Decoder layer for MLX.
    Consists of masked self-attention, cross-attention, and a feed-forward network.
    """

    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        # Use correct MLX MultiHeadAttention parameters
        self.masked_self_attn = nn.MultiHeadAttention(
            dims=embed_dim, num_heads=num_heads
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiHeadAttention(
            dims=embed_dim, num_heads=num_heads
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # FFN layers
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

        # Store dimensions for shape handling
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def __call__(
        self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None
    ):
        # Self-attention block with residual connection and normalization
        residual = tgt
        tgt_norm = self.norm1(tgt)  # [B, T, D]

        # MLX MultiHeadAttention expects batch-first: [B, T, D]
        # No need to transpose - pass directly
        assert tgt_norm.ndim == 3, f"Expected [B,T,D]; got {tgt_norm.shape}"
        self_attn_out = self.masked_self_attn(tgt_norm, tgt_norm, tgt_norm)
        tgt = residual + self.dropout1(self_attn_out)

        # Cross-attention block with residual connection and normalization
        residual = tgt
        q_cross = self.norm2(tgt)  # [B, T_tgt, D]
        k_cross = memory  # [B, T_src, D]
        v_cross = k_cross

        # Verify shapes before cross-attention
        assert (
            q_cross.ndim == 3 and q_cross.shape[0] == k_cross.shape[0]
        ), f"Expected [B,T,D]; got {q_cross.shape} vs {k_cross.shape}"

        # Pass directly to cross-attention - batch-first
        cross_attn_out = self.cross_attn(q_cross, k_cross, v_cross)
        tgt = residual + self.dropout2(cross_attn_out)

        # Feedforward block with residual connection and normalization
        residual = tgt
        tgt_norm = self.norm3(tgt)
        tgt = residual + self.ffn(tgt_norm)

        return tgt


# Basic pass statement or main block for runnable scripts
if __name__ == "__main__":
    pass
