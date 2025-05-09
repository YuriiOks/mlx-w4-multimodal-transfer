import mlx.core as mx
import mlx.nn as nn
import numpy as np


class ViTMLX(nn.Module):
    """
    Minimal Vision Transformer (ViT) for MLX (random weights).
    This is a stub for MLX-native vision encoding.
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        embed_dim=768,
        num_layers=4,
        num_heads=8,
        mlp_dim=3072,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.pos_embed = mx.zeros((1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)
        self.encoder_layers = [
            nn.TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, 0.1)
            for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(embed_dim)

    def _img_to_patches(self, images):
        # images: [B, 3, H, W] numpy or mx.array
        B, C, H, W = images.shape
        p = self.patch_size
        images = images.reshape(B, C, H // p, p, W // p, p)
        images = images.transpose(0, 2, 4, 3, 5, 1)
        patches = images.reshape(B, -1, p * p * C)
        return patches

    def __call__(self, pixel_values):
        # pixel_values: [B, 3, H, W] (numpy or mx.array)
        if isinstance(pixel_values, np.ndarray):
            pixel_values = mx.array(pixel_values)
        patches = self._img_to_patches(pixel_values)
        x = self.patch_embed(patches)
        B = x.shape[0]
        cls_tokens = mx.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        x = mx.concatenate([cls_tokens, x], axis=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        # MLX expects [seq_len, batch, embed_dim]
        x = mx.transpose(x, (1, 0, 2))
        for layer in self.encoder_layers:
            x = layer(x, mask=None)
        x = mx.transpose(x, (1, 0, 2))
        x = self.norm(x)
        # Return last_hidden_state (all tokens)
        return type("ViTMLXOutput", (), {"last_hidden_state": x})()
