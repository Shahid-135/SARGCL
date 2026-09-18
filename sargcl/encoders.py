"""CLIP-based foundational feature extractors for SARGCL."""

from typing import List

import torch
import torch.nn as nn
import clip


class CLIPEncoders(nn.Module):
    """Frozen CLIP image and text encoders.

    Provides global image embeddings, global text embeddings, and
    per-node text embeddings (for encoding captions / extracted text nodes).

    Args:
        clip_model_name: CLIP architecture variant (default ``"ViT-B/32"``).
        device: Torch device string.
    """

    def __init__(self, clip_model_name: str = "ViT-B/32", device: str = "cpu"):
        super().__init__()
        self.clip, self.preprocess = clip.load(clip_model_name, device=device, jit=False)
        self.clip.float()  # Use float32 for stability
        self.text_hidden = self.clip.text_projection.shape[-1]
        self.vit_hidden = self.clip.visual.proj.shape[-1]

    @torch.no_grad()
    def encode_image_global(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images into L2-normalised CLIP embeddings."""
        img_feats = self.clip.encode_image(pixel_values.to(self.clip.dtype))
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        return img_feats.float()

    @torch.no_grad()
    def encode_text_global(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode tokenised text into L2-normalised CLIP embeddings."""
        txt_feats = self.clip.encode_text(input_ids)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
        return txt_feats.float()

    @torch.no_grad()
    def encode_text_for_nodes(self, texts: List[str], device) -> torch.Tensor:
        """Encode a list of short text strings (captions, node labels) via CLIP."""
        if not texts:
            return torch.zeros(0, self.text_hidden, device=device)
        tokens = clip.tokenize(texts, truncate=True).to(device)
        text_feats = self.clip.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return text_feats.float()
