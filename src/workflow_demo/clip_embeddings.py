"""
CLIP embedding backend for shared text-image vector space.

CLIP is a model from OpenAI trained on text-image pairs. The key property
is that it puts text and images into the same vector space -- so you can
directly compare a text embedding against an image embedding and get a
meaningful similarity score.

Uses sentence-transformers' clip-ViT-B-32 for a lightweight, local
embedding option that supports both text and images. Requires:
- sentence-transformers
- Pillow
"""

from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

class CLIPEmbeddingBackend:
    """
    CLIP-based embeddings for text and images in a shared vector space.
    """

    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into CLIP text embeddings.
        """
        # Run texts through CLIP; returns L2-normalized numpy vectors.
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        # Ensure consistent float32 dtype for downstream math.
        return embeddings.astype("float32")

    def encode_images(self, image_paths: List[Path]) -> np.ndarray:
        """
        Encode a list of image file paths into CLIP image embeddings.
        """
        # Load each image as RGB; raise on any unreadable file.
        images = []
        for path in image_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception as exc:
                raise ValueError(
                    f"Failed to load image at {path}: {exc}"
                ) from exc

        # Encode images through CLIP into the shared vector space.
        embeddings = self.model.encode(
            images, convert_to_numpy=True, normalize_embeddings=True
        )
        return embeddings.astype("float32")

def normalize_dense(arr: np.ndarray) -> np.ndarray:
    """
    L2-normalize embeddings to unit vectors.
    """
    if arr.size == 0:
        return arr
    # Compute per-row L2 norms; epsilon avoids division by zero.
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


