"""
CLIP embedding backend for shared text-image vector space.
Uses sentence-transformers' clip-ViT-B-32 for a lightweight, local embedding
option that supports both text and images. Requires:
- sentence-transformers
- Pillow
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


class CLIPEmbeddingBackend:
    """
    CLIP-based embeddings for text and images in a shared vector space.
    """

    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "sentence-transformers is required for CLIP embeddings. "
                "Install with `pip install sentence-transformers Pillow`."
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into CLIP text embeddings.
        """
        if not texts:
            return np.zeros((0, 512), dtype="float32")
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return embeddings.astype("float32")

    def encode_images(self, image_paths: List[Path]) -> np.ndarray:
        """
        Encode a list of image file paths into CLIP image embeddings.
        """
        if not image_paths:
            return np.zeros((0, 512), dtype="float32")

        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "Pillow is required for image encoding. "
                "Install with `pip install Pillow`."
            ) from exc

        images = []
        for path in image_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception as exc:
                raise ValueError(
                    f"Failed to load image at {path}: {exc}"
                ) from exc

        if not images:
            return np.zeros((0, 512), dtype="float32")

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
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


