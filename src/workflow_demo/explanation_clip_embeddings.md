# clip_embeddings.py

## Purpose

Provides CLIP-based embedding utilities for encoding text and images into a shared 512-dimensional vector space. This allows direct similarity comparison between text queries and images using dot products. Used by the retriever for text-to-image and image-to-image search.

## Dependencies

- `sentence-transformers` (`SentenceTransformer`) -- loads and runs the CLIP model locally.
- `Pillow` (`PIL.Image`) -- opens and converts image files to RGB before encoding.
- `numpy` -- all embeddings are returned as float32 ndarrays.

## Standalone Function

### `normalize_dense(arr) -> np.ndarray`

L2-normalizes each row of a 2D array to unit length. Adds a small epsilon (1e-12) to norms to avoid division by zero. Returns the input unchanged if the array is empty.

**Called from:** `retriever.py` -- applied to image embeddings after `encode_images`, and to single image embeddings during image search.

## Class: CLIPEmbeddingBackend

### `__init__(model_name="clip-ViT-B-32")`

Loads the CLIP model via SentenceTransformer. The default model (`clip-ViT-B-32`) produces 512-dimensional embeddings for both text and images.

### `encode_text(texts) -> np.ndarray`

Takes a list of strings, encodes them through CLIP with L2 normalization, and returns a float32 ndarray of shape `(len(texts), 512)`.

**Called from:** `retriever.py` -- encodes a text query for text-to-image similarity search.

### `encode_images(image_paths) -> np.ndarray`

Takes a list of `Path` objects, opens each as RGB, encodes them through CLIP with L2 normalization, and returns a float32 ndarray of shape `(len(image_paths), 512)`. Raises `ValueError` if any image file cannot be read.

**Called from:** `retriever.py` -- batch-encodes all images during index build, and encodes a single image during image-based search.

## Flow

```
retriever builds image index
  |
  v
encode_images(image_paths)
  |-- open each image as RGB
  |-- run through CLIP model
  |-- return float32 ndarray (n, 512)
  |
  v
normalize_dense(embeddings)  [called by retriever]
  |-- L2-normalize each row to unit vector
  |-- store as self.image_embeddings

student submits a text query for image search
  |
  v
encode_text([query])
  |-- run text through CLIP model
  |-- return float32 vector (1, 512)
  |
  v
dot product: image_embeddings @ query_vec
  |-- rank images by similarity score
```

## Notes

- Both `encode_text` and `encode_images` pass `normalize_embeddings=True` to the model, so vectors are already unit-length. The retriever then calls `normalize_dense` again, which is a harmless no-op on already-normalized vectors.
- CLIP's key property: text and images live in the same vector space, so a text embedding can be directly compared against an image embedding for meaningful similarity.
