"""
Embedding-based retriever that assembles lightweight teaching packs from the demo KG.

Implements a multi-stage retrieval pipeline:
- Stage A: Dense semantic + BM25 lexical search (hybrid fusion)
- Stage B: LLM reranking via GPT-4o-mini (optional)
- Stage C: Graph expansion for prereqs and related content
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from src.workflow_demo.data_loader import KnowledgeGraphData, load_demo_frames
    from src.workflow_demo.models import (
        PlanStep, SessionPlan, TeachingPack,
        RetrievalCandidate, RetrievalResult,
    )
    from src.workflow_demo.clip_embeddings import CLIPEmbeddingBackend, normalize_dense
except ImportError:
    from .data_loader import KnowledgeGraphData, load_demo_frames
    from .models import (
        PlanStep, SessionPlan, TeachingPack,
        RetrievalCandidate, RetrievalResult,
    )
    from .clip_embeddings import CLIPEmbeddingBackend, normalize_dense


# Cache version - increment when cache format changes
_CACHE_VERSION = "v1"


def _compute_corpus_hash(texts: List[str]) -> str:
    """
    Compute SHA256 hash of corpus texts for cache invalidation.
    
    Uses length-prefixed encoding to prevent collisions from texts containing
    newlines or other special characters.
    
    Inputs:
        texts: List of strings to hash.
    
    Outputs:
        Hexadecimal hash string (64 chars).
    """
    # Use length-prefixed encoding to prevent collisions
    # e.g., ["a\nb", "c"] vs ["a", "b\nc"] would collide with simple join
    hasher = hashlib.sha256()
    hasher.update(_CACHE_VERSION.encode("utf-8"))
    for text in sorted(texts):
        # Encode length + content to prevent collision attacks
        encoded = text.encode("utf-8")
        hasher.update(len(encoded).to_bytes(4, "big"))
        hasher.update(encoded)
    return hasher.hexdigest()


def _get_cache_path(data_dir: Path, cache_name: str) -> Path:
    """
    Get cache file path for embeddings.
    
    Inputs:
        data_dir: Base data directory (e.g., demo/).
        cache_name: Cache filename (e.g., "text_embeddings_{hash}.npy").
    
    Outputs:
        Path to cache file in data_dir/.embedding_cache/
    """
    cache_dir = data_dir / ".embedding_cache"
    return cache_dir / cache_name


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    """
    Atomically save numpy array to prevent corruption on interrupted writes.
    
    Writes to a temp file first, then renames (atomic on POSIX).
    
    Inputs:
        path: Target file path.
        arr: Numpy array to save.
    
    Outputs:
        None. File is saved atomically.
    
    Raises:
        OSError: If write or rename fails.
    """
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file in same directory (ensures same filesystem for atomic rename)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".npy.tmp",
        prefix=path.stem + "_",
        dir=path.parent
    )
    try:
        os.close(fd)
        np.save(tmp_path, arr)
        # Atomic rename (POSIX guarantees atomicity for same-filesystem rename)
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _safe_load_cache(
    cache_path: Path,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_dtype: np.dtype = np.float32,
) -> Optional[np.ndarray]:
    """
    Safely load cached embeddings with validation.
    
    Inputs:
        cache_path: Path to cache file.
        expected_shape: Expected shape (None to skip shape check).
        expected_dtype: Expected dtype.
    
    Outputs:
        Loaded numpy array, or None if cache is invalid/missing.
    """
    if not cache_path.exists():
        return None
    
    try:
        arr = np.load(cache_path)
        
        # Validate dtype
        if arr.dtype != expected_dtype:
            print(f"  [Cache] dtype mismatch: expected {expected_dtype}, got {arr.dtype}")
            return None
        
        # Validate shape if expected
        if expected_shape is not None and arr.shape != expected_shape:
            print(f"  [Cache] shape mismatch: expected {expected_shape}, got {arr.shape}")
            return None
        
        # Check for NaN/Inf values (corrupted embeddings)
        if not np.isfinite(arr).all():
            print("  [Cache] contains NaN or Inf values, regenerating...")
            return None
        
        return arr
        
    except Exception as e:
        print(f"  [Cache] failed to load {cache_path.name}: {e}")
        # Attempt to remove corrupted cache file
        try:
            cache_path.unlink()
            print(f"  [Cache] removed corrupted cache file")
        except OSError:
            pass
        return None


class EmbeddingBackend:
    """
    Embedding backend using OpenAI text-embedding-3-large (SOTA).
    
    Requires OPENAI_API_KEY environment variable. No fallbacks.
    Also builds a BM25 index for hybrid retrieval (semantic + keyword).
    """

    # OpenAI embedding model (SOTA quality, 3072 dimensions)
    OPENAI_MODEL = "text-embedding-3-large"

    def __init__(self, model_name: str = "text-embedding-3-large") -> None:
        """
        Initialize embedding backend with OpenAI API.

        Inputs:
            model_name: "text-embedding-3-large" (default) or "text-embedding-3-small".

        Outputs:
            None. Internally stores the OpenAI client.

        Raises:
            RuntimeError: If OPENAI_API_KEY is not set or OpenAI client initialization fails.
        """
        self._openai_client: Optional[Any] = None
        self._openai_model: str = model_name
        self._fitted = False
        
        # BM25 index for hybrid search
        self._bm25_index: Optional[Any] = None
        self._bm25_corpus: List[List[str]] = []

        # Require OpenAI API
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is required. "
                "Please set it to use OpenAI embeddings."
            )
        
        if model_name not in ("text-embedding-3-large", "text-embedding-3-small"):
            raise ValueError(
                f"Unsupported model: {model_name}. "
                "Must be 'text-embedding-3-large' or 'text-embedding-3-small'."
            )
        
        try:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=api_key)
            self._openai_model = model_name
            print(f"    Using OpenAI embeddings ({model_name})")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize OpenAI client: {e}. "
                "Please check your OPENAI_API_KEY and network connection."
            ) from e

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        """
        Encode corpus texts and return normalized embeddings.
        Also builds the BM25 index for keyword search.

        Inputs:
            texts: Iterable of strings to encode.

        Outputs:
            Normalized numpy array of shape (len(texts), dim).

        Raises:
            RuntimeError: If OpenAI API call fails.
        """
        texts_list = list(texts)
        
        if not self._openai_client:
            raise RuntimeError("OpenAI client not initialized. Check OPENAI_API_KEY.")
        
        embeddings = self._encode_openai(texts_list)

        # Build BM25 index for hybrid search
        self._build_bm25_index(texts_list)

        self._fitted = True
        return embeddings.astype("float32")

    def _encode_openai(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Encode texts using OpenAI API in batches.

        Inputs:
            texts: List of strings to encode.
            batch_size: Number of texts per API call.

        Outputs:
            Normalized numpy array of embeddings.

        Raises:
            RuntimeError: If OpenAI API call fails.
        """
        import time
        all_vecs: List[List[float]] = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if i % 200 == 0 and i > 0:
                print(f"      Encoded {i}/{len(texts)}...")
            
            try:
                response = self._openai_client.embeddings.create(
                    model=self._openai_model,
                    input=batch
                )
                batch_vecs = [item.embedding for item in response.data]
                all_vecs.extend(batch_vecs)
                time.sleep(0.05)  # Rate limiting
            except Exception as e:
                raise RuntimeError(
                    f"OpenAI API call failed at batch {i}: {e}. "
                    "Please check your API key, network connection, and rate limits."
                ) from e
        
        embeddings = np.array(all_vecs).astype("float32")
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        return embeddings / norms

    def _build_bm25_index(self, texts: Sequence[str]) -> None:
        """
        Build BM25 index from corpus texts.

        Inputs:
            texts: Iterable of strings to index.

        Outputs:
            None. Stores BM25 index internally.
        """
        try:
            from rank_bm25 import BM25Okapi
            self._bm25_corpus = [text.lower().split() for text in texts]
            self._bm25_index = BM25Okapi(self._bm25_corpus)
            print("    BM25 index built successfully")
        except ImportError:
            print("    rank_bm25 not available, hybrid search will use semantic only")
            self._bm25_index = None

    def bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        Search using BM25 keyword matching.

        Inputs:
            query: Search query string.
            top_k: Number of top results to return.

        Outputs:
            List of (index, score) tuples sorted by relevance.
        """
        if self._bm25_index is None:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_idx if scores[idx] > 0]

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """
        Encode new queries using OpenAI API.

        Inputs:
            texts: Iterable of query strings.

        Outputs:
            Normalized numpy array of embeddings.

        Raises:
            RuntimeError: If OpenAI client not initialized or API call fails.
        """
        texts_list = list(texts)

        if not self._openai_client:
            raise RuntimeError("OpenAI client not initialized. Check OPENAI_API_KEY.")
        
        return self._encode_openai(texts_list)



class TeachingPackRetriever:
    """
    Retrieves the minimal high-signal context (key points, examples, practice) for a LO query.
    
    Implements a multi-stage pipeline:
    - Stage A: Hybrid search (semantic + BM25 with RRF fusion)
    - Stage B: LLM reranking (optional, uses gpt-5.4-mini)
    - Stage C: Graph expansion (prereqs + related content)
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        embedding_model: str = "text-embedding-3-large",
        rerank_model: str = "gpt-5.4-mini",
        force_refresh: bool = False,
    ) -> None:
        """
        Build embeddings over LO titles and content snippets.

        Inputs:
            data_dir: Directory containing the demo CSV artifacts. Defaults to <repo>/demo.
            embedding_model: OpenAI model name (default: text-embedding-3-large) or SentenceTransformer.
            rerank_model: LLM model for reranking (default: gpt-5.4-mini).
            force_refresh: If True, bypass cache and recompute embeddings.

        Outputs:
            None. Pre-computes embeddings for subsequent fast retrieval.
        """
        self._force_refresh = force_refresh

        repo_root = Path(__file__).resolve().parents[2]
        self.data_dir = data_dir or (repo_root / "demo")
        print(f"  Loading knowledge graph from {self.data_dir}...")
        self.kg: KnowledgeGraphData = load_demo_frames(self.data_dir)
        print(f"  Loaded {len(self.kg.los)} LOs and {len(self.kg.content)} content items")

        print(f"  Initializing embedding backend ({embedding_model})...")
        self.embedding_backend = EmbeddingBackend(model_name=embedding_model)
        self.embedding_model = embedding_model
        print(f"  Embedding backend ready")

        print("  Building corpus...")
        self.lo_corpus = self._build_lo_corpus()
        self.content_corpus = self._build_content_corpus()
        print(f"  Built {len(self.lo_corpus)} LO entries and {len(self.content_corpus)} content entries")

        lo_texts = [item["text"] for item in self.lo_corpus]
        content_texts = [item["text"] for item in self.content_corpus]
        all_texts = lo_texts + content_texts

        # Try loading cached embeddings
        corpus_hash = _compute_corpus_hash(all_texts)
        # Include model name in hash to invalidate cache when model changes
        model_hash = hashlib.sha256(embedding_model.encode("utf-8")).hexdigest()[:8]
        cache_name = f"text_embeddings_{corpus_hash[:16]}_{model_hash}.npy"
        cache_path = _get_cache_path(self.data_dir, cache_name)

        embeddings = None
        if not self._force_refresh:
            embeddings = _safe_load_cache(cache_path, expected_dtype=np.float32)
            if embeddings is not None:
                print(f"  Loaded cached embeddings from {cache_path.name} ({embeddings.shape})")
                # Still need to build BM25 index (fast, no API calls)
                self.embedding_backend._build_bm25_index(all_texts)
                self.embedding_backend._fitted = True
        
        if embeddings is None:
            if self._force_refresh:
                print(f"  Force refresh: recomputing embeddings...")
            print(f"  Computing embeddings for {len(all_texts)} texts (this may take a moment)...")
            embeddings = self.embedding_backend.fit_transform(all_texts)
            try:
                _atomic_save(cache_path, embeddings)
                print(f"  Cached embeddings to {cache_path.name}")
            except OSError as e:
                print(f"  [Cache] failed to save cache: {e} (continuing without cache)")

        self.lo_embeddings = embeddings[: len(lo_texts)]
        self.content_embeddings = embeddings[len(lo_texts) :]
        self._lo_text_count = len(lo_texts)  # Track split point for BM25
        print("  Embeddings ready!")

        # CLIP image index (optional)
        self.clip_backend: Optional[CLIPEmbeddingBackend] = None
        self.image_corpus: List[Dict[str, Any]] = []
        self.image_embeddings: np.ndarray = np.zeros((0, 512), dtype="float32")
        self._init_image_index()

        # LLM reranker setup
        self.rerank_model = rerank_model
        self._llm_client: Optional[Any] = None
        self._init_llm_client()

    def _init_image_index(self) -> None:
        """
        Initialize CLIP backend and load image metadata/embeddings if available.
        """
        # Look for image corpus in src/workflow_demo/image_corpus/
        workflow_demo_dir = Path(__file__).resolve().parent
        self.image_corpus_dir = workflow_demo_dir / "image_corpus"
        metadata_path = self.image_corpus_dir / "image_metadata.csv"
        if not metadata_path.exists():
            print("  [Retriever] No image_metadata.csv found; image search disabled.")
            return

        try:
            print("  Loading CLIP model for image search (may take a moment on first run)...")
            self.clip_backend = CLIPEmbeddingBackend()
            print("  CLIP model ready")
        except Exception as exc:
            print(f"  [Retriever] CLIP unavailable ({exc}); skipping image index.")
            return

        self.image_corpus = self._build_image_corpus(metadata_path)
        if not self.image_corpus:
            print("  [Retriever] No valid images loaded; image search disabled.")
            return

        image_paths = [item["path"] for item in self.image_corpus]
        
        # Try loading cached image embeddings
        image_hash = _compute_corpus_hash([str(p) for p in image_paths])
        image_cache_name = f"image_embeddings_{image_hash[:16]}.npy"
        image_cache_path = _get_cache_path(self.data_dir, image_cache_name)
        
        loaded_embeddings = None
        if not self._force_refresh:
            loaded_embeddings = _safe_load_cache(
                image_cache_path,
                expected_shape=(len(image_paths), 512),
                expected_dtype=np.float32,
            )
            if loaded_embeddings is not None:
                print(f"  [Retriever] Loaded cached image embeddings from {image_cache_path.name}")
                self.image_embeddings = loaded_embeddings
        
        if loaded_embeddings is None:
            print(f"  [Retriever] Computing embeddings for {len(image_paths)} images...")
            self.image_embeddings = self.clip_backend.encode_images(image_paths)
            self.image_embeddings = normalize_dense(self.image_embeddings)
            try:
                _atomic_save(image_cache_path, self.image_embeddings)
                print(f"  [Retriever] Cached image embeddings to {image_cache_path.name}")
            except OSError as e:
                print(f"  [Retriever] failed to save image cache: {e} (continuing without cache)")
        
        print(f"  [Retriever] Image index ready with {len(self.image_corpus)} images.")

    def _init_llm_client(self) -> None:
        """
        Initialize OpenAI client for LLM reranking.

        Inputs:
            None.

        Outputs:
            None. Sets self._llm_client if API key is available.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("  [Retriever] No OPENAI_API_KEY; LLM reranking disabled")
            return
        try:
            from openai import OpenAI
            self._llm_client = OpenAI(api_key=api_key)
            print(f"  [Retriever] LLM reranking enabled ({self.rerank_model})")
        except Exception as e:
            print(f"  [Retriever] Failed to init OpenAI client: {e}")
            self._llm_client = None

    def retrieve_plan(
        self,
        query: str,
        subject: str,
        learning_objective: Optional[str],
        mode: str,
        student_profile: Optional[Dict[str, Dict[str, float]]] = None,
        top_los: int = 6,
        top_content: int = 6,
        enable_rerank: bool = True,
    ) -> SessionPlan:
        """
        Execute multi-stage retrieval and assemble a simplified SessionPlan ready for the Coach.

        Pipeline:
        - Stage A: Hybrid search (semantic + BM25 with RRF fusion)
        - Stage B: LLM reranking (optional)
        - Stage C: Graph expansion (prereqs + related content)

        Inputs:
            query: Student's question or topic request.
            subject: Subject area (e.g., "calculus").
            learning_objective: Optional explicit LO name to anchor on.
            mode: Tutoring mode (conceptual_review, examples, practice).
            student_profile: Dict with lo_mastery scores for adaptive prereq inclusion.
            top_los: Number of LOs to retrieve.
            top_content: Number of content items to retrieve.
            enable_rerank: Whether to use LLM reranking (default: True).

        Outputs:
            SessionPlan with current/future steps and teaching pack.
        """

        canonical_mode = (mode or "conceptual_review").strip() or "conceptual_review"

        # Stage A: Hybrid search (semantic + BM25)
        query_vector = self.embedding_backend.transform([query])[0]
        
        # Semantic search
        semantic_lo_hits = self._search_embeddings(self.lo_embeddings, query_vector, top_los * 4)
        semantic_content_hits = self._search_embeddings(
            self.content_embeddings, query_vector, top_content * 4
        )
        
        # BM25 search (searches combined corpus)
        bm25_all_hits = self.embedding_backend.bm25_search(query, (top_los + top_content) * 4)
        
        # Filter LO hits (indices < lo_text_count)
        bm25_lo_hits = [
            (idx, score)
            for idx, score in bm25_all_hits
            if idx < self._lo_text_count
        ][:top_los * 4]
        
        # Adjust BM25 content indices (offset by LO count)
        bm25_content_hits = [
            (idx - self._lo_text_count, score)
            for idx, score in bm25_all_hits
            if idx >= self._lo_text_count
        ][:top_content * 4]
        
        # Hybrid fusion using RRF
        lo_hits = self._hybrid_fusion(semantic_lo_hits, bm25_lo_hits, top_k=top_los * 2)
        content_hits = self._hybrid_fusion(
            semantic_content_hits, bm25_content_hits, top_k=top_content * 2
        )
        
        # Stage B: LLM reranking (optional)
        if enable_rerank and self._llm_client:
            lo_hits = self._rerank_hits(query, lo_hits, self.lo_corpus, top_k=top_los)
            content_hits = self._rerank_hits(
                query, content_hits, self.content_corpus, top_k=top_content
            )
        else:
            lo_hits = lo_hits[:top_los]
            content_hits = content_hits[:top_content]

        # Stage C: Graph expansion (in _build_teaching_pack and _build_current_plan)
        primary_lo = self._select_primary_lo(lo_hits, learning_objective)
        teaching_pack = self._build_teaching_pack(primary_lo, lo_hits, content_hits, query=query)

        lo_mastery = student_profile.get("lo_mastery", {}) if student_profile else {}
        current_plan = self._build_current_plan(
            primary_lo=primary_lo,
            mode=canonical_mode,
            lo_mastery=lo_mastery,
        )
        session_guidance = self._generate_session_guidance(
            primary_lo=primary_lo,
            prereq_steps=[s for s in current_plan if s.step_type == "prereq_review"],
            lo_mastery=lo_mastery,
        )
        future_plan = self._build_future_plan(
            primary_lo=primary_lo,
            lo_hits=lo_hits,
            mode=canonical_mode,
        )
        first_question = (
            f"Before we dive in, what do you already know about {primary_lo['learning_objective']}?"
        )

        return SessionPlan(
            subject=subject,
            learning_objective=primary_lo["learning_objective"],
            mode=canonical_mode,
            current_plan=current_plan,
            future_plan=future_plan,
            first_question=first_question,
            teaching_pack=teaching_pack,
            session_guidance=session_guidance,
            book=primary_lo.get("book"),
            unit=primary_lo.get("unit"),
            chapter=primary_lo.get("chapter"),
        )

    def retrieve_candidates(
        self,
        text_query: str,
        image_path: Optional[str] = None,
        top_k: int = 6,
        debug: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve LO candidates from text and image embeddings.
        
        This is the new simplified retrieval interface that:
        1. Runs text embedding search (from OCR or user query)
        2. Runs image embedding search in parallel (if image provided)
        3. Merges and deduplicates candidates
        4. Prints debug output showing top retrieved LOs
        
        Inputs:
            text_query: Text query (user input or OCR result)
            image_path: Optional path to image for CLIP embedding search
            top_k: Number of top candidates to return per source
            debug: If True, print top retrieved LOs
        
        Outputs:
            RetrievalResult with text_candidates, image_candidates, and merged_candidates
        """
        text_candidates: List[RetrievalCandidate] = []
        image_candidates: List[RetrievalCandidate] = []
        
        # --- TEXT EMBEDDING RETRIEVAL ---
        if text_query.strip():
            query_vector = self.embedding_backend.transform([text_query])[0]
            
            # Semantic search on LO embeddings
            semantic_hits = self._search_embeddings(self.lo_embeddings, query_vector, top_k * 2)
            
            # BM25 keyword search
            bm25_all_hits = self.embedding_backend.bm25_search(text_query, top_k * 2)
            bm25_lo_hits = [
                (idx, score) for idx, score in bm25_all_hits
                if idx < self._lo_text_count
            ][:top_k * 2]
            
            # Hybrid fusion
            fused_hits = self._hybrid_fusion(semantic_hits, bm25_lo_hits, top_k=top_k)
            
            for idx, score in fused_hits:
                if 0 <= idx < len(self.lo_corpus):
                    lo = self.lo_corpus[idx]
                    text_candidates.append(RetrievalCandidate(
                        lo_id=lo["lo_id"],
                        title=lo["learning_objective"],
                        score=score,
                        source="text_embedding",
                        book=lo.get("book"),
                        unit=lo.get("unit"),
                        chapter=lo.get("chapter"),
                        how_to_teach=self._get_how_to_teach(lo["lo_id"]),
                        why_to_teach=self._get_why_to_teach(lo["lo_id"]),
                    ))
        
        # --- IMAGE EMBEDDING RETRIEVAL (CLIP) ---
        if image_path and self.clip_backend and self.image_embeddings.size > 0:
            try:
                # Encode image with CLIP
                image_embedding = self.clip_backend.encode_images([image_path])[0]
                image_embedding = normalize_dense(image_embedding.reshape(1, -1))[0]
                
                # Search image index
                scores = self.image_embeddings @ image_embedding.reshape(-1, 1)
                scores = scores.reshape(-1)
                top_idx = np.argsort(scores)[::-1][:top_k]
                
                for idx in top_idx:
                    if 0 <= idx < len(self.image_corpus):
                        img = self.image_corpus[idx]
                        lo_id = img.get("lo_id")
                        if lo_id is not None:
                            # Look up LO details from lo_corpus
                            lo_entry = next(
                                (lo for lo in self.lo_corpus if lo["lo_id"] == lo_id),
                                None
                            )
                            if lo_entry:
                                image_candidates.append(RetrievalCandidate(
                                    lo_id=lo_id,
                                    title=lo_entry["learning_objective"],
                                    score=float(scores[idx]),
                                    source="image_embedding",
                                    book=lo_entry.get("book"),
                                    unit=lo_entry.get("unit"),
                                    chapter=lo_entry.get("chapter"),
                                    how_to_teach=self._get_how_to_teach(lo_id),
                                    why_to_teach=self._get_why_to_teach(lo_id),
                                ))
            except Exception as e:
                print(f"  [Retriever] Image embedding search failed: {e}")
        
        # --- MERGE CANDIDATES ---
        merged_candidates = self._merge_candidates(text_candidates, image_candidates)
        
        # --- DEBUG OUTPUT ---
        if debug:
            print("\n" + "="*60)
            print("🔍 RETRIEVAL DEBUG OUTPUT")
            print("="*60)
            print(f"Query: {text_query[:100]}{'...' if len(text_query) > 100 else ''}")
            if image_path:
                print(f"Image: {image_path}")
            print()
            
            print(f"Text Embedding Candidates (top {len(text_candidates)})")
            for i, c in enumerate(text_candidates[:5], 1):
                print(f"  {i}. [{c.score:.3f}] {c.title} (LO #{c.lo_id})")
            
            if image_candidates:
                print(f"\nImage Embedding Candidates (top {len(image_candidates)})")
                for i, c in enumerate(image_candidates[:5], 1):
                    print(f"  {i}. [{c.score:.3f}] {c.title} (LO #{c.lo_id})")
            
            print(f"\nMerged Candidates (top {len(merged_candidates)})")
            for i, c in enumerate(merged_candidates[:5], 1):
                print(f"  {i}. [{c.score:.3f}] {c.title} (LO #{c.lo_id}, source: {c.source})")
            print("="*60 + "\n")
        
        return RetrievalResult(
            query=text_query,
            text_candidates=text_candidates,
            image_candidates=image_candidates,
            merged_candidates=merged_candidates,
        )
    
    def _merge_candidates(
        self,
        text_candidates: List[RetrievalCandidate],
        image_candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        """
        Merge and deduplicate candidates from text and image retrieval.
        
        Uses max score when same LO appears in both sources.
        """
        seen: Dict[int, RetrievalCandidate] = {}
        
        # Add text candidates first
        for c in text_candidates:
            if c.lo_id not in seen or c.score > seen[c.lo_id].score:
                seen[c.lo_id] = c
        
        # Add image candidates, keeping higher score
        for c in image_candidates:
            if c.lo_id not in seen or c.score > seen[c.lo_id].score:
                seen[c.lo_id] = c
        
        # Sort by score descending
        merged = sorted(seen.values(), key=lambda x: -x.score)
        return merged
    
    def _get_how_to_teach(self, lo_id: int) -> str:
        """Get how_to_teach guidance from KG for an LO."""
        # For now, generate a default. In full implementation, this comes from KG.
        lo_entry = next((lo for lo in self.lo_corpus if lo["lo_id"] == lo_id), None)
        if lo_entry:
            return f"Build understanding of {lo_entry['learning_objective']} through {lo_entry.get('unit', 'this unit')}."
        return "Use examples and step-by-step explanations."
    
    def _get_why_to_teach(self, lo_id: int) -> str:
        """Get why_to_teach rationale from KG for an LO."""
        # For now, generate a default. In full implementation, this comes from KG.
        prereqs = self.kg.prereq_in_map.get(lo_id, [])
        if prereqs:
            prereq_titles = [
                lo["learning_objective"] for lo in self.lo_corpus
                if lo["lo_id"] in prereqs
            ][:2]
            if prereq_titles:
                return f"This builds on: {', '.join(prereq_titles)}"
        return "This is a foundational concept for later topics."

    def _build_lo_corpus(self) -> List[Dict[str, str]]:
        """
        Craft LO corpus entries combining titles with structural metadata.
        """

        corpus: List[Dict[str, str]] = []
        for row in self.kg.los.itertuples(index=False):
            text = f"{row.learning_objective}. Unit: {row.unit}. Chapter: {row.chapter}. Book: {row.book}."
            corpus.append(
                {
                    "doc_id": f"lo_{int(row.lo_id)}",
                    "lo_id": int(row.lo_id),
                    "text": text,
                    "raw_title": row.learning_objective,
                    "learning_objective": row.learning_objective,  # Alias for consistency
                    "unit": row.unit,
                    "chapter": row.chapter,
                    "book": row.book,
                }
            )
        return corpus

    def _build_content_corpus(self) -> List[Dict[str, str]]:
        """
        Craft content corpus entries using truncated passages for retrieval.
        """

        corpus: List[Dict[str, str]] = []
        for row in self.kg.content.itertuples(index=False):
            snippet = (row.text or "")[:1200]
            corpus.append(
                {
                    "doc_id": str(row.content_id),
                    "lo_id_parent": int(row.lo_id_parent) if not pd.isna(row.lo_id_parent) else None,
                    "content_type": row.content_type,
                    "lo_title": row.learning_objective,
                    "text": snippet,
                    "raw_text": row.text,
                }
            )
        return corpus

    def _build_image_corpus(self, metadata_path: Path) -> List[Dict[str, Any]]:
        """
        Load image metadata and filter to existing files.
        """
        corpus: List[Dict[str, Any]] = []
        try:
            df = pd.read_csv(metadata_path)
        except Exception as exc:
            print(f"  [Retriever] Failed to read image metadata ({exc}); skipping.")
            return corpus

        images_dir = self.image_corpus_dir
        for row in df.itertuples(index=False):
            filename = getattr(row, "filename", None)
            image_id = getattr(row, "image_id", None)
            if not filename or not image_id:
                continue
            # Check derivatives subfolder first, then root
            nested_path = images_dir / "derivatives" / filename
            path = nested_path if nested_path.exists() else images_dir / filename
            if not path.exists():
                # Skip missing files to avoid failures
                continue
            lo_id = getattr(row, "lo_id", None)
            topic = getattr(row, "topic", "")
            description = getattr(row, "description", "")
            keywords = getattr(row, "keywords", "")
            keyword_list = []
            if isinstance(keywords, str):
                keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]

            corpus.append(
                {
                    "image_id": str(image_id),
                    "path": path,
                    "lo_id": int(lo_id) if pd.notna(lo_id) else None,
                    "topic": topic,
                    "description": description,
                    "keywords": keyword_list,
                }
            )
        return corpus

    def _search_embeddings(
        self,
        matrix: np.ndarray,
        query_vector: np.ndarray,
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """
        Compute cosine similarity between query and corpus embeddings.
        """

        scores = matrix @ query_vector.reshape(-1, 1)
        scores = scores.reshape(-1)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_idx]

    def _search_images(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search image index using CLIP text encoder.
        """
        if not self.clip_backend or self.image_embeddings.size == 0:
            return []

        try:
            query_vec = self.clip_backend.encode_text([query])[0]
            scores = self.image_embeddings @ query_vec.reshape(-1, 1)
            scores = scores.reshape(-1)
            top_idx = np.argsort(scores)[::-1][:top_k]
            results: List[Dict[str, Any]] = []
            for idx in top_idx:
                results.append(
                    {
                        **self.image_corpus[idx],
                        "score": float(scores[idx]),
                    }
                )
            return results
        except Exception as exc:
            print(f"  [Retriever] Image search failed ({exc}); returning none.")
            return []

    def _hybrid_fusion(
        self,
        semantic_hits: List[Tuple[int, float]],
        bm25_hits: List[Tuple[int, float]],
        top_k: int,
        rrf_k: int = 60,
    ) -> List[Tuple[int, float]]:
        """
        Combine semantic and BM25 results using Reciprocal Rank Fusion (RRF).

        Inputs:
            semantic_hits: List of (index, score) from semantic search.
            bm25_hits: List of (index, score) from BM25 search.
            top_k: Number of results to return.
            rrf_k: RRF constant (default 60, standard value).

        Outputs:
            List of (index, fused_score) tuples sorted by relevance.
        """
        scores: Dict[int, float] = {}
        
        # Add semantic scores using RRF formula
        for rank, (idx, _) in enumerate(semantic_hits):
            scores[idx] = scores.get(idx, 0) + 1 / (rrf_k + rank + 1)
        
        # Add BM25 scores using RRF formula
        for rank, (idx, _) in enumerate(bm25_hits):
            scores[idx] = scores.get(idx, 0) + 1 / (rrf_k + rank + 1)
        
        # Sort by fused score
        sorted_items = sorted(scores.items(), key=lambda x: -x[1])
        return [(idx, score) for idx, score in sorted_items[:top_k]]

    def _rerank_hits(
        self,
        query: str,
        hits: List[Tuple[int, float]],
        corpus: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Rerank top candidates using LLM for higher precision.

        Inputs:
            query: Original search query.
            hits: List of (index, score) from hybrid search.
            corpus: Corpus list to get document text.
            top_k: Number of results to return after reranking.

        Outputs:
            List of (index, score) tuples reordered by LLM relevance.
        """
        if not self._llm_client or not hits:
            return hits[:top_k]

        # Take top candidates for reranking (limit to save tokens)
        candidates = hits[:min(top_k * 2, len(hits), 15)]
        
        # Format documents for LLM
        doc_block = []
        for i, (idx, _) in enumerate(candidates):
            # Bounds check to prevent IndexError
            if idx < 0 or idx >= len(corpus):
                continue
            text = corpus[idx].get("text", "")[:400]
            doc_block.append(f"{i + 1}. {text}")
        
        prompt = f"""Rerank these documents for the query: "{query}"

Documents:
{chr(10).join(doc_block)}

Return ONLY a JSON object with the document numbers in order of relevance:
{{"ranks": [1, 3, 2, ...]}}

Most relevant first. Include all document numbers."""

        try:
            response = self._llm_client.chat.completions.create(
                model=self.rerank_model,
                messages=[
                    {"role": "system", "content": "You reorder search results by relevance. Output strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=150,
            )
            
            content = response.choices[0].message.content or "{}"
            # Parse JSON, handling potential markdown wrapping
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            payload = json.loads(content.strip())
            ranks = payload.get("ranks", [])
            
            # Reorder based on LLM ranks
            reordered = []
            used = set()
            for rank_num in ranks:
                idx_in_candidates = int(rank_num) - 1
                if 0 <= idx_in_candidates < len(candidates) and idx_in_candidates not in used:
                    orig_idx, orig_score = candidates[idx_in_candidates]
                    # Assign new score based on rerank position
                    new_score = 1.0 - (len(reordered) * 0.05)
                    reordered.append((orig_idx, new_score))
                    used.add(idx_in_candidates)
            
            # Add any missing candidates at the end
            for i, (idx, score) in enumerate(candidates):
                if i not in used:
                    reordered.append((idx, score * 0.5))
            
            return reordered[:top_k]
            
        except Exception as e:
            print(f"  [Retriever] Rerank failed ({e}); using pre-rerank order")
            return hits[:top_k]

    def _select_primary_lo(
        self,
        lo_hits: List[Tuple[int, float]],
        learning_objective: Optional[str],
    ) -> Dict[str, str]:
        """
        Choose the LO anchor either by explicit name or top similarity.
        """

        if learning_objective:
            lowered = learning_objective.lower()
            for entry in self.lo_corpus:
                if entry["raw_title"].lower() == lowered:
                    return entry

        best_idx = lo_hits[0][0] if lo_hits else 0
        return self.lo_corpus[best_idx]

    def _build_teaching_pack(
        self,
        primary_lo: Dict[str, str],
        lo_hits: List[Tuple[int, float]],
        content_hits: List[Tuple[int, float]],
        query: str,
    ) -> TeachingPack:
        """
        Convert retrieval hits into a concise TeachingPack object.
        """

        key_points = [
            f"{primary_lo['raw_title']} is the focus objective drawn from {primary_lo['chapter']} ({primary_lo['book']})."
        ]
        for idx, _score in lo_hits[1:3]:
            neighbor = self.lo_corpus[idx]
            key_points.append(
                f"Related LO: {neighbor['raw_title']} in {neighbor['unit']} – useful for branching."
            )

        examples, practice = [], []
        citations = []
        for idx, score in content_hits:
            content_item = self.content_corpus[idx]
            snippet = (content_item["raw_text"] or content_item["text"])[:320]
            record = {
                "content_id": content_item["doc_id"],
                "lo_title": content_item["lo_title"],
                "content_type": content_item["content_type"],
                "snippet": snippet,
                "score": round(score, 4),
            }

            citations.append(
                {
                    "content_id": content_item["doc_id"],
                    "lo_title": content_item["lo_title"],
                    "score": round(score, 4),
                }
            )

            if content_item["content_type"] in {"concept", "example"} and len(examples) < 2:
                examples.append(record)
            elif content_item["content_type"] in {"try_it", "exercise", "practice"} and len(practice) < 2:
                practice.append(record)

        prereq_entries = []
        prereq_ids = self.kg.prereq_in_map.get(primary_lo["lo_id"], [])[:3]
        for prereq_id in prereq_ids:
            lo_row = self.kg.lo_lookup.get(prereq_id)
            if lo_row:
                prereq_entries.append(
                    {
                        "lo_id": prereq_id,
                        "title": lo_row["learning_objective"],
                        "note": "Recommended refresher before tackling the main concept.",
                    }
                )

        images = []
        image_results = self._search_images(query, top_k=3)
        for item in image_results:
            images.append(
                {
                    "image_id": item.get("image_id"),
                    "path": str(item.get("path")),
                    "lo_id": item.get("lo_id"),
                    "topic": item.get("topic"),
                    "description": item.get("description"),
                    "keywords": item.get("keywords", []),
                    "score": round(float(item.get("score", 0.0)), 4),
                }
            )

        return TeachingPack(
            key_points=key_points,
            examples=examples,
            practice=practice,
            prerequisites=prereq_entries,
            citations=citations,
            images=images,
        )

    def _build_current_plan(
        self,
        primary_lo: Dict[str, str],
        mode: str,
        lo_mastery: Dict[str, float],
    ) -> List[PlanStep]:
        """
        Compose the current tutoring plan with an optional prereq warm-up and a primary LO step.
        """

        steps: List[PlanStep] = []
        prereq_ids = self.kg.prereq_in_map.get(primary_lo["lo_id"], [])
        PROFICIENCY_THRESHOLD = 0.65

        for prereq_id in prereq_ids[:2]:
            prereq = self.kg.lo_lookup.get(prereq_id)
            if not prereq:
                continue

            prereq_proficiency = lo_mastery.get(
                prereq["learning_objective"],
                lo_mastery.get(prereq_id, 0.4),
            )

            if prereq_proficiency < PROFICIENCY_THRESHOLD:
                how, why = self._prereq_guidance(
                    prereq["learning_objective"], primary_lo["raw_title"], mode
                )
                steps.append(
                    PlanStep(
                        step_id=str(len(steps) + 1),
                        step_type="prereq_review",
                        goal=f"Refresh {prereq['learning_objective']} before tackling {primary_lo['raw_title']}.",
                        lo_id=prereq_id,
                        content_id=None,
                        budget_tokens=200,
                        how_to_teach=how,
                        why_to_teach=why,
                    )
                )

        how, why = self._primary_guidance(primary_lo["raw_title"], mode)
        steps.append(
            PlanStep(
                step_id=str(len(steps) + 1),
                step_type="explain",
                goal=f"Describe the core idea behind {primary_lo['raw_title']}.",
                lo_id=primary_lo["lo_id"],
                content_id=None,
                budget_tokens=250,
                how_to_teach=how,
                why_to_teach=why,
            )
        )

        return steps

    def _generate_session_guidance(
        self,
        primary_lo: Dict[str, str],
        prereq_steps: List[PlanStep],
        lo_mastery: Dict[str, float],
    ) -> str:
        """
        Generate explicit instructional guidance for the tutor based on proficiency.
        """
        main_lo_name = primary_lo["raw_title"]

        if prereq_steps:
            prereq_names: List[str] = []
            for step in prereq_steps:
                prereq = self.kg.lo_lookup.get(step.lo_id)
                if prereq:
                    prereq_names.append(prereq["learning_objective"])

            prereq_list = " and ".join(prereq_names) if prereq_names else "prerequisites"
            return (
                "The learner has gaps in prerequisite knowledge. "
                f"Begin the session with focused refreshers on {prereq_list} "
                f"before transitioning to {main_lo_name}. "
                "Check for understanding after each refresher."
            )

        return (
            "The learner has strong prerequisite knowledge. "
            f"Focus primarily on {main_lo_name} without extensive prerequisite review. "
            "Move quickly to deeper applications and nuances."
        )

    def _build_future_plan(
        self,
        primary_lo: Dict[str, str],
        lo_hits: List[Tuple[int, float]],
        mode: str,
    ) -> List[PlanStep]:
        """
        Suggest a compact future plan with an extension LO and optional supporting prereqs.
        """

        future_steps: List[PlanStep] = []
        followup = self._select_followup_lo(primary_lo, lo_hits)
        reuse_primary = False
        if not followup:
            followup = primary_lo
            reuse_primary = True

        how, why = self._extension_guidance(primary_lo["raw_title"], followup["raw_title"], mode)
        future_steps.append(
            PlanStep(
                step_id="F1",
                step_type="extension",
                goal=(
                    f"Extend to related LO {followup['raw_title']}."
                    if not reuse_primary
                    else f"Reinforce {followup['raw_title']} with a short forward-looking preview."
                ),
                lo_id=followup["lo_id"],
                content_id=None,
                budget_tokens=200,
                how_to_teach=how,
                why_to_teach=why,
            )
        )

        prereq_ids = self.kg.prereq_in_map.get(followup["lo_id"], [])[:2]
        counter = 2
        for prereq_id in prereq_ids:
            prereq = self.kg.lo_lookup.get(prereq_id)
            if not prereq:
                continue
            how, why = self._future_prereq_guidance(
                prereq["learning_objective"], followup["raw_title"], mode
            )
            future_steps.append(
                PlanStep(
                    step_id=f"F{counter}",
                    step_type="prereq_review",
                    goal=f"Warm up with {prereq['learning_objective']} before {followup['raw_title']}.",
                    lo_id=prereq_id,
                    content_id=None,
                    budget_tokens=180,
                    how_to_teach=how,
                    why_to_teach=why,
                )
            )
            counter += 1
            if counter > 3:
                break

        return future_steps

    def _select_followup_lo(
        self, primary_lo: Dict[str, str], lo_hits: List[Tuple[int, float]]
    ) -> Optional[Dict[str, str]]:
        for idx, _score in lo_hits:
            candidate = self.lo_corpus[idx]
            if candidate["lo_id"] != primary_lo["lo_id"]:
                return candidate
        return None

    @staticmethod
    def _primary_guidance(lo_title: str, mode: str) -> Tuple[str, str]:
        if mode == "practice":
            how = f"Work a short problem on {lo_title} alongside the student, pausing to check their reasoning."
            why = "Active problem solving exposes misconceptions immediately."
        elif mode == "examples":
            how = f"Demonstrate {lo_title} with a step-by-step example, narrating the purpose of each move."
            why = "Concrete walkthroughs show how the abstract idea is applied."
        else:  # conceptual_review default
            how = f"Start with an intuitive story or visual for {lo_title}, then connect it to the formal definition."
            why = "A mental model makes the later symbolic rules easier to absorb."
        return how, why

    @staticmethod
    def _prereq_guidance(prereq_title: str, primary_title: str, mode: str) -> Tuple[str, str]:
        if mode == "practice":
            how = f"Work through a quick practice problem on {prereq_title} to refresh the key skills."
            why = f"Active practice with {prereq_title} ensures readiness before tackling {primary_title}."
        elif mode == "examples":
            how = f"Walk through a concise example of {prereq_title} to refresh understanding."
            why = f"Seeing {prereq_title} in action prepares the student for {primary_title} examples."
        else:  # conceptual_review default
            how = f"Spend two minutes revisiting {prereq_title} with a quick conceptual overview and a comprehension check."
            why = f"Confidence with {prereq_title} keeps {primary_title} from feeling overwhelming."
        return how, why

    @staticmethod
    def _future_prereq_guidance(prereq_title: str, followup_title: str, mode: str) -> Tuple[str, str]:
        if mode == "practice":
            how = f"Assign a practice problem on {prereq_title} as preparation for {followup_title}."
            why = f"Working through {prereq_title} problems will make {followup_title} practice smoother."
        elif mode == "examples":
            how = f"Review an example of {prereq_title} before diving into {followup_title} examples."
            why = f"Understanding {prereq_title} examples sets up the context for {followup_title}."
        else:  # conceptual_review default
            how = f"Assign a brief conceptual refresher on {prereq_title} as preparation homework."
            why = f"It unlocks the follow-up topic {followup_title} so the next session can move faster."
        return how, why

    @staticmethod
    def _extension_guidance(current_title: str, followup_title: str, mode: str) -> Tuple[str, str]:
        if mode == "practice":
            how = f"Preview a practice problem that connects {current_title} to {followup_title}."
            why = "Showing the bridge between topics motivates continued practice."
        elif mode == "examples":
            how = f"Preview an example that shows how {followup_title} builds on {current_title}."
            why = "Concrete examples demonstrate the progression between concepts."
        else:  # conceptual_review default
            how = f"Highlight how {followup_title} builds on {current_title} and preview one real example."
            why = "Setting the expectation keeps the learner curious about the next milestone."
        return how, why

