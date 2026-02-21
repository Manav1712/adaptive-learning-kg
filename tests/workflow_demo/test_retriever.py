"""Unit tests for workflow_demo.retriever module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.workflow_demo.retriever import EmbeddingBackend, TeachingPackRetriever


@pytest.mark.unit
def test_embedding_backend_requires_openai_key(monkeypatch):
    """EmbeddingBackend should raise error when OPENAI_API_KEY is not set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        EmbeddingBackend()


@pytest.mark.unit
def test_embedding_backend_rejects_invalid_model(monkeypatch):
    """Invalid model names should raise ValueError."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="Unsupported model"):
        EmbeddingBackend(model_name="sentence-transformers/does-not-exist")


@pytest.mark.unit
@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_embedding_backend_bm25_index_creation(monkeypatch):
    """BM25 index should be built during fit_transform."""
    # Mock OpenAI client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1] * 3072),
        MagicMock(embedding=[0.2] * 3072),
        MagicMock(embedding=[0.3] * 3072),
    ]
    mock_client.embeddings.create.return_value = mock_response
    
    # Mock OpenAI import
    mock_openai_class = MagicMock(return_value=mock_client)
    monkeypatch.setattr("openai.OpenAI", mock_openai_class)
    
    backend = EmbeddingBackend(model_name="text-embedding-3-large")
    texts = ["derivative calculus", "integral calculus", "linear algebra"]
    backend.fit_transform(texts)
    
    # BM25 index should exist (or be None if rank_bm25 not installed)
    if backend._bm25_index is not None:
        assert len(backend._bm25_corpus) == 3
        # Search should return results
        results = backend.bm25_search("derivative", top_k=2)
        assert len(results) > 0
        # First result should be "derivative calculus"
        assert results[0][0] == 0


@pytest.mark.unit
@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_embedding_backend_bm25_search_empty_query(monkeypatch):
    """BM25 search with empty query should return empty results."""
    # Mock OpenAI client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1] * 3072),
        MagicMock(embedding=[0.2] * 3072),
    ]
    mock_client.embeddings.create.return_value = mock_response
    
    # Mock OpenAI import
    mock_openai_class = MagicMock(return_value=mock_client)
    monkeypatch.setattr("openai.OpenAI", mock_openai_class)
    
    backend = EmbeddingBackend(model_name="text-embedding-3-large")
    texts = ["derivative calculus", "integral calculus"]
    backend.fit_transform(texts)
    
    if backend._bm25_index is not None:
        results = backend.bm25_search("", top_k=2)
        # Empty query returns no results with score > 0
        assert all(score == 0 for _, score in results) or len(results) == 0


@pytest.mark.unit
def test_embedding_backend_bm25_no_index():
    """BM25 search should return empty list when index is None."""
    backend = object.__new__(EmbeddingBackend)
    backend._bm25_index = None  # Force no index
    results = backend.bm25_search("derivative", top_k=2)
    assert results == []


@pytest.mark.unit
def test_build_lo_corpus_includes_learning_objective(sample_kg_data):
    """_build_lo_corpus should mirror KG metadata."""
    retriever = object.__new__(TeachingPackRetriever)
    retriever.kg = sample_kg_data
    lo_corpus = retriever._build_lo_corpus()
    assert lo_corpus[0]["learning_objective"] == "The Tangent Problem and Differential Calculus"
    assert lo_corpus[0]["unit"] == "A Preview of Calculus"


@pytest.mark.unit
def test_retrieve_plan_assembles_teaching_pack(monkeypatch, tmp_path, sample_kg_data):
    """Full retrieve_plan flow should produce a SessionPlan when embeddings are stubbed."""

    class StubEmbeddingBackend:
        def __init__(self, *args, **kwargs):
            self._mode = "tfidf"
            self._fitted = False
            self._dim = 0
            self._bm25_index = None

        def fit_transform(self, texts):
            self._fitted = True
            self._dim = len(texts)
            return np.eye(self._dim, dtype="float32")

        def transform(self, texts):
            if not self._fitted:
                raise AssertionError("fit_transform must be called before transform.")
            dim = self._dim if self._dim else 1
            return np.ones((len(texts), dim), dtype="float32")

        def bm25_search(self, query, top_k):
            return []

    monkeypatch.setattr(
        "src.workflow_demo.retriever.load_demo_frames",
        lambda _path: sample_kg_data,
    )
    monkeypatch.setattr(
        "src.workflow_demo.retriever.EmbeddingBackend",
        StubEmbeddingBackend,
    )

    retriever = TeachingPackRetriever(data_dir=tmp_path, embedding_model="stub")
    session_plan = retriever.retrieve_plan(
        query="Teach me derivatives",
        subject="calculus",
        learning_objective=None,
        mode="conceptual_review",
    )

    assert session_plan.learning_objective in {
        "The Tangent Problem and Differential Calculus",
        "Linear Functions and Slope",
    }
    assert session_plan.current_plan, "Current plan should contain at least one step."
    assert len(session_plan.current_plan) <= 3
    assert session_plan.current_plan[0].how_to_teach
    assert session_plan.current_plan[0].why_to_teach
    assert session_plan.future_plan, "Future plan should include at least one follow-up step."
    for step in session_plan.current_plan + session_plan.future_plan:
        assert step.step_type in {"prereq_review", "explain", "extension"}
        assert step.how_to_teach and step.why_to_teach
    assert session_plan.teaching_pack.key_points, "Teaching pack key points must not be empty."


@pytest.mark.unit
def test_hybrid_fusion_combines_results():
    """_hybrid_fusion should combine semantic and BM25 results using RRF."""
    retriever = object.__new__(TeachingPackRetriever)
    
    semantic_hits = [(0, 0.9), (1, 0.8), (2, 0.7)]
    bm25_hits = [(2, 5.0), (3, 4.0), (0, 3.0)]
    
    fused = retriever._hybrid_fusion(semantic_hits, bm25_hits, top_k=4)
    
    # Should have up to 4 unique results
    assert len(fused) <= 4
    # Index 0 and 2 appear in both, should have higher fused scores
    indices = [idx for idx, _ in fused]
    assert 0 in indices  # Appears in both
    assert 2 in indices  # Appears in both
    # Fused scores should be positive
    assert all(score > 0 for _, score in fused)


@pytest.mark.unit
def test_hybrid_fusion_empty_bm25():
    """_hybrid_fusion should work when BM25 returns empty results."""
    retriever = object.__new__(TeachingPackRetriever)
    
    semantic_hits = [(0, 0.9), (1, 0.8)]
    bm25_hits = []
    
    fused = retriever._hybrid_fusion(semantic_hits, bm25_hits, top_k=2)
    
    assert len(fused) == 2
    assert fused[0][0] == 0  # Highest semantic score first


@pytest.mark.unit
def test_hybrid_fusion_empty_semantic():
    """_hybrid_fusion should work when semantic returns empty results."""
    retriever = object.__new__(TeachingPackRetriever)
    
    semantic_hits = []
    bm25_hits = [(2, 5.0), (3, 4.0)]
    
    fused = retriever._hybrid_fusion(semantic_hits, bm25_hits, top_k=2)
    
    assert len(fused) == 2
    assert fused[0][0] == 2  # Highest BM25 score first


@pytest.mark.unit
def test_rerank_hits_fallback_no_client():
    """_rerank_hits should return original order when LLM client is None."""
    retriever = object.__new__(TeachingPackRetriever)
    retriever._llm_client = None
    
    hits = [(0, 0.9), (1, 0.8), (2, 0.7)]
    corpus = [{"text": "doc1"}, {"text": "doc2"}, {"text": "doc3"}]
    
    result = retriever._rerank_hits("query", hits, corpus, top_k=2)
    
    assert len(result) == 2
    assert result[0] == hits[0]
    assert result[1] == hits[1]


@pytest.mark.unit
def test_rerank_hits_fallback_on_error():
    """_rerank_hits should return original order when LLM call fails."""
    retriever = object.__new__(TeachingPackRetriever)
    retriever.rerank_model = "gpt-4o-mini"
    
    # Mock LLM client that raises an error
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API error")
    retriever._llm_client = mock_client
    
    hits = [(0, 0.9), (1, 0.8), (2, 0.7)]
    corpus = [{"text": "doc1"}, {"text": "doc2"}, {"text": "doc3"}]
    
    result = retriever._rerank_hits("query", hits, corpus, top_k=2)
    
    assert len(result) == 2
    assert result[0] == hits[0]


@pytest.mark.unit
def test_rerank_hits_success():
    """_rerank_hits should reorder based on LLM response."""
    retriever = object.__new__(TeachingPackRetriever)
    retriever.rerank_model = "gpt-4o-mini"
    
    # Mock LLM client that returns reordered ranks
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"ranks": [2, 1, 3]}'
    
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    retriever._llm_client = mock_client
    
    hits = [(0, 0.9), (1, 0.8), (2, 0.7)]
    corpus = [{"text": "doc1"}, {"text": "doc2"}, {"text": "doc3"}]
    
    result = retriever._rerank_hits("query", hits, corpus, top_k=3)
    
    # LLM said doc 2 is most relevant (index 1 in hits)
    assert result[0][0] == 1  # Original index 1 should be first


@pytest.mark.unit
def test_retrieve_plan_with_rerank_disabled(monkeypatch, tmp_path, sample_kg_data):
    """retrieve_plan should work with enable_rerank=False."""

    class StubEmbeddingBackend:
        def __init__(self, *args, **kwargs):
            self._mode = "tfidf"
            self._fitted = False
            self._dim = 0
            self._bm25_index = None

        def fit_transform(self, texts):
            self._fitted = True
            self._dim = len(texts)
            return np.eye(self._dim, dtype="float32")

        def transform(self, texts):
            dim = self._dim if self._dim else 1
            return np.ones((len(texts), dim), dtype="float32")

        def bm25_search(self, query, top_k):
            return []

    monkeypatch.setattr(
        "src.workflow_demo.retriever.load_demo_frames",
        lambda _path: sample_kg_data,
    )
    monkeypatch.setattr(
        "src.workflow_demo.retriever.EmbeddingBackend",
        StubEmbeddingBackend,
    )

    retriever = TeachingPackRetriever(data_dir=tmp_path, embedding_model="stub")
    retriever._llm_client = None  # Ensure no LLM client
    
    session_plan = retriever.retrieve_plan(
        query="Teach me derivatives",
        subject="calculus",
        learning_objective=None,
        mode="conceptual_review",
        enable_rerank=False,
    )

    assert session_plan.learning_objective
    assert session_plan.current_plan
