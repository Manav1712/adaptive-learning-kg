"""Unit tests for workflow_demo.retriever module."""

from pathlib import Path

import numpy as np
import pytest

from src.workflow_demo.retriever import EmbeddingBackend, TeachingPackRetriever


@pytest.mark.unit
def test_embedding_backend_fallback_to_tfidf():
    """Invalid model names should trigger the TF-IDF fallback."""
    backend = EmbeddingBackend(model_name="sentence-transformers/does-not-exist")
    vectors = backend.fit_transform(["The tangent problem", "Linear functions"])
    assert backend._mode == "tfidf"
    assert vectors.shape[0] == 2
    query_vector = backend.transform(["Derivative basics"])
    assert query_vector.shape[0] == 1


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

        def fit_transform(self, texts):
            self._fitted = True
            self._dim = len(texts)
            return np.eye(self._dim, dtype="float32")

        def transform(self, texts):
            if not self._fitted:
                raise AssertionError("fit_transform must be called before transform.")
            dim = self._dim if self._dim else 1
            return np.ones((len(texts), dim), dtype="float32")

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
    assert session_plan.teaching_pack.key_points, "Teaching pack key points must not be empty."
