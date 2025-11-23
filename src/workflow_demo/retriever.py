"""
Embedding-based retriever that assembles lightweight teaching packs from the demo KG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from src.workflow_demo.data_loader import KnowledgeGraphData, load_demo_frames
    from src.workflow_demo.models import PlanStep, SessionPlan, TeachingPack
except ImportError:
    from .data_loader import KnowledgeGraphData, load_demo_frames
    from .models import PlanStep, SessionPlan, TeachingPack


class EmbeddingBackend:
    """
    Minimal wrapper around SentenceTransformer with TF-IDF fallback when the
    transformer model cannot be loaded (e.g., offline environments).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """
        Initialize encoder and fallback vectorizer.

        Inputs:
            model_name: Hugging Face identifier for the primary embedding model.

        Outputs:
            None. Internally stores the encoder or TF-IDF vectorizer.
        """

        try:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(model_name)
            self._vectorizer = None
            self._mode = "st"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._encoder = None
            self._vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
            self._mode = "tfidf"

        self._fitted = False

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        """
        Fit the encoder/vectorizer on corpus texts and return normalized embeddings.

        Inputs:
            texts: Iterable of strings to encode.

        Outputs:
            Normalized numpy array of shape (len(texts), dim).
        """

        if self._mode == "st":
            embeddings = self._encoder.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            matrix = self._vectorizer.fit_transform(texts)
            embeddings = _normalize_dense(matrix.toarray())

        self._fitted = True
        return embeddings.astype("float32")

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """
        Encode new queries against the already-fitted encoder/vectorizer.

        Inputs:
            texts: Iterable of query strings.

        Outputs:
            Normalized numpy array of embeddings.
        """

        if not self._fitted and self._mode == "tfidf":
            raise RuntimeError("TF-IDF backend must be fitted before encoding queries.")

        if self._mode == "st":
            embeddings = self._encoder.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            matrix = self._vectorizer.transform(texts)
            embeddings = _normalize_dense(matrix.toarray())

        return embeddings.astype("float32")


def _normalize_dense(arr: np.ndarray) -> np.ndarray:
    """
    Normalize rows of a dense matrix to unit length.
    """

    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms


class TeachingPackRetriever:
    """
    Retrieves the minimal high-signal context (key points, examples, practice) for a LO query.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """
        Build embeddings over LO titles and content snippets.

        Inputs:
            data_dir: Directory containing the demo CSV artifacts. Defaults to <repo>/demo.
            embedding_model: Name of the encoder used by EmbeddingBackend.

        Outputs:
            None. Pre-computes embeddings for subsequent fast retrieval.
        """

        repo_root = Path(__file__).resolve().parents[2]
        self.data_dir = data_dir or (repo_root / "demo")
        self.kg: KnowledgeGraphData = load_demo_frames(self.data_dir)

        self.embedding_backend = EmbeddingBackend(model_name=embedding_model)

        self.lo_corpus = self._build_lo_corpus()
        self.content_corpus = self._build_content_corpus()

        lo_texts = [item["text"] for item in self.lo_corpus]
        content_texts = [item["text"] for item in self.content_corpus]
        all_texts = lo_texts + content_texts

        embeddings = self.embedding_backend.fit_transform(all_texts)
        self.lo_embeddings = embeddings[: len(lo_texts)]
        self.content_embeddings = embeddings[len(lo_texts) :]

    def retrieve_plan(
        self,
        query: str,
        subject: str,
        learning_objective: Optional[str],
        mode: str,
        student_profile: Optional[Dict[str, Dict[str, float]]] = None,
        top_los: int = 5,
        top_content: int = 6,
    ) -> SessionPlan:
        """
        Execute embedding search and assemble a SessionPlan ready for the Coach.

        Inputs:
            query: Student request or paraphrased LO description.
            subject: High-level subject (calculus, algebra, etc.).
            learning_objective: Optional explicit LO name to prioritize.
            mode: Tutoring mode (conceptual_review, examples, practice).
            student_profile: Dict with proficiency scores used to bias the plan.
            top_los: Number of LO candidates to consider.
            top_content: Max content items pulled for teaching pack.

        Outputs:
            SessionPlan containing current steps, future steps, and a teaching pack.
        """

        query_vector = self.embedding_backend.transform([query])[0]
        lo_hits = self._search_embeddings(self.lo_embeddings, query_vector, top_los)
        content_hits = self._search_embeddings(self.content_embeddings, query_vector, top_content)

        primary_lo = self._select_primary_lo(lo_hits, learning_objective)
        teaching_pack = self._build_teaching_pack(primary_lo, lo_hits, content_hits)

        proficiency = 0.4
        if student_profile:
            proficiency = student_profile.get("lo_mastery", {}).get(primary_lo["lo_id"], 0.4)

        current_plan = self._build_current_plan(primary_lo, teaching_pack, mode, proficiency)
        future_plan = self._build_future_plan(primary_lo, lo_hits)
        first_question = (
            f"Before we dive in, what do you already know about {primary_lo['learning_objective']}?"
        )

        return SessionPlan(
            subject=subject,
            learning_objective=primary_lo["learning_objective"],
            mode=mode,
            current_plan=current_plan,
            future_plan=future_plan,
            first_question=first_question,
            teaching_pack=teaching_pack,
        )

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

        return TeachingPack(
            key_points=key_points,
            examples=examples,
            practice=practice,
            prerequisites=prereq_entries,
            citations=citations,
        )

    def _build_current_plan(
        self,
        primary_lo: Dict[str, str],
        teaching_pack: TeachingPack,
        mode: str,
        proficiency: float,
    ) -> List[PlanStep]:
        """
        Compose the current tutoring plan, optionally inserting prerequisite review.
        """

        steps: List[PlanStep] = []
        step_counter = 1

        needs_prereq = proficiency < 0.5 and teaching_pack.prerequisites
        if needs_prereq:
            prereq_lo = teaching_pack.prerequisites[0]
            steps.append(
                PlanStep(
                    step_id=str(step_counter),
                    step_type="prereq_review",
                    goal=f"Refresh {prereq_lo['title']} to close knowledge gaps.",
                    lo_id=prereq_lo["lo_id"],
                    budget_tokens=200,
                )
            )
            step_counter += 1

        steps.append(
            PlanStep(
                step_id=str(step_counter),
                step_type="explain",
                goal=f"Describe the core idea behind {primary_lo['raw_title']}.",
                lo_id=primary_lo["lo_id"],
                budget_tokens=250,
            )
        )
        step_counter += 1

        if teaching_pack.examples:
            steps.append(
                PlanStep(
                    step_id=str(step_counter),
                    step_type="example",
                    goal="Walk through an anchored example pulled from the KG.",
                    lo_id=primary_lo["lo_id"],
                    content_id=teaching_pack.examples[0]["content_id"],
                    budget_tokens=230,
                )
            )
            step_counter += 1

        if teaching_pack.practice and mode != "conceptual_review":
            steps.append(
                PlanStep(
                    step_id=str(step_counter),
                    step_type="practice",
                    goal="Guide the student through a short practice check.",
                    lo_id=primary_lo["lo_id"],
                    content_id=teaching_pack.practice[0]["content_id"],
                    budget_tokens=220,
                )
            )

        return steps

    def _build_future_plan(
        self,
        primary_lo: Dict[str, str],
        lo_hits: List[Tuple[int, float]],
    ) -> List[PlanStep]:
        """
        Suggest future plan steps consisting of additional LOs and prerequisites.
        """

        future_steps: List[PlanStep] = []
        step_counter = 1

        prereq_ids = self.kg.prereq_in_map.get(primary_lo["lo_id"], [])[:2]
        for prereq_id in prereq_ids:
            lo_row = self.kg.lo_lookup.get(prereq_id)
            if lo_row:
                future_steps.append(
                    PlanStep(
                        step_id=f"F{step_counter}",
                        step_type="prereq_review",
                        goal=f"Deepen foundation on {lo_row['learning_objective']}.",
                        lo_id=prereq_id,
                        budget_tokens=180,
                    )
                )
                step_counter += 1

        for idx, _score in lo_hits[1:3]:
            neighbor = self.lo_corpus[idx]
            future_steps.append(
                PlanStep(
                    step_id=f"F{step_counter}",
                    step_type="extension",
                    goal=f"Extend to related LO {neighbor['raw_title']}.",
                    lo_id=neighbor["lo_id"],
                    budget_tokens=200,
                )
            )
            step_counter += 1

        return future_steps

