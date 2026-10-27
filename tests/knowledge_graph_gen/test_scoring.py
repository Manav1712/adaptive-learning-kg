"""Guards around LLM edge scoring. No API calls."""

from __future__ import annotations

import warnings

import pandas as pd

from src.knowledge_graph_gen.discover_content_links import (
    CONTENT_EDGE_COLUMNS,
    DiscoveryConfig,
    content_edge_from_result,
    select_chronological_content,
)
from src.knowledge_graph_gen.discover_prereqs import (
    PREREQ_EDGE_COLUMNS,
    PrereqConfig,
    prereq_edge_from_result,
    select_chronological_los,
)


def _lo(lo_id: str, order: int) -> dict:
    return {
        "lo_id": lo_id,
        "learning_objective": f"LO {lo_id}",
        "book_order": 1,
        "chapter_order": 1,
        "unit_order": 1,
        "lo_order": order,
    }


def _lookup() -> dict:
    return {"1": _lo("1", 1), "2": _lo("2", 2)}


def test_prereq_keeps_an_earlier_source_above_threshold():
    edge = prereq_edge_from_result(
        {"lo_id": 1, "score": 0.9, "confidence": 0.8, "rationale": "needed first"},
        target_id="2",
        allowed_source_ids={"1"},
        lo_lookup=_lookup(),
        config=PrereqConfig(),
    )
    assert edge is not None
    assert edge["source_lo_id"] == "1"
    assert edge["target_lo_id"] == "2"
    assert edge["relation"] == "prerequisite"
    assert edge["score"] == 0.9


def test_prereq_drops_low_score_and_low_confidence():
    config = PrereqConfig()
    common = dict(target_id="2", allowed_source_ids={"1"}, lo_lookup=_lookup(), config=config)
    assert prereq_edge_from_result({"lo_id": "1", "score": 0.69, "confidence": 0.9}, **common) is None
    assert prereq_edge_from_result({"lo_id": "1", "score": 0.9, "confidence": 0.59}, **common) is None


def test_prereq_keeps_high_score_when_confidence_is_omitted():
    edge = prereq_edge_from_result(
        {"lo_id": "1", "score": 0.9},
        target_id="2",
        allowed_source_ids={"1"},
        lo_lookup=_lookup(),
        config=PrereqConfig(),
    )
    assert edge is not None
    assert edge["confidence"] is None


def test_prereq_ignores_unparseable_results_and_ids_outside_the_batch():
    common = dict(
        target_id="2",
        allowed_source_ids={"1"},
        lo_lookup=_lookup(),
        config=PrereqConfig(),
    )
    assert prereq_edge_from_result({"lo_id": "1", "score": None}, **common) is None
    assert prereq_edge_from_result({"lo_id": "1", "score": "high"}, **common) is None
    assert prereq_edge_from_result("not-a-dict", **common) is None
    assert prereq_edge_from_result({"score": 0.9}, **common) is None
    # Id 2 was not in this batch. The old code could still write it, and the
    # chronological guard was skipped when the id was missing from the lookup.
    assert prereq_edge_from_result(
        {"lo_id": "999", "score": 1.0, "confidence": 1.0},
        **common,
    ) is None


def test_prereq_drops_a_source_that_is_not_earlier():
    edge = prereq_edge_from_result(
        {"lo_id": "2", "score": 1.0, "confidence": 1.0},
        target_id="1",
        allowed_source_ids={"2"},
        lo_lookup=_lookup(),
        config=PrereqConfig(),
    )
    assert edge is None


def test_empty_prereq_frame_keeps_columns():
    frame = pd.DataFrame([], columns=PREREQ_EDGE_COLUMNS)
    assert list(frame.columns) == PREREQ_EDGE_COLUMNS


def test_content_edge_uses_content_type_and_rejects_foreign_ids():
    row = pd.Series({"content_type": "example", "text": "a worked example"})
    config = DiscoveryConfig()
    edge = content_edge_from_result(
        {"lo_id": "5", "score": 0.85, "confidence": 0.4, "rationale": "shows it"},
        content_id="5_example_1",
        allowed_lo_ids={"5"},
        content_row=row,
        config=config,
    )
    assert edge is not None
    assert edge["relation"] == "exemplified_by"
    assert edge["target_content_id"] == "5_example_1"
    # Confidence is recorded and does not filter content edges.
    assert edge["confidence"] == 0.4

    assert content_edge_from_result(
        {"lo_id": "9", "score": 1.0},
        content_id="5_example_1",
        allowed_lo_ids={"5"},
        content_row=row,
        config=config,
    ) is None
    assert content_edge_from_result(
        {"lo_id": "5", "score": "n/a"},
        content_id="5_example_1",
        allowed_lo_ids={"5"},
        content_row=row,
        config=config,
    ) is None
    assert content_edge_from_result(
        {"lo_id": "5", "score": 0.79},
        content_id="5_example_1",
        allowed_lo_ids={"5"},
        content_row=row,
        config=config,
    ) is None


def test_empty_content_frame_keeps_columns():
    frame = pd.DataFrame([], columns=CONTENT_EDGE_COLUMNS)
    assert "target_content_id" in frame.columns
    assert len(frame) == 0


def test_limit_sort_does_not_warn_about_inplace():
    los = pd.DataFrame(
        {
            "lo_id": [3, 1, 2],
            "book": ["b", "b", "b"],
            "book_order": [1, 1, 1],
            "chapter_order": [2, 1, 1],
            "unit_order": [1, 1, 1],
            "lo_order": [3, 1, 2],
        }
    )
    content = pd.DataFrame(
        {
            "content_id": ["b", "a"],
            "content_type": ["try_it", "concept"],
            "book_order": [1, 1],
            "chapter_order": [1, 1],
            "unit_order": [1, 1],
            "lo_order": [2, 1],
            "book": ["b", "b"],
            "unit": ["u", "u"],
            "chapter": ["c", "c"],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        limited_los = select_chronological_los(los, 2)
        limited_content = select_chronological_content(content, 1)
    assert list(limited_los["lo_id"]) == [1, 2]
    assert list(limited_content["content_id"]) == ["a"]
    assert not any("inplace" in str(item.message) for item in caught)
