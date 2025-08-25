"""
BASELINE V3 EXPERIMENT: Ontology-Enforced Ingestion with Type Balancing

What this does:
- Defines and applies a custom ontology (entity + edge types) before ingestion
- Sets a fact rating instruction on the graph to help filter noisy edges later
- Validates with a small sample batch, then ingests the full dataset
- Enables type balancing so Concepts dominate over Examples/Problems
- Enriches episode metadata (unit/chapter)

Customize at top: GRAPH_ID, CSV_PATHS, BATCH_SIZE, VALIDATION_LIMIT, MAX_PER_TYPE
"""

import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


# -------- Experiment configuration --------
GRAPH_ID: str = "baseline_v3_ontology_enforced"  # <-- set for this experiment
CSV_PATHS: List[str] = [
    "data/raw/try_it_draft_contents.csv",
    "data/raw/example_draft_contents (3).csv",
    "data/raw/concept_draft_contents (4).csv",
]
BATCH_SIZE: int = 20  # Zep API limit
VALIDATION_LIMIT: int = 40  # small sample prior to full ingestion
MAX_PER_TYPE: Optional[int] = 250  # enable type balancing; None to disable


# -------- Episode model & conversion --------
@dataclass
class CSVEpisode:
    lo_id: str
    content: str
    episode_type: str  # concept|example|exercise|try-it
    metadata: Dict[str, Any]
    raw_content: Dict[str, Any]

    def to_zep_episode(self) -> Dict[str, Any]:
        # Ontology hint header kept minimal; ontology will be enforced server-side
        header = (
            f"[LO: {self.lo_id}, Type: {self.episode_type}, "
            f"Unit: {self.metadata.get('unit', 'N/A')}, "
            f"Chapter: {self.metadata.get('chapter', 'N/A')}]\n\n"
        )

        parts: List[str] = []
        if "problem" in self.raw_content:
            parts.append(f"Problem: {self.raw_content['problem']}")
        sol = self.raw_content.get("solution")
        if isinstance(sol, dict):
            steps = sol.get("steps")
            if isinstance(steps, list) and steps:
                parts.append("\nSolution:")
                for i, step in enumerate(steps, 1):
                    if isinstance(step, dict) and "step" in step:
                        parts.append(f"{i}. {step['step']}")
        text = "\n\n".join(parts)
        return {"type": "text", "data": header + text}


class CSVToEpisodeConverter:
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path

    def convert_csv_to_episodes(self) -> List[CSVEpisode]:
        episodes: List[CSVEpisode] = []
        with open(self.csv_file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ep = self._convert_row(row)
                if ep:
                    episodes.append(ep)
        return episodes

    def _convert_row(self, row: Dict[str, str]) -> Optional[CSVEpisode]:
        try:
            raw = json.loads(row.get("raw_content", ""))
            if not isinstance(raw, dict):
                raw = {"problem": row.get("raw_content", "")}
        except json.JSONDecodeError:
            raw = {"problem": row.get("raw_content", "")}

        metadata = {
            "book": row.get("book", ""),
            "learning_objective": row.get("learning_objective", ""),
            "unit": row.get("unit", ""),
            "chapter": row.get("chapter", ""),
        }

        lo_id = row.get("lo_id", "").strip()
        ep_type = row.get("type", "").strip().lower()
        if not lo_id or not ep_type:
            return None

        return CSVEpisode(
            lo_id=lo_id,
            content=row.get("raw_content", ""),
            episode_type=ep_type,
            metadata=metadata,
            raw_content=raw,
        )


def load_all_episodes(paths: List[str]) -> List[CSVEpisode]:
    all_eps: List[CSVEpisode] = []
    for p in paths:
        try:
            conv = CSVToEpisodeConverter(p)
            eps = conv.convert_csv_to_episodes()
            print(f"Loaded {len(eps)} episodes from {p}")
            all_eps.extend(eps)
        except FileNotFoundError:
            print(f"CSV not found: {p} (skipping)")
    return all_eps


def balance_types(episodes: List[CSVEpisode], max_per_type: Optional[int]) -> List[CSVEpisode]:
    if not max_per_type:
        return episodes
    capped: List[CSVEpisode] = []
    per_type: Dict[str, int] = {}
    for ep in episodes:
        t = ep.episode_type
        if per_type.get(t, 0) < max_per_type:
            capped.append(ep)
            per_type[t] = per_type.get(t, 0) + 1
    print(f"Type balancing enabled (max {max_per_type}/type). Final: {len(capped)} episodes")
    return capped


# -------- Zep integration --------
class ZepKnowledgeGraphBuilder:
    def __init__(self, api_key: str):
        from zep_cloud import Zep
        self.client = Zep(api_key=api_key)

    def ensure_graph(self, graph_id: str, fact_rating_instruction: Optional[str] = None) -> None:
        print(f"Ensuring graph exists: {graph_id}...")
        try:
            self.client.graph.create(
                graph_id=graph_id,
                description="Baseline V3: Ontology-enforced knowledge graph with custom entity types, constrained relationships, and fact rating for noise reduction. Built on OpenStax calculus content with type balancing.",
                fact_rating_instruction=fact_rating_instruction
            )
            print("Graph created with description.")
        except Exception as e:
            print(f"Graph create returned: {e}. Proceeding.")
        if fact_rating_instruction:
            try:
                self.client.graph.update(graph_id=graph_id, fact_rating_instruction=fact_rating_instruction)
                print("Fact rating instruction set.")
            except Exception as e:
                print(f"Graph update (fact rating) returned: {e}.")

    def set_ontology(self, graph_id: str) -> None:
        """Define and apply custom entity/edge types with source/target constraints."""
        from zep_cloud import EntityEdgeSourceTarget

        # Entities: minimal schemas (description-only); fields optional for this experiment
        class Concept:
            description = "Represents a calculus concept or definition."

        class Example:
            description = "Represents a worked example."

        class Exercise:
            description = "Represents a practice problem or exercise."

        class TryIt:
            description = "Represents a short try-it/practice item."

        # Edges: map name -> (EdgeModel, [allowed source/target])
        class PREREQUISITE_OF:
            description = "Indicates prerequisite relationship between concepts."

        class PART_OF:
            description = "Indicates hierarchical/structural membership."

        class ASSESSED_BY:
            description = "Indicates an assessment item evaluates a concept."

        edges = {
            "PREREQUISITE_OF": (
                PREREQUISITE_OF,
                [EntityEdgeSourceTarget(source="Concept", target="Concept")],
            ),
            "PART_OF": (
                PART_OF,
                [
                    EntityEdgeSourceTarget(source="Concept", target="Concept"),
                    EntityEdgeSourceTarget(source="Example", target="Concept"),
                    EntityEdgeSourceTarget(source="Exercise", target="Concept"),
                    EntityEdgeSourceTarget(source="TryIt", target="Concept"),
                ],
            ),
            "ASSESSED_BY": (
                ASSESSED_BY,
                [
                    EntityEdgeSourceTarget(source="Exercise", target="Concept"),
                    EntityEdgeSourceTarget(source="TryIt", target="Concept"),
                ],
            ),
        }

        entities = {
            "Concept": Concept,
            "Example": Example,
            "Exercise": Exercise,
            "TryIt": TryIt,
        }

        print("Applying ontology to graph...")
        # Apply ontology for this specific graph
        self.client.graph.set_ontology(entities=entities, edges=edges, graph_ids=[graph_id])
        print("Ontology applied.")

    def add_batch(self, episodes: List[CSVEpisode], graph_id: str) -> None:
        payload = [{"type": ep.to_zep_episode()["type"], "data": ep.to_zep_episode()["data"]} for ep in episodes]
        self.client.graph.add_batch(graph_id=graph_id, episodes=payload)


def run_ingestion():
    load_dotenv()
    api_key = os.getenv("ZEP_API_KEY")
    if not api_key:
        print("Error: ZEP_API_KEY not set")
        return

    builder = ZepKnowledgeGraphBuilder(api_key)

    # 1) Ensure graph & set fact rating
    fact_rating_instruction = (
        "Rate graph facts by relevance to calculus learning. High: prerequisite/assessment/part-of relations; "
        "Medium: definitional/supporting relations; Low: incidental or stylistic statements."
    )
    builder.ensure_graph(GRAPH_ID, fact_rating_instruction=fact_rating_instruction)

    # 2) Apply ontology (entities + constrained edges)
    builder.set_ontology(GRAPH_ID)

    # 3) Load episodes
    episodes = load_all_episodes(CSV_PATHS)
    if not episodes:
        print("No episodes to ingest.")
        return

    # 4) Enable type balancing so Concepts dominate
    episodes = balance_types(episodes, MAX_PER_TYPE)

    # 5) Validation batch (small)
    print("\nValidation ingest (small sample)...")
    builder.add_batch(episodes[:VALIDATION_LIMIT], graph_id=GRAPH_ID)
    print(f"Validation batch added: {min(VALIDATION_LIMIT, len(episodes))} episodes")

    # 6) Full ingestion in batches
    print("\nFull ingest in batches...")
    total = len(episodes)
    added = 0
    for start in range(VALIDATION_LIMIT, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = episodes[start:end]
        try:
            builder.add_batch(batch, graph_id=GRAPH_ID)
            added += len(batch)
            print(f"Progress: {added + min(VALIDATION_LIMIT, total)}/{total}")
        except Exception as e:
            print(f"Batch {start}-{end} failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    run_ingestion()


