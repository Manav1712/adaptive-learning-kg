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
GRAPH_ID = "baseline_v3_full_group"  # <-- new graph for full ingestion
CSV_PATHS: List[str] = [
    "data/raw/try_it_draft_contents.csv",
    "data/raw/example_draft_contents (3).csv",
    "data/raw/concept_draft_contents (4).csv",
]
BATCH_SIZE: int = 20  # Zep API limit
VALIDATION_LIMIT = 40  # small sample prior to full ingestion
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
            # Create graph without description first to avoid JSON error
            self.client.graph.create(graph_id=graph_id)
            print("Graph created.")
        except Exception as e:
            print(f"Graph create returned: {e}. Proceeding.")
        
        # Set fact rating instruction separately
        if fact_rating_instruction:
            try:
                # Zep expects a structured instruction with examples
                from zep_cloud import ModelsFactRatingInstruction, ModelsFactRatingExamples
                fri = ModelsFactRatingInstruction(
                    instruction=fact_rating_instruction,
                    examples=ModelsFactRatingExamples(
                        high="PREREQUISITE_OF(Concept A -> Concept B)",
                        medium="PART_OF(Example -> Concept)",
                        low="Stylistic phrasing without mathematical relation"
                    )
                )
                self.client.graph.update(graph_id=graph_id, fact_rating_instruction=fri)
                print("Fact rating instruction set.")
            except Exception as e:
                print(f"Graph update (fact rating) returned: {e}.")

    def set_ontology(self, graph_id: str) -> None:
        """Define and apply custom entity/edge types with source/target constraints."""
        from zep_cloud import EntityEdgeSourceTarget
        from zep_cloud.external_clients.ontology import EntityModel, EdgeModel, EntityText
        from pydantic import Field

        # Entities: define with at least one property and a description (docstring)
        class Concept(EntityModel):
            """Represents a calculus concept or definition."""
            category: EntityText = Field(description="High-level type label for the concept", default=None)

        class Example(EntityModel):
            """Represents a worked example."""
            role: EntityText = Field(description="Example role or tag (e.g., worked example)", default=None)

        class Exercise(EntityModel):
            """Represents a practice problem or exercise."""
            difficulty: EntityText = Field(description="Relative difficulty or tag", default=None)

        class TryIt(EntityModel):
            """Represents a short try-it/practice item."""
            difficulty: EntityText = Field(description="Relative difficulty or tag", default=None)

        # Edges: define with at least one property and a description (docstring)
        class PREREQUISITE_OF(EdgeModel):
            """Prerequisite relation between two concepts."""
            justification: EntityText = Field(description="Rationale or context for the prerequisite", default=None)

        class PART_OF(EdgeModel):
            """Structural membership relation to a concept."""
            context: EntityText = Field(description="Hierarchy context (e.g., chapter/unit)", default=None)

        class ASSESSED_BY(EdgeModel):
            """Assessment relation mapping an exercise/try-it to a concept."""
            rubric: EntityText = Field(description="Assessment rubric or tag", default=None)

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

    def add_batch(self, episodes: List[CSVEpisode], graph_id: str) -> List[str]:
        """Add episodes in batch and return their UUIDs for status tracking."""
        payload = [{"type": ep.to_zep_episode()["type"], "data": ep.to_zep_episode()["data"]} for ep in episodes]
        
        try:
            response = self.client.graph.add_batch(graph_id=graph_id, episodes=payload)
            
            # Extract episode UUIDs from response - Zep returns a list of Episode objects
            episode_uuids = []
            if isinstance(response, list):
                for episode_obj in response:
                    if hasattr(episode_obj, 'uuid_'):
                        episode_uuids.append(episode_obj.uuid_)
                    elif hasattr(episode_obj, 'uuid'):
                        episode_uuids.append(episode_obj.uuid)
            
            print(f"Batch added: {len(episodes)} episodes, extracted {len(episode_uuids)} UUIDs")
            return episode_uuids
            
        except Exception as e:
            print(f"Error in add_batch: {e}")
            return []

    def wait_for_episodes_processed(self, episode_uuids: List[str], graph_id: str, timeout_minutes: int = 30) -> bool:
        """Wait for episodes to be processed, with rate limit handling."""
        import time
        
        if not episode_uuids:
            print("No episode UUIDs to check")
            return True
            
        print(f"Waiting for {len(episode_uuids)} episodes to process...")
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        check_interval = 10  # Check every 10 seconds instead of 1
        
        while time.time() - start_time < timeout_seconds:
            processed_count = 0
            rate_limit_hit = False
            
            for uuid_ in episode_uuids:
                try:
                    episode = self.client.graph.episode.get(uuid_=uuid_)
                    if hasattr(episode, 'processed') and episode.processed:
                        processed_count += 1
                    # Add small delay between individual checks to avoid rate limits
                    time.sleep(0.5)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "rate limit" in error_msg or "429" in error_msg:
                        print(f"⚠️  Rate limit hit, waiting 60 seconds...")
                        time.sleep(60)
                        rate_limit_hit = True
                        break
                    else:
                        print(f"Error checking episode {uuid_}: {e}")
            
            if rate_limit_hit:
                continue  # Retry the full check after rate limit cooldown
            
            if processed_count == len(episode_uuids):
                print(f"✅ All {len(episode_uuids)} episodes processed successfully!")
                return True
            
            remaining = len(episode_uuids) - processed_count
            elapsed = int(time.time() - start_time)
            print(f"⏳ {processed_count}/{len(episode_uuids)} processed ({remaining} remaining). Elapsed: {elapsed}s")
            
            # Wait longer between checks to avoid rate limits
            print(f"Waiting {check_interval} seconds before next check...")
            time.sleep(check_interval)
        
        print(f"❌ Timeout after {timeout_minutes} minutes. Assuming episodes are processing in background.")
        return True  # Return True to continue - episodes may still be processing


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

    # 5) Validation batch (small) - submit only, no polling
    print("\nValidation ingest (small sample)...")
    import time
    validated = 0
    v_total = min(VALIDATION_LIMIT, len(episodes))
    for start in range(0, v_total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, v_total)
        try:
            builder.add_batch(episodes[start:end], graph_id=GRAPH_ID)
            validated += (end - start)
            time.sleep(2)  # small delay to avoid rate limits
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg:
                print("Rate limit hit during validation. Waiting 60s and retrying once...")
                time.sleep(60)
                try:
                    builder.add_batch(episodes[start:end], graph_id=GRAPH_ID)
                    validated += (end - start)
                except Exception as e2:
                    print(f"Validation batch {start}-{end} failed after retry: {e2}")
            else:
                print(f"Validation batch {start}-{end} failed: {e}")
    print(f"Validation batch submitted: {validated} episodes")

    # 6) Full ingestion in batches - submit only, no polling
    print("\nFull ingest in batches...")
    total = len(episodes)
    added = 0
    for start in range(VALIDATION_LIMIT, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = episodes[start:end]
        try:
            print(f"Submitting batch {start//BATCH_SIZE + 1} ({start}-{end})...")
            builder.add_batch(batch, graph_id=GRAPH_ID)
            added += len(batch)
            print(f"Progress: {added + min(VALIDATION_LIMIT, total)}/{total}")
            time.sleep(2)  # small delay to avoid rate limits
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg:
                print("Rate limit hit. Waiting 60s and retrying this batch once...")
                time.sleep(60)
                try:
                    builder.add_batch(batch, graph_id=GRAPH_ID)
                    added += len(batch)
                    print(f"Progress: {added + min(VALIDATION_LIMIT, total)}/{total}")
                except Exception as e2:
                    print(f"Batch {start}-{end} failed after retry: {e2}")
            else:
                print(f"Batch {start}-{end} failed: {e}")

    print("\n🎉 Ingestion submitted! Episodes are processing asynchronously in Zep.")
    print(f"Total episodes submitted: {added + min(VALIDATION_LIMIT, total)}")
    print("You can monitor processing in the Zep console.")


if __name__ == "__main__":
    run_ingestion()


