"""
BASELINE V2 EXPERIMENT: Improved Knowledge Graph Ingestion

What this does:
- Ingests CSV episodes into Zep with schema hints for tighter entity extraction
- Uses relationship whitelist (PREREQUISITE_OF, PART_OF, ASSESSED_BY) to reduce noise
- Optional content type balancing to prevent skewed distributions
- Cleaner episode formatting with structured metadata

Key differences from baseline_v1:
- Added schema hints in episode headers to guide Zep's extraction
- Constrained relationships to core types instead of letting Zep infer freely
- Added optional type balancing to prevent examples from dominating concepts
- More structured episode content with clearer problem/solution parsing
- Experiment-specific configuration at the top for easy customization

Goal: Build a cleaner, more focused KG with better entity classification and relationship quality.
"""


import csv
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv


# NOTE: Customize these for this experiment
# - graph_id: set the target graph for ingestion (e.g., "baseline_retriever_v2")
# - csv_paths: set the input CSV files for this run
# - batch_size: keep at or below 20 to satisfy API limits
GRAPH_ID: str = "baseline_v2_experiment"  # <-- set for this experiment
CSV_PATHS: List[str] = [
    "data/raw/try_it_draft_contents.csv",
    "data/raw/example_draft_contents (3).csv",
    "data/raw/concept_draft_contents (4).csv",
]
BATCH_SIZE: int = 20


@dataclass
class CSVEpisode:
    """
    Represents a CSV row converted to a Zep episode.

    Fields:
        lo_id: Learning objective ID
        content: Original content field (unprocessed)
        episode_type: concept|example|exercise|try-it
        metadata: Book/unit/chapter and other attributes
        raw_content: Parsed JSON from raw_content column
    """
    lo_id: str
    content: str
    episode_type: str
    metadata: Dict[str, Any]
    raw_content: Dict[str, Any]

    def to_zep_episode(self) -> Dict[str, Any]:
        """
        Convert to Zep episode payload.

        Adds a short schema hint header to improve extraction quality and
        keep relationships constrained in the baseline_v2 experiment.

        For this experiment, we gently hint the extractor to use a small set
        of relationships and prefer classifying concepts correctly.
        """
        # --- Schema/relationship hint header (experiment-specific guidance) ---
        # Keep hints brief to avoid overfitting. Adjust as needed per experiment.
        schema_hint = (
            "[SCHEMA_HINT: entity_types={Concept, Example, Exercise, TryIt}; "
            "relationships={PREREQUISITE_OF, PART_OF, ASSESSED_BY}]\n"
        )

        # Build readable text from raw_content
        content_parts: List[str] = []
        if "problem" in self.raw_content:
            content_parts.append(f"Problem: {self.raw_content['problem']}")

        if "solution" in self.raw_content and isinstance(self.raw_content["solution"], dict):
            steps = self.raw_content["solution"].get("steps")
            if isinstance(steps, list) and steps:
                content_parts.append("\nSolution:")
                for idx, step in enumerate(steps, 1):
                    # Each step expected to be {"step": "..."}
                    if isinstance(step, dict) and "step" in step:
                        content_parts.append(f"{idx}. {step['step']}")

        full_content: str = "\n\n".join(content_parts)

        # Minimal metadata header to aid context
        metadata_header = (
            f"[LO: {self.lo_id}, Type: {self.episode_type}, "
            f"Unit: {self.metadata.get('unit', 'N/A')}, "
            f"Chapter: {self.metadata.get('chapter', 'N/A')}]\n\n"
        )

        final_text = schema_hint + metadata_header + full_content
        return {"data": final_text, "type": "text"}


class CSVToEpisodeConverter:
    """Converts CSV rows to CSVEpisode objects."""

    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path

    def convert_csv_to_episodes(self) -> List[CSVEpisode]:
        episodes: List[CSVEpisode] = []
        with open(self.csv_file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                episode = self._convert_row_to_episode(row)
                if episode:
                    episodes.append(episode)
        return episodes

    def _convert_row_to_episode(self, row: Dict[str, str]) -> Optional[CSVEpisode]:
        # Parse raw_content JSON if available; fallback to plain text
        raw_content: Dict[str, Any]
        try:
            raw_content = json.loads(row.get("raw_content", ""))
            if not isinstance(raw_content, dict):
                raw_content = {"problem": row.get("raw_content", "")}
        except json.JSONDecodeError:
            raw_content = {"problem": row.get("raw_content", "")}

        metadata = {
            "book": row.get("book", ""),
            "learning_objective": row.get("learning_objective", ""),
            "unit": row.get("unit", ""),
            "chapter": row.get("chapter", ""),
        }

        lo_id = row.get("lo_id", "")
        episode_type = row.get("type", "").strip().lower()

        if not lo_id or not episode_type:
            return None

        return CSVEpisode(
            lo_id=lo_id,
            content=row.get("raw_content", ""),
            episode_type=episode_type,
            metadata=metadata,
            raw_content=raw_content,
        )


class ZepKnowledgeGraphBuilder:
    """
    Adds episodes to Zep's knowledge graph.

    Experiment-specific fields to customize:
      - GRAPH_ID (set above)
      - CSV_PATHS (set above)
      - BATCH_SIZE (set above)
    """

    def __init__(self, api_key: str):
        try:
            from zep_cloud import Zep
            self.client = Zep(api_key=api_key)
        except ImportError:
            print("Error: zep-cloud package not installed. Run: pip install zep-cloud")
            self.client = None

    def add_batch_episodes(self, episodes: List[CSVEpisode], graph_id: Optional[str]) -> Dict[str, Any]:
        if not self.client:
            print("Zep client not initialized")
            return {}

        if not graph_id:
            print("Error: graph_id is required for group graph ingestion")
            return {}

        payload = []
        for ep in episodes:
            e = ep.to_zep_episode()
            payload.append({"data": e["data"], "type": e["type"]})

        try:
            response = self.client.graph.add_batch(graph_id=graph_id, episodes=payload)
            return response
        except Exception as e:
            print(f"Error adding batch episodes: {e}")
            return {}


def load_all_episodes(csv_paths: List[str]) -> List[CSVEpisode]:
    all_eps: List[CSVEpisode] = []
    for path in csv_paths:
        try:
            converter = CSVToEpisodeConverter(path)
            eps = converter.convert_csv_to_episodes()
            print(f"Loaded {len(eps)} episodes from {path}")
            all_eps.extend(eps)
        except FileNotFoundError:
            print(f"CSV file not found at: {path} — skipping")
    return all_eps


def optionally_balance_types(episodes: List[CSVEpisode], max_per_type: Optional[int] = None) -> List[CSVEpisode]:
    """
    Optional down-sampling to prevent a skewed graph by type.
    - Set max_per_type (e.g., 200) to roughly balance concept/example/exercise/try-it.
    - Leave as None to keep all episodes.
    """
    if not max_per_type:
        return episodes

    per_type_counts: Dict[str, int] = {}
    balanced: List[CSVEpisode] = []
    for ep in episodes:
        t = ep.episode_type
        c = per_type_counts.get(t, 0)
        if c < max_per_type:
            balanced.append(ep)
            per_type_counts[t] = c + 1
    print(f"Type balancing applied (max {max_per_type} per type). Final count: {len(balanced)}")
    return balanced


def ensure_graph_exists(builder: ZepKnowledgeGraphBuilder, graph_id: str) -> None:
    print(f"Ensuring graph exists: {graph_id}...")
    try:
        graph = builder.client.graph.create(graph_id=graph_id)
        print(f"Graph created: {graph}")
    except Exception as e:
        # Likely already exists; proceed
        print(f"Graph create returned: {e}. Proceeding.")


def run_ingestion():
    # 1) Load environment
    load_dotenv()
    api_key = os.getenv("ZEP_API_KEY")
    if not api_key:
        print("Error: ZEP_API_KEY not found in environment variables")
        print("Create .env with: ZEP_API_KEY=your_key_here")
        return

    # 2) Init client
    builder = ZepKnowledgeGraphBuilder(api_key)

    # 3) Load episodes from CSVs
    episodes = load_all_episodes(CSV_PATHS)
    if not episodes:
        print("No episodes found. Aborting.")
        return

    # 4) Optional balancing (set max_per_type to enable)
    # NOTE: Enable this if examples dominate concepts. Tune per experiment.
    # episodes = optionally_balance_types(episodes, max_per_type=200)

    # 5) Ensure target graph exists
    ensure_graph_exists(builder, GRAPH_ID)

    # 6) Ingest in batches
    total = len(episodes)
    added = 0
    print(f"Adding {total} episodes to graph: {GRAPH_ID}")
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = episodes[start:end]
        try:
            builder.add_batch_episodes(batch, graph_id=GRAPH_ID)
            added += len(batch)
            print(f"Progress: {added}/{total}")
        except Exception as e:
            print(f"Batch {start}-{end} failed: {e}")
            continue

    print("Done.")


if __name__ == "__main__":
    run_ingestion()


