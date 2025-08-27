"""
BASELINE V4: Minimal ingestion with light pruning & tighter schema hints

Goals:
- Keep ingestion simple (no ontology enforcement). 
- Light pruning: keep only Concept & Exercise episodes.
- Tight schema hints to bias extractor toward PREREQUISITE_OF and ASSESSED_BY.
"""

import os
import csv
import json
from typing import List, Dict, Any
from dotenv import load_dotenv


GRAPH_ID = "baseline_v4_pruned_schema"
CSV_PATHS = [
    "data/raw/try_it_draft_contents.csv",
    "data/raw/example_draft_contents (3).csv",
    "data/raw/concept_draft_contents (4).csv",
]
BATCH_SIZE = 20


def to_episode(row: Dict[str, str]) -> Dict[str, Any]:
    lo_id = (row.get("lo_id") or "").strip()
    ep_type = (row.get("type") or "").strip().lower()
    if not lo_id or ep_type not in {"concept", "exercise"}:  # prune to Concept/Exercise
        return {}

    # Prefer problem/solution text if available
    try:
        raw = json.loads(row.get("raw_content", "") or "{}")
    except json.JSONDecodeError:
        raw = {"problem": row.get("raw_content", "")}

    unit = row.get("unit", ""); chapter = row.get("chapter", "")
    header = f"[LO: {lo_id}, Type: {ep_type}, Unit: {unit}, Chapter: {chapter}]\n\n"

    parts: List[str] = []
    if isinstance(raw, dict) and raw.get("problem"):
        parts.append(f"Problem: {raw['problem']}")
    sol = raw.get("solution") if isinstance(raw, dict) else None
    if isinstance(sol, dict) and isinstance(sol.get("steps"), list):
        parts.append("\nSolution:")
        for i, step in enumerate(sol["steps"], 1):
            if isinstance(step, dict) and step.get("step"):
                parts.append(f"{i}. {step['step']}")

    body = "\n\n".join(parts)

    # Tight schema hint to bias extraction
    hint = (
        "[SCHEMA_HINT: Entities(Concept, Exercise); "
        "Edges(PREREQUISITE_OF Concept->Concept, ASSESSED_BY Exercise->Concept)]\n\n"
    )

    return {"type": "text", "data": hint + header + body}


def load_episodes() -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for path in CSV_PATHS:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ep = to_episode(row)
                if ep:
                    episodes.append(ep)
    return episodes


def run_ingestion():
    load_dotenv()
    from zep_cloud.client import Zep

    api_key = os.getenv("ZEP_API_KEY")
    if not api_key:
        print("Error: ZEP_API_KEY not set"); return

    client = Zep(api_key=api_key)
    try:
        client.graph.create(graph_id=GRAPH_ID)
    except Exception as e:
        print(f"Graph create: {e} (continuing)")

    episodes = load_episodes()
    if not episodes:
        print("No episodes after pruning"); return

    print(f"Submitting {len(episodes)} episodes to {GRAPH_ID}...")
    added = 0
    for i in range(0, len(episodes), BATCH_SIZE):
        batch = episodes[i:i+BATCH_SIZE]
        try:
            client.graph.add_batch(graph_id=GRAPH_ID, episodes=batch)
            added += len(batch)
            print(f"Progress: {added}/{len(episodes)}")
        except Exception as e:
            print(f"Batch {i}-{i+len(batch)} failed: {e}")

    print("Done submitting. Processing will complete asynchronously in Zep.")


if __name__ == "__main__":
    run_ingestion()



