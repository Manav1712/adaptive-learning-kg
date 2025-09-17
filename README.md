# Adaptive Learning Knowledge Graph Platform

A local-first adaptive learning system that uses Zep's automated knowledge graph construction to provide personalized tutoring, built on OpenStax content with GPT-4 agents.

## Project Overview

This platform creates an adaptive learning system that:
- Builds a knowledge graph from OpenStax mathematics content
- Provides personalized tutoring through multiple AI agents
- Tracks student progress and adapts content accordingly
- Uses local-first architecture for privacy and portability

## Architecture

See the [architecture/](architecture/) folder for detailed design documents:
- [Phases](architecture/phases.md) - Development roadmap
- [Implementation Plan](architecture/implementation.md) - Technical implementation details
- [Sequence Diagrams](architecture/sequence_diagrams.md) - System interactions

## Progress Report

### Phase 1: Static Knowledge Graph — Status

- Baselines built using Zep ingestion experiments (V1–V4)
- Ontology- and schema-guided ingestion validated (Baseline V3/V4)
- Evaluation scripts in place to measure structure and retrieval

### Current Focus

- Compare two approaches on the same subset:
  - Zep automated ingestion with constrained ontology (Coach/Retriever baselines)
  - LLM-driven edge discovery (multimodal) to build a Retriever graph with LO→LO and content↔LO edges
- Decide on production direction based on quality, effort, and cost

### Recent Progress (Sept 2025)

- Tightened content discovery: threshold 0.85; pruning `same_unit=true`, `same_chapter=true`.
- Added strict LLM evaluator (`src/experiments_manual/evaluate_with_llm.py`): JSON-only, retries, progress; 3-labels for prereqs and content.
- Regenerated and evaluated content links with 0% JSON parse errors.

Run:

```bash
# Prerequisites (LO→LO)
python3 src/experiments_manual/evaluate_with_llm.py \
  --edges-in data/processed/edges_prereqs.csv \
  --jsonl-out data/processed/llm_edge_checks.jsonl \
  --summary-out data/processed/llm_edge_checks_summary.json

# Content links (LO→Content)
python3 src/experiments_manual/evaluate_with_llm.py \
  --edges-in data/processed/edges_content.csv \
  --jsonl-out data/processed/llm_edge_checks_content_final.jsonl \
  --summary-out data/processed/llm_edge_checks_content_final_summary.json
```

### Phase 2: Manual LLM-based Graph (experiments_manual) — Status

- **Scope**: Offline pipeline to discover edges using an LLM with multimodal prompts (text + image URLs)
- **Inputs**: `data/processed/lo_index.csv`, `data/processed/content_items.csv` (from `src/experiments_manual/prepare_lo_view.py`)
- **Config**: `src/experiments_manual/config.yaml`
  - **model**: `gpt-4o-mini`, **modality**: `multimodal`, **threshold**: `0.6`, **batch**: `8`, **retries**: `3`
- **Discovery scripts**:
  - Content → LO: `src/experiments_manual/discover_content_links.py`
  - LO → LO prerequisites: `src/experiments_manual/discover_prereqs.py`
  - Both support: sharding via `--num-shards N --shard-index i`, progress logs (`progress_links.jsonl`, `progress_prereqs.jsonl`), heuristic fallback
- **Evaluation**: `src/experiments_manual/evaluate_outputs.py` (edge counts, score stats, integrity, coverage, top-N)
- **Visualization**: `src/experiments_manual/build_and_visualize.py` → `data/processed/graph_preview.html` (labeled nodes/edges + legend)
- **Report**: `src/experiments_manual/report_offline_multimodal_sample100.md` (sample-100 multimodal run summary)

#### Repro (offline, multimodal)

```bash
# Prepare processed inputs (LO index + content items)
python3 src/experiments_manual/prepare_lo_view.py

# Content → LO: candidates + scoring (sample via --limit)
python3 src/experiments_manual/discover_content_links.py \
  --config src/experiments_manual/config.yaml --mode both --limit 100

# LO → LO prerequisites: candidates + scoring (sample via --limit)
python3 src/experiments_manual/discover_prereqs.py \
  --config src/experiments_manual/config.yaml --mode both --limit 100

# Evaluate (human-readable summary; add --json-out for machine output)
python3 src/experiments_manual/evaluate_outputs.py --edges data/processed/edges_content.csv --top-n 5
python3 src/experiments_manual/evaluate_outputs.py --edges data/processed/edges_prereqs.csv --top-n 5

# Build interactive HTML preview
python3 src/experiments_manual/build_and_visualize.py --out data/processed/graph_preview.html
```

#### Notes

- Multimodal prompts include image URLs when available; JSON-only responses are enforced.
- Sharding deterministically splits work (CRC32) and auto-suffixes output/progress paths per shard.
- Heuristic scoring provides deterministic fallback when LLM calls are unavailable or fail.

## Development Setup

1. Create virtual environment: `python3 -m venv venv`
2. Activate: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Set environment variables:
   - `export ZEP_API_KEY=your_key_here`
   - For notebooks/POC only: `export OPENAI_API_KEY=your_key_here` (do not hardcode in notebooks)
5. Run experiments:
   - Ingestion (ontology): `python src/experiments/baseline_v3/ingestion_v3.py`
   - Evaluation (v3): `python src/experiments/baseline_v3/evaluation_v3.py`
   - Minimal/pruned ingest (v4): `python src/experiments/baseline_v4/ingestion_v4.py`
   - Evaluation (v4): `python src/experiments/baseline_v4/evaluation_v4.py`

## Project Structure (Phase 1)

```
src/
├── experiments/           # Iterative experiment framework
│   ├── baseline_v1/      # Initial KG construction (474 nodes, 493 edges)
│   │   ├── ingestion_v1.py
│   │   ├── evaluation_v1.py
│   │   └── retrieval_baseline_results.md
│   ├── baseline_v2/      # Schema hints experiment (270 episodes, 24.6% effectiveness)
│   │   ├── ingestion_v2.py
│   │   ├── evaluation_v2.py
│   │   └── retrieval_baseline_v2_results.md
│   ├── baseline_v3/      # Ontology enforcement experiment
│   │   ├── ingestion_v3.py
│   │   └── evaluation_v3.py
│   └── baseline_v4/      # Minimal ingestion with light pruning
│       ├── ingestion_v4.py
│       └── evaluation_v4.py
├── processing/           # Content processing utilities
├── evaluation/           # KG evaluation utilities
└── retrieval/            # Retrieval package scaffold

data/                    # Data storage (raw/processed)
tests/                   # Basic testing
architecture/            # Design documents and specifications
```

## Technology Stack

- **Language**: Python 3.11+
- **LLM**: OpenAI GPT-4
- **Knowledge Graph**: Zep (automated construction)
- **Storage**: Local JSON + SQLite (extensible to Zep cloud)
- **Vector Search**: Sentence-transformers + FAISS
- **Frontend**: Vanilla JS + D3.js
- **Architecture**: Multi-agent system with Zep integration
 
