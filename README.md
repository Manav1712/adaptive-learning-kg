# Adaptive Learning Knowledge Graph Platform

A local-first adaptive learning system that manually generates knowledge graph nodes and edges from OpenStax content, leverages Zep's temporal layer for conversation memory, and provides personalized tutoring through a RAG system with Retriever, Coach Generator, and Tutor agents.

## Project Overview

This platform creates an adaptive learning system that:
- **Manually generates** knowledge graph nodes (Learning Objectives, Content Items) and edges (prerequisites, content relationships)
- **Uses Zep's temporal layer** for conversation memory and context management
- **Implements a RAG system** with three specialized agents:
  - **Retriever**: Finds relevant content and learning paths
  - **Coach Generator**: Creates personalized learning strategies
  - **Tutor**: Provides interactive tutoring sessions
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

- **Manual Knowledge Graph Construction**: Using LLM-driven edge discovery to build high-quality relationships
- **Graph Validation**: Comprehensive evaluation of edge quality through structural analysis and semantic LLM validation
- **RAG System Development**: Building Retriever, Coach Generator, and Tutor agents that leverage the manually constructed knowledge graph
- **Zep Integration**: Utilizing Zep's temporal layer for conversation memory and context management

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

### Phase 2: Manual Knowledge Graph Construction — Status

- **Scope**: Offline pipeline to manually generate high-quality knowledge graph nodes and edges
- **Node Generation**: Learning Objectives and Content Items extracted from OpenStax mathematics content
- **Edge Discovery**: LLM-driven discovery of relationships using multimodal prompts (text + image URLs)
  - **Prerequisites**: LO → LO relationships (learning dependencies)
  - **Content Links**: LO → Content relationships (explained_by, exemplified_by, practiced_by)
- **Inputs**: `data/processed/lo_index.csv`, `data/processed/content_items.csv` (from `src/experiments_manual/prepare_lo_view.py`)
- **Config**: `src/experiments_manual/config.yaml`
  - **model**: `gpt-4o-mini`, **modality**: `multimodal`, **threshold**: `0.85`, **batch**: `8`, **retries**: `3`
- **Discovery Scripts**:
  - Content → LO: `src/experiments_manual/discover_content_links.py`
  - LO → LO prerequisites: `src/experiments_manual/discover_prereqs.py`
  - Both support: sharding via `--num-shards N --shard-index i`, progress logs, heuristic fallback
- **Evaluation Pipeline**:
  - **Structural Analysis**: `src/experiments_manual/evaluate_outputs.py` (integrity, cycles, coverage, graph stats)
  - **Semantic Validation**: `src/experiments_manual/evaluate_with_llm.py` (LLM-based edge correctness)
- **Visualization**: `src/experiments_manual/build_and_visualize.py` → `data/processed/graph_preview.html`

#### Knowledge Graph Construction Pipeline

```bash
# 1. Prepare processed inputs (LO index + content items)
python3 src/experiments_manual/prepare_lo_view.py

# 2. Discover Content → LO relationships
python3 src/experiments_manual/discover_content_links.py \
  --config src/experiments_manual/config.yaml --mode both

# 3. Discover LO → LO prerequisite relationships  
python3 src/experiments_manual/discover_prereqs.py \
  --config src/experiments_manual/config.yaml --mode both

# 4. Structural validation (integrity, cycles, coverage)
python3 src/experiments_manual/evaluate_outputs.py --edges data/processed/edges_content.csv
python3 src/experiments_manual/evaluate_outputs.py --edges data/processed/edges_prereqs.csv

# 5. Semantic validation (LLM-based edge correctness)
python3 src/experiments_manual/evaluate_with_llm.py \
  --edges-in data/processed/edges_prereqs.csv \
  --jsonl-out data/processed/llm_edge_checks.jsonl \
  --summary-out data/processed/llm_edge_checks_summary.json

python3 src/experiments_manual/evaluate_with_llm.py \
  --edges-in data/processed/edges_content.csv \
  --jsonl-out data/processed/llm_edge_checks_content_final.jsonl \
  --summary-out data/processed/llm_edge_checks_content_final_summary.json

# 6. Build interactive HTML visualization
python3 src/experiments_manual/build_and_visualize.py --out data/processed/graph_preview.html
```

#### Key Features

- **Multimodal Discovery**: LLM prompts include both text content and image URLs when available
- **Strict Validation**: JSON-only responses enforced with robust parsing and retry logic
- **Scalable Processing**: Sharding support for large-scale edge discovery with deterministic work splitting
- **Comprehensive Evaluation**: Two-tier validation (structural + semantic) ensures high-quality relationships
- **Production Ready**: Heuristic fallbacks and error handling for reliable operation

#### Next Steps: RAG System Development

The manually constructed knowledge graph serves as the foundation for the RAG system:

1. **Retriever Agent**: Uses the knowledge graph to find relevant content and learning paths
2. **Coach Generator**: Creates personalized learning strategies based on student progress and graph relationships  
3. **Tutor Agent**: Provides interactive tutoring sessions with context from Zep's temporal layer
4. **Integration**: All agents leverage the validated knowledge graph for accurate, contextual responses

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
- **LLM**: OpenAI GPT-4 (GPT-4o-mini for edge discovery and validation)
- **Knowledge Graph**: Manually constructed nodes and edges from OpenStax content
- **Temporal Memory**: Zep's temporal layer for conversation context and memory
- **Storage**: Local JSON + CSV (extensible to Zep cloud)
- **Vector Search**: Sentence-transformers + FAISS
- **Frontend**: Vanilla JS + D3.js
- **Architecture**: Multi-agent RAG system with Retriever, Coach Generator, and Tutor agents
 
