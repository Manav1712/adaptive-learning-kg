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
 
