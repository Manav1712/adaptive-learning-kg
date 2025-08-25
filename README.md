# Adaptive Learning Knowledge Graph Platform

A local-first adaptive learning system that uses Zep's automated knowledge graph construction to provide personalized tutoring, built on OpenStax content with GPT-4 agents.

## Project Overview

This platform creates an adaptive learning system that:
- Builds a static knowledge graph from OpenStax mathematics content using Zep's automated entity extraction
- Provides personalized tutoring through multiple AI agents
- Tracks student progress and adapts content accordingly
- Uses local-first architecture for privacy and portability

## Architecture

See the [architecture/](architecture/) folder for detailed design documents:
- [Phases](architecture/phases.md) - Development roadmap
- [Implementation Plan](architecture/implementation.md) - Technical implementation details
- [Sequence Diagrams](architecture/sequence_diagrams.md) - System interactions
- [Minimal Contracts](architecture/minimal_contracts.md) - Agent interfaces

## Progress Report

### Phase 1: Static Knowledge Graph (Weeks 1-3) - COMPLETED ✅

**Current Status**: Baseline experiments completed
- ✅ Project structure created for Phase 1
- ✅ Basic data models implemented  
- ✅ Dependencies defined and virtual environment configured
- ✅ Git workflow established (feature branches + main)
- ✅ Zep MCP server configured for documentation access
- ✅ **Baseline V1**: Built initial KG with 474 nodes, 493 edges (474 relationships)
- ✅ **Baseline V2**: Improved KG with schema hints and relationship constraints (270 episodes)
- ✅ Evaluation framework implemented for both experiments
- ✅ Experiment structure organized for iterative development

**Key Achievements**:
- Successfully ingested OpenStax content into Zep knowledge graphs
- Implemented comprehensive evaluation metrics (node/edge counts, retrieval quality, content coverage)
- Created reproducible experiment framework with ingestion/evaluation scripts
- Identified areas for improvement: entity type classification, relationship constraints, content balancing

**Next Phase**: Enhanced KG Construction
- Implement custom entity types using Zep's `set_ontology()` method
- Apply relationship constraints before ingestion
- Balance content distribution (concepts vs examples)
- Scale to full dataset once received

**Target**: Clean, focused KG with 1000+ nodes, constrained relationships, balanced content types

## Development Setup

1. Create virtual environment: `python3 -m venv venv`
2. Activate: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Set environment variable: `export ZEP_API_KEY=your_key_here`
5. Run: `python src/main.py`

## Project Structure (Phase 1)

```
src/
├── experiments/       # Iterative experiment framework
│   ├── baseline_v1/  # Initial KG construction (474 nodes, 493 edges)
│   │   ├── ingestion_v1.py
│   │   └── evaluation_v1.py
│   └── baseline_v2/  # Improved KG with schema hints (270 episodes)
│       ├── ingestion_v2.py
│       └── evaluation_v2.py
├── processing/        # Content processing utilities
├── data/models/      # Basic data models
├── evaluation/       # KG evaluation framework
├── zep_client.py     # Zep integration utilities
└── main.py           # Phase 1 entry point

data/                 # Data storage (raw CSV files)
tests/                # Basic testing
architecture/          # Design documents and specifications
```

## Technology Stack

- **Language**: Python 3.11+
- **LLM**: OpenAI GPT-4
- **Knowledge Graph**: Zep (automated construction)
- **Storage**: Local JSON + SQLite (extensible to Zep cloud)
- **Vector Search**: Sentence-transformers + FAISS
- **Frontend**: Vanilla JS + D3.js
- **Architecture**: Multi-agent system with Zep integration