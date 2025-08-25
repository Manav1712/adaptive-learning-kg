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

### Phase 1: Static Knowledge Graph (Weeks 1-3) - IN PROGRESS 🚧

**Current Status**: Advanced ontology experiments underway
- ✅ Project structure created for Phase 1
- ✅ Basic data models implemented  
- ✅ Dependencies defined and virtual environment configured
- ✅ Git workflow established (feature branches + main)
- ✅ Zep MCP server configured for documentation access
- ✅ **Baseline V1**: Built initial KG with 474 nodes, 493 edges (baseline performance)
- ✅ **Baseline V2**: Improved KG with schema hints (270 episodes, 24.6% constraint effectiveness)
- ✅ **Baseline V3**: Ontology-enforced ingestion with fact rating (270 episodes, processing overnight)
- ✅ Evaluation framework implemented for all experiments
- ✅ Experiment structure organized for iterative development
- ✅ Fixed ingestion pipeline with proper rate limit handling

**Key Achievements**:
- Successfully ingested OpenStax content into Zep knowledge graphs across multiple experiments
- Implemented comprehensive evaluation metrics (node/edge counts, retrieval quality, content coverage, relationship constraint analysis)
- Created reproducible experiment framework with robust ingestion/evaluation scripts
- Solved async processing issues and rate limiting on Zep's FREE plan
- Applied custom ontology enforcement using Zep's `set_ontology()` method with entity/edge constraints
- Identified that schema hints alone are insufficient (24.6% effectiveness vs target 80%+)

**Current Investigation**: Baseline V3 Ontology Enforcement
- Custom entity types: Concept, Example, Exercise, TryIt with property definitions
- Constrained edge types: PREREQUISITE_OF, PART_OF, ASSESSED_BY with source/target restrictions
- Fact rating instruction applied to filter low-relevance relationships
- Type balancing enabled (250 max per type) to ensure concept dominance
- **Status**: Processing overnight on Zep FREE plan (2-6 hour processing time for 270 episodes)

**Next Steps**: 
- Evaluate V3 results for constraint effectiveness and edge type reduction
- If ontology enforcement successful → Ready for production merge
- If still high edge-type noise → Investigate alternative constraint approaches
- Scale to full dataset once approach validated

**Target**: Production-ready KG with constrained relationships (3-5 edge types vs current 200+), 80%+ constraint effectiveness

## Development Setup

1. Create virtual environment: `python3 -m venv venv`
2. Activate: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Set environment variable: `export ZEP_API_KEY=your_key_here`
5. Run: `python src/main.py`

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
│   └── baseline_v3/      # Ontology enforcement experiment (270 episodes, processing)
│       ├── ingestion_v3.py
│       └── evaluation_v3.py
├── processing/           # Content processing utilities
├── data/models/         # Basic data models  
├── evaluation/          # KG evaluation framework
├── zep_integration/     # Zep client and utilities
└── main.py              # Phase 1 entry point

data/                    # Data storage (raw CSV files)
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