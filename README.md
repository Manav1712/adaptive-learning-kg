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

### Phase 1: Static Knowledge Graph (Weeks 1-3) - IN PROGRESS

**Current Status**: Project setup and foundation
- Project structure created for Phase 1
- Basic data models implemented
- Dependencies defined
- Git workflow established (feature branches + main)
- Zep MCP server configured for documentation access

**Next Steps**:
- Implement OpenStax corpus ingestion
- Integrate with Zep's automated KG construction
- Build basic knowledge graph
- Develop simple visualization

**Target**: Static KG with 500+ nodes, visualizable in D3

## Development Setup

1. Create virtual environment: `python3 -m venv venv`
2. Activate: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Set environment variable: `export ZEP_API_KEY=your_key_here`
5. Run: `python src/main.py`

## Project Structure (Phase 1)

```
src/
├── processing/         # OpenStax content ingestion
├── data/models/       # Basic data models
├── zep_client.py      # Simple Zep integration
└── main.py            # Phase 1 entry point

data/                  # Data storage
tests/                 # Basic testing
```

## Technology Stack

- **Language**: Python 3.11+
- **LLM**: OpenAI GPT-4
- **Knowledge Graph**: Zep (automated construction)
- **Storage**: Local JSON + SQLite (extensible to Zep cloud)
- **Vector Search**: Sentence-transformers + FAISS
- **Frontend**: Vanilla JS + D3.js
- **Architecture**: Multi-agent system with Zep integration