# Adaptive Learning Knowledge Graph Platform

A local-first adaptive learning system that uses a knowledge graph to provide personalized tutoring, built on OpenStax content with GPT-4 agents.

## Project Overview

This platform creates an adaptive learning system that:
- Builds a static knowledge graph from OpenStax mathematics content
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
- Project structure created
- Basic data models implemented
- Dependencies defined
- Git workflow established (feature branches + main)
- Zep MCP server configured for documentation access

**Next Steps**:
- Implement OpenStax corpus ingestion
- Build entity extraction pipeline with GPT-4
- Create knowledge graph construction system
- Develop basic visualization capabilities

**Target**: Static KG with 500+ nodes, visualizable in D3

## Development Setup

1. Create virtual environment: `python3 -m venv venv`
2. Activate: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

## Project Structure

```
src/
├── agents/          # AI agent implementations
├── data/models/     # Knowledge graph data models
├── processing/      # Data ingestion and extraction
└── retrieval/       # Search and retrieval components

data/
├── raw/            # OpenStax source files
├── processed/      # Extracted and cleaned data
└── storage/        # Knowledge graph and session data
```

## Technology Stack

- **Language**: Python 3.11+
- **LLM**: OpenAI GPT-4
- **Storage**: Local JSON + SQLite (extensible to Zep/Neo4j)
- **Vector Search**: Sentence-transformers + FAISS
- **Frontend**: Vanilla JS + D3.js
- **Architecture**: Multi-agent system with local-first design