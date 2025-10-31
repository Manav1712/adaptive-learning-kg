# Coach-Retriever Architecture

## Overview

This package implements the Phase 2 Coach-Retriever system based on the architecture defined in `design.md`.

## Components

### Coach (`coach.py`)
Front-door interface for user interactions.

**Responsibilities:**
- Classify user intent (tutoring, practice, definition, comparison, troubleshooting)
- Build structured retrieval requests
- Handle session management (continue/branch/switch)
- Evaluate retrieval results and decide on actions
- Ask clarifying questions when needed

**Main Entry Point:** `process_turn(user_input, student_id, session_id)`

### Retriever (`retriever.py`)
Multi-stage retrieval pipeline.

**Responsibilities:**
- **Stage A**: Hybrid recall (dense + lexical search)
- **Stage B**: Re-ranking + diversity (MMR)
- **Stage C**: Graph-aware expansion (prerequisite traversal + content alignment)
- Context compression (3-7 bullet facts)

**Main Entry Point:** `retrieve(request: RetrievalRequest)`

### Models (`models.py`)
Data models for Coach-Retriever I/O contracts.

**Key Models:**
- `RetrievalRequest`: Structured request from Coach to Retriever
- `RetrievalResponse`: Response with matched LOs, content, context, and citations
- `CoachDecision`: Decision made by Coach (answer, ask clarification, etc.)
- `MatchedLO`, `SupportingLO`, `ContentItem`: Retrieved entities
- `RetrievalConstraints`: Budgets and filters
- `RetrievalTelemetry`: Performance metrics

## Usage Example

```python
import pandas as pd
from src.agents import Coach, Retriever

# Load knowledge graph data
los_df = pd.read_csv("data/processed/lo_index.csv")
content_df = pd.read_csv("data/processed/content_items.csv")
edges_df = pd.read_csv("data/processed/edges_prereqs.csv")

# Initialize Retriever
retriever = Retriever(
    los_df=los_df,
    content_df=content_df,
    edges_df=edges_df
)

# Initialize Coach with Retriever
coach = Coach(retriever=retriever)

# Process user query
decision = coach.process_turn(
    user_input="How do I solve a quadratic equation?",
    student_id="student_123",
    session_id=None
)

# Use decision.response for display
# Use decision.retrieval_request for downstream processing
```

## Architecture Alignment

This implementation follows the `design.md` specification:

✅ **Just-enough context**: Returns 3-7 bullet facts with hard token budget
✅ **Graph-aware retrieval**: Traverses PREREQUISITE_OF and ASSESSED_BY edges
✅ **Safety and precision**: Constrains to KG; attributes all facts to sources
✅ **Multi-stage pipeline**: Stage A (recall) → Stage B (precision) → Stage C (expansion)
✅ **Context compression**: Selective extraction with citations
✅ **Guardrails**: No hallucinations; LLM only for rewriting/reranking

## Phase 2 Implementation Notes

Current implementation uses simple keyword matching for retrieval (placeholder for dense embeddings + BM25).

**Placeholder implementations:**
- `_stage_a_hybrid_recall`: Simple keyword matching (will be replaced with FAISS + BM25)
- `_stage_b_rerank_diversity`: Score boosting + simple MMR (will be replaced with cross-encoder)
- Session storage: In-memory (will be replaced with file-based or SQLite)

**Ready for enhancement:**
- Add FAISS index for dense embeddings
- Add BM25/SPLADE for lexical retrieval
- Add cross-encoder re-ranking
- Add LLM-based intent classification
- Add persistent session storage

## Next Steps

1. **Integrate existing indexes**: Add FAISS embeddings and BM25 indexes from experiments
2. **Test end-to-end**: Run Coach → Retriever → response flow with demo queries
3. **Add Tutor agent**: Create Pack Constructor + Tutor to generate final responses
4. **Evaluation**: Measure recall@k, precision@k, response time
