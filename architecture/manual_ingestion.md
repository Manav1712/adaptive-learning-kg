# Manual LLM-Based Graph Construction

## Overview
Alternative approach to Zep auto-ingestion: use multimodal LLMs to discover edges between predefined Learning Objective nodes and content pieces. Focuses on pedagogical accuracy through explicit reasoning rather than embedding similarity.

## Core Strategy

### Coach Graph (LO → LO Prerequisites)
- **Dataset**: Sample 100 LOs from one book (as in LOs_combined.csv subset)
- **Inputs**: LO text + normalized messages_json (text + image_url blocks)
- **Variants**:
  - **Modality**: text-only vs multimodal (images included)
  - **Prompt**: strict YES/NO vs scored (0–1) + threshold (e.g., ≥0.7)
  - **Candidate set**: N×N vs pruned shortlist (same chapter/unit + top-K lexical match)
  - **Model**: gpt-4.1-mini vs gpt-4.1 (temperature 0)
- **Rules**: prevent reverse-edge cycles; keep book edges; mark extras as inferred
- **Outputs**: LO_edges.csv; metrics = book-edge recall, extra-edge rate, cycles (0), cross-chapter %, avg degree

### Retriever Graph (Content ↔ LO)
- **Nodes**: concept/example/try_it, unique IDs; auto-edge to its own LO
- **Cross-LO mapping**: LLM decides if content "directly helps" target LO; relation by type:
  - concept → LO: explained_by
  - example → LO: exemplified_by
  - try_it → LO: practiced_by
- **Variants**:
  - **Modality**: text-only vs multimodal
  - **Gating**: YES/NO vs scored threshold
  - **Candidate pruning**: target LOs from same unit/chapter + top-K lexical shortlist
- **Outputs**: final_nodes.csv, final_edges.csv; metrics = % content linked, targets per content, cross-chapter %, edge distribution

## Unified Graph + Visuals
- **Merge**: LO→LO + content edges → final_graph_nodes.csv, final_graph_edges.csv
- **Views**: full `final_graph.html` and sampled `final_graph_sampled.html`
- **Structural metrics** (networkx): degree distribution, connected components, avg path length, cluster coefficient

## Implementation Details

### Cost, Logging, and Resume
- Temperature 0; progress files (progress.json, progress_content.json)
- Batch prompts (pack multiple pairs per call when possible) to cut calls
- Run matrix on 20 LOs first, then 100 LOs

### Code Structure
New experiment folder: `src/experiments/llm_edges_v1/` with:
- `prepare_lo_view.py` (port notebook's LOs_combined logic, image block normalization)
- `discover_prereqs.py` (coach graph)
- `discover_content_links.py` (retriever edges)
- `build_and_visualize.py` (merge + HTML + metrics)
- `config.yaml` (model, modality, pruning, thresholds, sample size)

## Decision Criteria
- Recover book edges ≥ target recall
- Low spurious cross-chapter edges
- Clear, navigable visuals
- Acceptable call cost/time
- Compare 4 cells: {text, multimodal} × {N×N, pruned} on same LO sample

## Advantages Over Auto-Ingestion
1. **Pedagogical Accuracy**: Explicit reasoning about prerequisites vs. embedding similarity
2. **Multimodal Understanding**: Can process images and text together
3. **Cross-Linking**: Content can naturally connect to multiple LOs
4. **Controlled Discovery**: Focus on specific relationship types rather than all possible connections

## Scalability Challenges
1. **API Costs**: Each LLM call costs money, scales with N² comparisons
2. **Time**: Sequential processing vs. parallel ingestion
3. **Prompt Engineering**: Requires careful tuning for consistent outputs
4. **Rate Limits**: Need to handle API throttling gracefully
