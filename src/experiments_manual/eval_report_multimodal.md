## Adaptive Learning KG – Evaluation Report (Multimodal Pipeline)

This report summarizes the latest full-graph evaluation for the multimodal manual pipeline after recent threshold and scoring updates.

### Scope and Inputs
- **Pipeline**: `prepare_lo_view.py`, `discover_content_links.py`, `discover_prereqs.py`, `evaluate_outputs.py`, `build_and_visualize.py`
- **Model**: `gpt-4o-mini`
- **Modality**: multimodal (text + image URLs)
- **Data artifacts**:
  - `data/processed/lo_index.csv` (138 LOs)
  - `data/processed/content_items.csv` (270 content items)
  - `data/processed/edges_content.csv` (823 content→LO edges)
  - `data/processed/edges_prereqs.csv` (290 LO→LO prerequisite edges)

### Methodology (Brief)
- **Preparation**: `prepare_lo_view.py` consolidates LOs and content, attaches curriculum metadata (book/unit/chapter), and generates chronological keys.
- **Candidate generation**:
  - Content→LO: constrained by curriculum groups for candidate selection.
  - LO→LO: prerequisites constrained to same unit/chapter; chronological guard prevents future→past edges.
- **LLM scoring**: batched candidates with text and image URLs; score ∈ [0,1], confidence ∈ [0,1]; retries with simple backoff.
- **Thresholding (discovery-time)**:
  - Prereqs: `score_threshold = 0.6`, `min_confidence = 0.0`.
  - Content links: `score_threshold = 0.6`.
- **Evaluation (reporting-time)**: metrics computed with an evaluation threshold of `0.6` for "kept" counts only; files retain full scores.

### Results – Content→LO Edges (823 total)
- **Score quality**: min 0.300, p25 0.700, p50 0.900, p75 0.900, max 1.000, mean 0.790
- **Coverage**: 98.1% content coverage (265/270 content items linked)
- **Relations**: `explained_by` 405, `exemplified_by` 228, `practiced_by` 190
- **Kept at eval ≥0.6**: 691
- **Integrity**: no missing source LOs or content
- **Curriculum**: intra-unit 0.000, intra-chapter 0.000 (links are cross-curriculum)
- **Parsimony**: duplicates 0; LO out-degree p95 = 15.0

### Results – LO→LO Prerequisites (290 total)
- **Score quality**: min 0.400, p25 0.800, p50 0.850, p75 0.900, max 0.950, mean 0.835
- **Incoming coverage**: 76.1% of LOs have at least one prerequisite (105/138)
- **Structure**: DAG = True, cycles = 0, longest path = 8
- **Curriculum alignment**: intra-unit 1.000, intra-chapter 1.000
- **Kept at eval ≥0.6**: 287
- **Parsimony**: duplicates 0, redundancy 0.610, out-degree p95 6.0, in-degree p95 6.0

### Interpretation
- **Strengths**
  - High scoring quality across both edge types with clear rationales
  - Strong content coverage (98%) and solid LO coverage (76% incoming)
  - Prerequisite graph is acyclic and well-structured (DAG, longest path 8)
  - Semantically rich content relations with balanced distribution

- **Areas to improve**
  - Prereq redundancy (0.61) suggests benefit from transitive reduction
  - Optional: tune curriculum preferences for content links if more locality is desired

### Recommendations
1. Apply transitive reduction on prerequisites to remove inferable 2-hop edges while preserving reachability.
2. Maintain chronological and curriculum guards to keep the prereq graph acyclic.
3. Optionally calibrate content-link prompting/filters if intra-unit alignment becomes a requirement.
4. Run targeted human spot-checks on highest-impact edges.


### Configuration Snapshot
- **Model**: `gpt-4o-mini`
- **Modality**: multimodal (text + images)
- **Discovery thresholds**:
  - Prereqs: score ≥ 0.5, confidence ≥ 0.0
  - Content links: score ≥ 0.3
- **Evaluation display threshold**: 0.6 (for kept-count reporting only)

### Summary
- Nodes: 408 (138 LOs + 270 content)
- Edges: 1,113 (823 content→LO + 290 LO→LO)
- The graph exhibits high coverage and quality, a clean DAG structure for prerequisites, and actionable next steps (transitive reduction and optional alignment tuning) to further improve parsimony and curriculum locality.

—
Report date: September 2025  
Pipeline: Phase 2 Manual (Multimodal)





1. eval notes - tell llm and check for use case
2. a - b, b - a cannot exist, c can't also go back to b
3. evaluation, g10 on what we did and why we think it works - asked llm, checked for cycles
4. PROMPT TUNING - EVALUATION - EVAL SUCCESSFUL - GRAPH SOMEWHAT REPRODUCABLE, COST ANALYSIS, PRESENT - READY TO SCALE
5. 