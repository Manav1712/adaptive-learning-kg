### Adaptive Learning KG – Evaluation Report (Multimodal Pipeline)

This report documents the evaluation of the manual experiments pipeline that discovers Content→LO links and LO→LO prerequisites using LLM scoring with multimodal prompts. It covers methodology, the provisional ground-truth approach using GPT‑5 as a judge, current quantitative results, interpretation, and next steps.

## Scope and Inputs
- **Pipeline**: `src/experiments_manual/prepare_lo_view.py`, `discover_content_links.py`, `discover_prereqs.py`, `evaluate_outputs.py`, `evaluate_graph_quality.py`, `build_and_visualize.py`
- **Model for scoring**: `gpt-4o-mini`
- **Modality**: `multimodal` (text + image URLs)
- **Data artifacts**:
  - `data/processed/lo_index.csv`, `data/processed/content_items.csv`
  - `data/processed/edges_content.csv`, `data/processed/edges_prereqs.csv`
  - `data/processed/graph_quality_report.json` (evaluated in this report)

## Methodology
- **Data preparation**: `prepare_lo_view.py` normalizes LOs and content into `lo_index.csv` and `content_items.csv`, extracting text fields and image URLs. Stable content IDs are generated.
- **Candidate generation**:
  - Content→LO: lexical/structural filters select candidate LOs per content item.
  - LO→LO: candidate prerequisites constrained by curriculum grouping (unit/chapter) and lexical overlap.
- **LLM scoring (multimodal)**:
  - Prompts include the target item and batched candidates. When `modality=multimodal`, image URLs for target and candidates are appended to the message blocks.
  - Model: `gpt-4o-mini` with low temperature and a JSON-only response schema. Heuristic token-overlap fallback is used upon failure.
  - Sharding: CRC32-based sharding over IDs enables parallel execution and resumability; progress is logged to JSONL.
- **Outputs**:
  - Content links conform to the relation type `explained_by` with `score ∈ [0,1]`.
  - Prerequisites conform to relation type `prerequisite` with `score ∈ [0,1]`.
- **Evaluation** (`evaluate_outputs.py`, `evaluate_graph_quality.py`):
  - Referential integrity and coverage checks.
  - Curriculum consistency: intra-unit and intra-chapter ratios.
  - Structural metrics (LO→LO): DAG check, cycles, longest path, reciprocal edges.
  - Parsimony: duplicate edges, redundancy (2-hop inferability), degree distribution (P95).

## Provisional Ground Truth via GPT‑5 (LLM-as-a-Judge)
To obtain an initial, scalable reference for accuracy without full human annotation, we use GPT‑5 to label a sampled subset of edges.

- **Goal**: Estimate precision and provide qualitative rationales to guide thresholding and post-processing before human labeling.
- **Sampling**:
  - Stratified by score deciles and relation type (Content→LO vs LO→LO), optionally by unit/chapter.
  - Typical batch: 50–200 edges per relation.
- **Prompt (judge)**:
  - System: "You are an expert math curriculum designer. Given a candidate edge, decide if it is valid with a confidence in [0,1]. Respond ONLY in JSON."
  - User: Includes target node, source node, their texts, any image URLs, and the proposed relation. Requests JSON `{ valid: boolean, confidence: number, rationale: string }`.
- **Aggregation**:
  - Per-edge: accept if `valid=true` and `confidence≥τ` (e.g., 0.7).
  - Batch: compute precision@τ, mean confidence, and surface rationales for failure analysis.
- **Calibration**:
  - Compare GPT‑5 judgments with a small human spot-check (via `spot_check_edges.py`) to assess alignment and bias.
- **Limitations**:
  - LLM judgments are not ground truth; they are provisional and may share model biases with the generator. Use only to triage and guide thresholds before human review.

Note: The metrics below are from the pipeline’s computed outputs; GPT‑5 judgments are intended as a provisional reference layer and are not included in the numeric aggregates unless explicitly stated.

## Current Results (from `graph_quality_report.json`)

### Content→LO (`explained_by`)
- **Edges**: 278 (all multimodal; run id: `gpt-4o-mini`)
- **Score stats**: min=1.00, max=1.00, mean=1.00, p25=1.00, p50=1.00, p75=1.00
- **Coverage**: 98/270 content items linked (36.3% coverage)
- **Uniqueness**: unique sources=100, unique targets=98
- **Curriculum consistency**: intra-unit ratio=0.0, intra-chapter ratio=0.0 (0/278 considered)
- **Parsimony**: duplicate edges=0; LO out-degree P95=5

Observations:
- Scores saturate at 1.0, indicating thresholding or prompt/response bias; calibration recommended.
- 0% intra-unit/chapter alignment suggests cross-unit linking dominates or metadata mapping is incomplete.

### LO→LO Prerequisites (`prerequisite`)
- **Edges**: 299 (all multimodal; run id: `gpt-4o-mini`)
- **Score stats**: min=0.60, max=1.00, mean≈0.816, p25=0.70, p50=0.80, p75=0.90
- **Coverage**: 94/138 LOs have at least one incoming edge (68.1%)
- **Structure**: DAG=false; cycles detected=1000; longest path length=null; reciprocal pairs=116
- **Curriculum consistency**: intra-unit=1.0, intra-chapter=1.0 (all 299 considered are intra-group)
- **Parsimony**: duplicate edges=0; redundancy≈0.910 (≈91% edges inferable via 2-hop paths); out-degree P95=7; in-degree P95=5

Observations:
- Graph is not a DAG; substantial cycles and 116 reciprocal pairs compromise learning progression semantics.
- High redundancy (~91%) indicates dense local connectivity; transitive reduction should substantially simplify the graph.

## Interpretation
- **Strengths**:
  - Solid coverage: ~36% of content linked; ~68% of LOs with incoming prerequisites.
  - Referential integrity is clean (no missing references; 0 duplicates in both edge sets).
  - Degree distribution appears bounded (P95 out-degree 7, in-degree 5), avoiding extreme hubs.
- **Issues**:
  - Content link scores saturate at 1.0; likely overconfident outputs and/or insufficiently discriminative prompts.
  - Curriculum consistency mismatch: Content links are 0% intra-unit/chapter; verify unit/chapter mapping and candidate constraints.
  - Prereq graph is cyclic with high redundancy; violates expected DAG structure for prerequisites.

## Recommendations
1. **Post-process structure**
   - Break cycles and remove reciprocal edges; enforce DAG with minimal cuts.
   - Apply transitive reduction to eliminate ~90% redundant edges.
2. **Prompt and scoring calibration**
   - Tighten JSON schema, require calibrated confidence bands, and penalize over-linking.
   - Introduce contrastive negatives in batches to reduce score saturation.
   - Revisit thresholds (e.g., raise score threshold above 0.6 for prereqs).
3. **Curriculum alignment**
   - Validate unit/chapter metadata joins; if correct, explicitly encourage intra-unit links in prompts/candidates.
4. **Provisional ground truth with GPT‑5**
   - Run GPT‑5 judge on stratified samples and compute precision@τ; compare with human spot-checks from `spot_check_edges.py` to calibrate.
5. **Human-in-the-loop**
   - Label a small gold set (50–100 edges per relation) to validate GPT‑5 judgments and finalize thresholds.

## Reproducibility
- Configure `src/experiments_manual/config.yaml` with `modality: multimodal` and model `gpt-4o-mini`.
- Run preparation, discovery (with sharding as needed), evaluation, and visualization. Example flow:
  - Prepare: `python src/experiments_manual/prepare_lo_view.py`
  - Discover content links: `python src/experiments_manual/discover_content_links.py --num-shards N --shard-index i`
  - Discover prerequisites: `python src/experiments_manual/discover_prereqs.py --num-shards N --shard-index i`
  - Evaluate single files: `python src/experiments_manual/evaluate_outputs.py --edges data/processed/edges_content.csv`
  - Aggregate report: `python src/experiments_manual/evaluate_graph_quality.py --out data/processed/graph_quality_report.json`
  - Visualize: `python src/experiments_manual/build_and_visualize.py`

## Appendix: Key Metrics Snapshot
- Content→LO: edges=278; coverage=36.3%; intra-unit=0.0; duplicates=0; LO out-degree P95=5; all scores=1.0
- LO→LO: edges=299; coverage=68.1%; DAG=false; cycles=1000; reciprocal pairs=116; redundancy≈91%; out-degree P95=7; in-degree P95=5


