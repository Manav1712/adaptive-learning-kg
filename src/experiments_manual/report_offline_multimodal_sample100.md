# Offline LLM Scoring (Multimodal) — Sample-100 Results

## Overview
- Goal: Generate a high-quality Coach Graph offline by linking Content → LO and LO → LO (prerequisites) using an LLM with multimodal inputs.
- Pipeline: prepare → candidate generation → LLM scoring → evaluation.
- This report summarizes the most recent sample run (100 items target for each task) with multimodal prompts enabled.

## Data Inputs
- LO index: `data/processed/lo_index.csv` (≈138 LOs)
- Content items: `data/processed/content_items.csv` (≈270 items)

## Configuration (key)
- Model: `gpt-4o-mini`
- Modality: `multimodal` (text + image URLs)
- Temperature: `0.0`
- Score threshold: `0.6`
- Max targets per call: `8`
- Candidate pruning:
  - Content→LO: restrict_same_unit=True, restrict_same_chapter=False, lexical_top_k=5, lexical_min_overlap=1
  - LO→LO: restrict_same_unit=True, restrict_same_chapter=False, lexical_top_k=10, lexical_min_overlap=1

## Content → LO Links (100-item sample)
- Script: `src/experiments_manual/discover_content_links.py`
- Mode: candidates + LLM scoring
- Result file: `data/processed/edges_content.csv`

Summary
- Total edges: 278
- Unique sources (LOs): 100
- Unique targets (content): 98
- Coverage: 98/270 content items ≈ 36.3%
- Score stats: min=1.00, p25=1.00, p50=1.00, p75=1.00, max=1.00, mean=1.00
- Relation mix: explained_by=278
- Modality: multimodal=278
- Integrity: missing source LOs=0, missing content=0

Observations
- Scores saturated at 1.00 across kept edges (with threshold 0.6). Rationale text is coherent and aligned with objectives. Multimodal pathways are exercised (image_url blocks present when available).

## LO → LO Prerequisites (100-target sample)
- Script: `src/experiments_manual/discover_prereqs.py`
- Mode: candidates + LLM scoring
- Result file: `data/processed/edges_prereqs.csv`

Summary
- Total edges: 299
- Unique sources (LOs): 87
- Unique targets (LOs): 94
- Incoming coverage: 94/138 LOs ≈ 68.1%
- Score stats: min=0.60, p25=0.70, p50=0.80, p75=0.90, max=1.00, mean=0.816
- Relation mix: prerequisite=299
- Modality: multimodal=299
- Integrity: missing source LOs=0, missing target LOs=0

Observations
- Healthy score distribution centered high (mean ≈ 0.816) with explanatory rationales. Multimodal prompts include LO-level images when available.

## Runtime (approx.)
- Content→LO (100 items): ~16–17 minutes end-to-end on laptop (single process).
- LO→LO (100 targets): ~7–8 minutes end-to-end on laptop (single process).

## Evaluation Pipeline
- Script: `src/experiments_manual/evaluate_outputs.py`
- What it checks:
  - Edge counts, unique sources/targets
  - Score statistics and relations
  - Referencing integrity (IDs exist in inputs)
  - Coverage: content with ≥1 link; LOs with ≥1 incoming prereq
  - Top-N edges with rationales (for spot inspection)

Example usage
```
python3 src/experiments_manual/evaluate_outputs.py --edges data/processed/edges_content.csv --top-n 5
python3 src/experiments_manual/evaluate_outputs.py --edges data/processed/edges_prereqs.csv --top-n 5
```

## Scaling Readiness
- Parallel sharding built into both scripts (CRC32 by target):
  - Content→LO: `--num-shards N --shard-index i` (shards content_id)
  - LO→LO: `--num-shards N --shard-index i` (shards target_lo_id)
  - Output/progress files auto-suffixed per shard (disable with `--no-suffix-outputs`).
- Multimodal prompts wired for both tasks; candidate pruning reduces LLM call volume.

## Key Findings
- The pipeline is stable and produces high-quality, well-rationalized edges for both link types.
- Multimodal support is functioning (image blocks sent when present).
- Coverage from a 100-item sample is already meaningful (≈36% of content; ≈68% of LOs with incoming prereqs).

## Risks / Limitations
- Cost/time increases linearly with number of candidates; content scores saturating at 1.00 suggests threshold tuning or prompt tightening may be useful for higher precision.
- Full dataset processing (270 content items, 138 LOs in this prepared view) is tractable; much larger corpora require parallel runs and budget planning.

## Next Steps
1. Run multi-shard, full-dataset scoring (remove `--limit`) and concatenate shard outputs.
2. Tighten prompts or raise threshold if precision needs to increase (esp. for content links saturating at 1.00).
3. Add lightweight caching keyed by (model+prompt) to avoid re-scoring on reruns.
4. Optional: build quick visualization from `edges_content.csv` and `edges_prereqs.csv`.

## Repro Commands
```
# Prepare (refresh inputs)
python3 src/experiments_manual/prepare_lo_view.py

# Content → LO, sample or full (multimodal)
python3 src/experiments_manual/discover_content_links.py --config src/experiments_manual/config.yaml --mode both --limit 100

# LO → LO prerequisites, sample or full (multimodal)
python3 src/experiments_manual/discover_prereqs.py --config src/experiments_manual/config.yaml --mode both --limit 100

# Evaluate
python3 src/experiments_manual/evaluate_outputs.py --edges data/processed/edges_content.csv --top-n 5
python3 src/experiments_manual/evaluate_outputs.py --edges data/processed/edges_prereqs.csv --top-n 5
```

---
Prepared by: experiments_manual
Date: autogenerated
