# Adaptive Learning KG – Evaluation Report (Multimodal Pipeline)

This report documents the evaluation of the manual experiments pipeline that discovers Content→LO links and LO→LO prerequisites using LLM scoring with multimodal prompts. It covers methodology, quantitative results, interpretation, and next steps.

## Scope and Inputs
- **Pipeline**: `src/experiments_manual/prepare_lo_view.py`, `discover_content_links.py`, `discover_prereqs.py`, `evaluate_outputs.py`, `build_and_visualize.py`
- **Model for scoring**: `gpt-4o-mini`
- **Modality**: `multimodal` (text + image URLs)
- **Data artifacts**:
  - `data/processed/lo_index.csv` (138 LOs)
  - `data/processed/content_items.csv` (270 content items)
  - `data/processed/edges_content.csv` (688 content→LO edges)
  - `data/processed/edges_prereqs.csv` (524 LO→LO prerequisite edges)

## Methodology
- **Data preparation**: `prepare_lo_view.py` normalizes LOs and content into structured CSVs with chronological ordering and stable IDs
- **Candidate generation**:
  - Content→LO: Unit/chapter-based filtering selects candidate LOs per content item
  - LO→LO: Candidate prerequisites constrained by curriculum grouping (unit/chapter)
- **LLM scoring (multimodal)**:
  - Prompts include target item and batched candidates with image URLs
  - Model: `gpt-4o-mini` with signed scoring `[-1, 1]`, confidence `[0, 1]`, and few-shot examples
  - Score threshold: 0.6 (only positive scores ≥0.6 are kept)
  - Sequential processing with progress logging
- **Outputs**:
  - Content links: `explained_by`, `exemplified_by`, `practiced_by` relations
  - Prerequisites: `prerequisite` relation with signed scores
- **Evaluation**: Referential integrity, coverage, curriculum consistency, structural metrics, and parsimony

## Current Results

### Content→LO Edges (688 total)
- **Score Quality**: 
  - Min: 0.600, Max: 1.000, Mean: 0.852
  - P25: 0.800, P50: 0.900, P75: 0.900
  - 100% kept (all scores ≥ 0.6 threshold)
- **Coverage**: 
  - 96.3% content coverage (260/270 content items linked)
  - 137 unique source LOs, 260 unique target content items
- **Relations Distribution**:
  - `explained_by`: 320 edges (46.5%)
  - `exemplified_by`: 206 edges (29.9%)
  - `practiced_by`: 162 edges (23.6%)
- **Integrity**: Perfect - no missing source LOs or content references
- **Curriculum**: 0% intra-unit, 0% intra-chapter (cross-curriculum linking)
- **Parsimony**: No duplicates, LO out-degree P95 = 13.0

**Top Quality Examples**:
- `1870 → 1870_example_2` (exemplified_by, score=1.0): "Content explicitly teaches symmetry in functions by determining if they are even, odd, or neither"
- `976 → 976_concept_1` (explained_by, score=1.0): "Content specifically focused on solving equations involving a single trigonometric function"
- `1870 → 1870_try_it_1` (practiced_by, score=1.0): "Content explicitly addresses symmetry of functions by evaluating whether they are even or odd"

### LO→LO Prerequisite Edges (524 total)
- **Score Quality**:
  - Min: 0.600, Max: 1.000, Mean: 0.789
  - P25: 0.700, P50: 0.800, P75: 0.900
  - 100% kept (all scores ≥ 0.6 threshold)
- **Coverage**: 
  - 99.3% incoming coverage (137/138 LOs have prerequisites)
  - 132 unique source LOs, 137 unique target LOs
- **Structure**: 
  - **⚠️ CRITICAL**: Not a DAG - contains 1000 cycles
  - Reciprocal pairs: High number of bidirectional dependencies
  - Longest path: Cannot be computed due to cycles
- **Curriculum**: 100% intra-unit, 100% intra-chapter (all edges within same curriculum groups)
- **Parsimony**: 
  - No duplicates
  - **High redundancy**: 96.6% (most edges inferable via 2-hop paths)
  - Out-degree P95 = 8.0, In-degree P95 = 8.0

**Top Quality Examples**:
- `1056 → 1053` (score=1.0): "Writing complex numbers in polar form is directly related to the target LO"
- `1041 → 1040` (score=1.0): "Solving applied problems using Law of Cosines builds upon understanding for target LO"
- `1237 → 1235` (score=1.0): "Create a Function by Composition directly aligns with composing functions"

## Interpretation

### Strengths
- **High Coverage**: 96.3% content linking, 99.3% LO prerequisite coverage
- **Quality Scoring**: Mean scores of 0.852 (content) and 0.789 (prereqs) indicate good LLM confidence
- **Perfect Integrity**: No missing references or duplicates
- **Diverse Relations**: Content links properly categorized into explained/exemplified/practiced
- **Bounded Degrees**: Reasonable out-degree distribution (P95 = 13 max)

### Critical Issues
1. **Cyclic Prerequisites**: 1000 cycles violate prerequisite logic - students cannot have circular learning dependencies
2. **High Redundancy**: 96.6% of prerequisite edges are redundant (inferable via 2-hop paths)
3. **Cross-Curriculum Content Links**: 0% intra-unit/chapter suggests content linking ignores curriculum structure
4. **Reciprocal Dependencies**: Many bidirectional prerequisite relationships

### Moderate Concerns
- **Score Distribution**: Content scores cluster at high values (P75=0.9) - may indicate overconfident LLM outputs
- **Curriculum Alignment**: Content links don't respect unit/chapter boundaries

## Recommendations

### Immediate (High Priority)
1. **Break Prerequisite Cycles**:
   - Implement cycle detection and removal algorithm
   - Enforce DAG structure by removing minimal edge set
   - Consider temporal/curriculum ordering constraints

2. **Apply Transitive Reduction**:
   - Remove ~96% redundant prerequisite edges
   - Keep only direct dependencies, remove inferable 2-hop paths
   - Maintain learning progression semantics

3. **Fix Reciprocal Dependencies**:
   - Remove bidirectional prerequisite relationships
   - Enforce unidirectional learning progression

### Medium Priority
4. **Improve Content-Curriculum Alignment**:
   - Add intra-unit/chapter preference in candidate generation
   - Modify prompts to encourage curriculum-aware linking
   - Verify unit/chapter metadata accuracy

5. **Score Calibration**:
   - Introduce contrastive negatives in prompts
   - Add confidence penalty for over-linking
   - Consider raising threshold above 0.6 for stricter filtering

### Long-term
6. **Human Validation**:
   - Spot-check high-confidence edges for accuracy
   - Validate cycle-breaking decisions with domain experts
   - Create gold standard for evaluation metrics

## Technical Implementation Notes

### Current Pipeline Flow
```bash
# Data preparation
python src/experiments_manual/prepare_lo_view.py

# Edge discovery
python src/experiments_manual/discover_content_links.py --config config.yaml --mode both
python src/experiments_manual/discover_prereqs.py --config config.yaml --mode both

# Evaluation
python src/experiments_manual/evaluate_outputs.py --edges data/processed/edges_content.csv
python src/experiments_manual/evaluate_outputs.py --edges data/processed/edges_prereqs.csv

# Visualization
python src/experiments_manual/build_and_visualize.py --out data/processed/graph_preview.html
```

### Configuration
- Model: `gpt-4o-mini`
- Score threshold: 0.6
- Modality: multimodal (text + images)
- Signed scoring: `[-1, 1]` with confidence `[0, 1]`

## Summary Statistics
- **Total Edges**: 1,212 (688 content→LO + 524 LO→LO)
- **Nodes**: 138 LOs + 270 content items = 408 total
- **Edge Density**: ~2.97% (1,212 / (408 × 408))
- **Processing**: ~15 minutes total (content discovery + prerequisite discovery)
- **Cost**: ~$15-20 in GPT-4o-mini API calls

## Next Steps
1. Implement cycle-breaking algorithm for prerequisites
2. Apply transitive reduction to eliminate redundancy
3. Validate curriculum metadata and improve content-curriculum alignment
4. Run human spot-checks on high-confidence edges
5. Generate final clean knowledge graph for adaptive learning applications

---
*Report generated: December 2024*  
*Pipeline version: Phase 2 Manual Ingestion*  
*Model: gpt-4o-mini with multimodal prompts*