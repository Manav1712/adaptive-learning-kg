# Adaptive Learning Knowledge Graph Platform

A local-first adaptive learning system that constructs high-quality knowledge graphs from OpenStax content using LLM-driven edge discovery, provides comprehensive evaluation pipelines, and offers interactive D3.js visualizations for exploring learning relationships.

## Project Overview

This platform creates an adaptive learning knowledge graph system that:
- **Constructs Knowledge Graphs**: Extracts Learning Objectives (LOs) and Content Items from OpenStax materials
- **Discovers Relationships**: Uses multimodal LLM prompts to identify prerequisites and content relationships
- **Validates Quality**: Implements two-tier evaluation (structural + semantic) for high-confidence edges
- **Visualizes Interactively**: Provides D3.js-based graph visualization with legend and interaction controls
- **Supports Scalability**: Includes sharding, batching, and retry mechanisms for large-scale processing
- Uses local-first architecture for privacy and portability

## Architecture

See the [architecture/](architecture/) folder for detailed design documents:
- [Phases](architecture/phases.md) - Development roadmap
- [Implementation Plan](architecture/implementation.md) - Technical implementation details
- [Sequence Diagrams](architecture/sequence_diagrams.md) - System interactions


#### Knowledge Graph Construction Pipeline

```bash
# 1. Prepare processed inputs (LO index + content items)
python3 src/experiments_manual/prepare_lo_view.py

# 2. Discover Content → LO relationships
python3 src/experiments_manual/discover_content_links.py \
  --config src/experiments_manual/config.yaml --mode both

# 3. Discover LO → LO prerequisite relationships  
python3 src/experiments_manual/discover_prereqs.py \
  --config src/experiments_manual/config.yaml --mode both

# 4. Structural validation (integrity, cycles, coverage)
python3 src/experiments_manual/evaluate_outputs.py --edges data/processed/edges_content.csv
python3 src/experiments_manual/evaluate_outputs.py --edges data/processed/edges_prereqs.csv

# 5. Semantic validation (LLM-based edge correctness)
python3 src/experiments_manual/evaluate_with_llm.py \
  --edges-in data/processed/edges_prereqs.csv \
  --jsonl-out data/processed/llm_edge_checks.jsonl \
  --summary-out data/processed/llm_edge_checks_summary.json

python3 src/experiments_manual/evaluate_with_llm.py \
  --edges-in data/processed/edges_content.csv \
  --jsonl-out data/processed/llm_edge_checks_content_final.jsonl \
  --summary-out data/processed/llm_edge_checks_content_final_summary.json

# 6. Build interactive HTML visualization
python3 src/experiments_manual/build_and_visualize.py --out data/processed/graph_preview.html
```

#### Key Features

- **Multimodal Discovery**: LLM prompts include both text content and image URLs when available
- **Strict Validation**: JSON-only responses enforced with robust parsing and retry logic
- **Scalable Processing**: Sharding support for large-scale edge discovery with deterministic work splitting
- **Comprehensive Evaluation**: Two-tier validation (structural + semantic) ensures high-quality relationships
- **Production Ready**: Heuristic fallbacks and error handling for reliable operation

#### Next Steps: RAG System Development

The manually constructed knowledge graph serves as the foundation for the RAG system:

1. **Retriever Agent**: Uses the knowledge graph to find relevant content and learning paths
2. **Coach Generator**: Creates personalized learning strategies based on student progress and graph relationships  
3. **Tutor Agent**: Provides interactive tutoring sessions with context from Zep's temporal layer
4. **Integration**: All agents leverage the validated knowledge graph for accurate, contextual responses

## Quick Start

### 1. Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_key_here
```

### 2. Run Complete Pipeline
```bash
# Discover content→LO relationships
python src/experiments_manual/discover_content_links.py \
  --config src/experiments_manual/config.yaml --mode both

# Discover LO→LO prerequisite relationships  
python src/experiments_manual/discover_prereqs.py \
  --config src/experiments_manual/config.yaml --mode both

# View interactive visualization (start server)
python -m http.server 8000
# Then open: http://localhost:8000/vis/graph_preview.html
```

### 3. View Interactive Visualization
```bash
# Start local server
python -m http.server 8000

# Open in browser: http://localhost:8000/vis/graph_preview.html
# Or use the processed version: http://localhost:8000/data/processed/graph_preview.html
```

## Core Pipeline Components

### Data Flow
```
OpenStax Content → LO Index + Content Items → Edge Discovery → Evaluation → Visualization
```

### Key Files and Directories

#### Configuration
- **`src/experiments_manual/config.yaml`**: Main configuration file
  - Model: `gpt-4o-mini` with multimodal support
  - Scoring threshold: `0.85` for high-quality edges
  - Batch size: `8` targets per LLM call
  - Retry logic: `3` attempts with exponential backoff

#### Discovery Scripts
- **`src/experiments_manual/discover_content_links.py`**: Finds LO↔Content relationships
  - Supports `explained_by`, `exemplified_by`, `practiced_by` relations
  - Multimodal prompts include text + image URLs
  - Configurable pruning (same unit/chapter filtering)
  
- **`src/experiments_manual/discover_prereqs.py`**: Finds LO→LO prerequisite relationships
  - Cross-unit/chapter prerequisite discovery
  - Aggregates content per LO for comprehensive context
  - Temporal ordering based on curriculum structure

#### Evaluation Pipeline
- **`src/experiments_manual/evaluate_heuristic.py`**: Structural analysis
  - Referential integrity checks
  - Cycle detection in prerequisite graphs
  - Coverage metrics and graph statistics
  
- **`src/experiments_manual/evaluate_llm.py`**: Semantic validation
  - LLM-based edge correctness assessment
  - JSON-only responses with retry logic
  - Binary labels: `correct`/`incorrect` with reasoning

#### Visualization
- **`vis/graph_preview.html`**: D3.js-based interactive visualization
  - Color-coded nodes by type (LO, Concept, Example, Try It)
  - Edge styling by relationship type (solid/dashed)
  - Interactive legend with node/edge explanations
  - Hover tooltips and click interactions
  - Reads CSV data dynamically from same directory

### Data Files

#### Input Data
- **`data/processed/lo_index.csv`**: Learning Objectives metadata
  - Columns: `lo_id`, `learning_objective`, `unit`, `chapter`, `book`
- **`data/processed/content_items.csv`**: Content Items metadata  
  - Columns: `content_id`, `content_type`, `lo_id_parent`, `text`, `image_urls`

#### Generated Edges
- **`data/processed/edges_content.csv`**: LO→Content relationships
  - Columns: `source_lo_id`, `target_content_id`, `relation`, `score`, `confidence`, `rationale`
- **`data/processed/edges_prereqs.csv`**: LO→LO prerequisite relationships
  - Columns: `source_lo_id`, `target_lo_id`, `relation`, `score`, `confidence`, `rationale`

#### Evaluation Outputs
- **`data/processed/llm_edge_checks_*.jsonl`**: Per-edge LLM evaluations
- **`data/processed/llm_*_summary.json`**: Aggregate evaluation metrics
- **`data/processed/graph_quality_report.json`**: Structural analysis results



### Evaluation and Quality Control
```bash
# Structural evaluation
python src/experiments_manual/evaluate_heuristic.py \
  --edges data/processed/edges_content.csv \
  --json-out data/processed/content_eval.json

# Semantic evaluation with LLM
python src/experiments_manual/evaluate_llm.py \
  --edges-in data/processed/edges_prereqs.csv \
  --jsonl-out data/processed/llm_prereq_eval.jsonl \
  --summary-out data/processed/llm_prereq_summary.json
```

## Visualization Features

The interactive D3.js visualization (`vis/graph_preview.html`) provides:

### Node Types
- **Learning Objectives**: Blue circles
- **Concepts**: Green circles  
- **Examples**: Yellow/orange circles
- **Try It Activities**: Teal circles

### Edge Types
- **Explained By**: Solid gray lines (LO↔Content)
- **Prerequisites**: Dashed red lines (LO→LO)
- **Thickness**: Represents confidence scores

### Interactions
- **Hover**: View detailed node/edge information
- **Click edges**: See relationship details in modal
- **Drag nodes**: Reposition for better layout
- **Zoom/Pan**: Navigate large graphs
- **Legend**: Always-visible explanation panel

### Customization
The visualization automatically:
- Sizes nodes by connection degree
- Colors edges by relationship type  
- Provides responsive layout with force simulation
- Includes collision detection and smooth animations

## Configuration Reference

### `config.yaml` Structure
```yaml
model: gpt-4o-mini           # LLM model for edge discovery
modality: multimodal         # Include images in prompts
scoring:
  threshold: 0.85            # Minimum score for edge inclusion
runtime:
  max_targets_per_call: 8    # Batch size for LLM calls
  max_retries: 3             # Retry attempts on failure
pruning:
  same_unit: false           # Allow cross-unit relationships
  same_chapter: false        # Allow cross-chapter relationships
llm:
  temperature: 0.0           # Deterministic responses
  score_threshold: 0.7       # Alternative threshold for prereqs
  min_confidence: 0.6        # Minimum confidence for inclusion
```

## Development Setup

### Prerequisites
- Python 3.9+
- OpenAI API key
- Virtual environment recommended

### Installation
```bash
git clone <repository>
cd adaptive-learning-kg
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables
```bash
export OPENAI_API_KEY=your_key_here
# Optional for Zep integration:
export ZEP_API_KEY=your_zep_key_here
```

### Running Tests
```bash
# Run evaluation on sample data
python src/experiments_manual/evaluate_heuristic.py \
  --edges demo/edges_content.csv

# Test visualization (copy demo data to vis folder)
cp demo/*.csv vis/
python -m http.server 8000
# Open: http://localhost:8000/vis/graph_preview.html
```

## Project Structure
```
adaptive-learning-kg/
├── src/experiments_manual/     # Core pipeline scripts
│   ├── config.yaml            # Configuration file
│   ├── discover_content_links.py  # LO↔Content discovery
│   ├── discover_prereqs.py    # LO→LO prerequisite discovery  
│   ├── evaluate_heuristic.py  # Structural evaluation
│   └── evaluate_llm.py        # Semantic evaluation
├── data/processed/            # Generated data files
│   ├── lo_index.csv          # Learning objectives
│   ├── content_items.csv     # Content metadata
│   ├── edges_content.csv     # LO↔Content relationships
│   ├── edges_prereqs.csv     # LO→LO prerequisites
│   └── graph_preview.html    # Generated visualization (legacy)
├── vis/                      # Self-contained visualization
│   ├── graph_preview.html    # D3.js interactive graph
│   └── *.csv                # Required data files
├── demo/                     # Sample data for testing
├── architecture/             # Design documentation
└── requirements.txt          # Python dependencies
```
