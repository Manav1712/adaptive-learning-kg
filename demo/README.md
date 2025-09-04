# Adaptive Learning Knowledge Graph Demo

## Project Overview

This demo showcases an adaptive learning platform built on a knowledge graph constructed from OpenStax Calculus content. The system uses AI-powered edge discovery to create relationships between learning objectives and content, enabling personalized tutoring experiences.

## Core Concept

- **Knowledge Graph**: 138 learning objectives, 270 content items, 577 total edges
- **Two Edge Types**:
  - Content → Learning Objective links (278 edges) - "explained_by" relationships
  - Learning Objective → Learning Objective prerequisites (299 edges) - "prerequisite" relationships
- **AI-Powered**: Uses GPT-4o-mini with multimodal prompts for edge discovery
- **Local-First**: All data stored locally, no cloud dependencies

## Essential Data Files

To build the knowledge graph, you need these **4 core CSV files**:

### 1. Learning Objectives Index (`lo_index.csv`)
- **Purpose**: Defines all learning objectives (nodes)
- **Columns**: `lo_id`, `learning_objective`, `unit`, `chapter`, `book`
- **Example**: 138 LOs from Precalculus 2e covering trigonometric functions

### 2. Content Items (`content_items.csv`)
- **Purpose**: Defines all content pieces (concept, example, exercise nodes)
- **Columns**: `content_id`, `content_type`, `lo_id_parent`, `text`, `image_urls`, `book`, `learning_objective`, `unit`, `chapter`
- **Example**: 270 content items including concepts, examples, and exercises

### 3. Content Edges (`edges_content.csv`)
- **Purpose**: Links content items to learning objectives
- **Columns**: `source_lo_id`, `target_content_id`, `relation`, `score`, `rationale`, `modality`, `run_id`
- **Example**: 278 "explained_by" relationships with confidence scores

### 4. Prerequisite Edges (`edges_prereqs.csv`)
- **Purpose**: Links learning objectives in prerequisite chains
- **Columns**: `source_lo_id`, `target_lo_id`, `relation`, `score`, `rationale`, `modality`, `run_id`
- **Example**: 299 "prerequisite" relationships showing learning dependencies

## Optional Data Files

### 5. Graph Quality Report (`graph_quality_report.json`)
- **Purpose**: Analytics and quality metrics
- **Contains**: Coverage statistics, integrity checks, parsimony analysis
- **Usage**: Dashboard analytics, quality assurance

### 6. Progress Logs (`progress_*.jsonl`)
- **Purpose**: Track processing progress during edge discovery
- **Usage**: Debugging, monitoring batch processing

## PyVis Visualization Setup

### Dependencies
```python
# Core visualization
pyvis>=0.3.2
networkx>=3.0
pandas>=2.0.0

# Data processing
numpy>=1.24.0
```

### Key Features
- **Interactive Network**: Zoom, pan, click interactions
- **Node Highlighting**: Click to highlight connected nodes
- **Color Coding**: Different colors for LOs vs content
- **Responsive Layout**: Automatic positioning with physics simulation
- **Search/Filter**: Built-in search functionality

### Visualization Code Structure
```python
# Main visualization script: src/experiments_manual/build_and_visualize.py
# Input: 4 CSV files above
# Output: Interactive HTML file with vis.js network

# Key functions:
- load_csv(): Load data from CSV files
- filter_by_scope(): Filter by unit/chapter
- build_network(): Create PyVis network
- add_nodes(): Add LO and content nodes
- add_edges(): Add prerequisite and content edges
- generate_html(): Export interactive visualization
```

## Data Schema

### Learning Objectives
- **ID**: Unique identifier (e.g., "957")
- **Title**: Human-readable name (e.g., "Solving Trigonometric Equations with Identities")
- **Unit**: Course unit (e.g., "Solving Trigonometric Equations with Identities")
- **Chapter**: Course chapter (e.g., "Trigonometric Identities and Equations")
- **Book**: Source material (e.g., "Precalculus 2e")

### Content Items
- **ID**: Unique identifier (e.g., "957_concept_1")
- **Type**: Content type ("concept", "example", "exercise")
- **Parent LO**: Associated learning objective
- **Text**: Full content text
- **Image URLs**: Associated images (if any)
- **Metadata**: Unit, chapter, book information

### Edges
- **Source/Target**: Node IDs
- **Relation**: Relationship type ("explained_by", "prerequisite")
- **Score**: Confidence score (0.0-1.0)
- **Rationale**: AI explanation for the relationship
- **Modality**: Input type ("multimodal")
- **Run ID**: Processing run identifier

## Usage Instructions

### 1. Prepare Data
```bash
# Ensure you have the 4 essential CSV files in data/processed/
ls data/processed/lo_index.csv
ls data/processed/content_items.csv
ls data/processed/edges_content.csv
ls data/processed/edges_prereqs.csv
```

### 2. Generate Visualization
```bash
# Basic visualization
python src/experiments_manual/build_and_visualize.py

# Filtered by unit
python src/experiments_manual/build_and_visualize.py --focus-unit "Solving Trigonometric Equations with Identities"

# Limited nodes for performance
python src/experiments_manual/build_and_visualize.py --max-nodes 50
```

### 3. View Results
- Open `data/processed/graph_preview.html` in browser
- Interactive network visualization with all features

## Architecture Notes

### Current Implementation
- **Backend**: Python scripts for data processing
- **Frontend**: PyVis-generated HTML with vis.js
- **Storage**: Local CSV files
- **Processing**: Batch edge discovery with GPT-4o-mini

### Future Enhancements
- **API Layer**: FastAPI endpoints for real-time data
- **Modern Frontend**: React/Vue.js with enhanced UI
- **Real-time Updates**: WebSocket connections
- **User Management**: Student progress tracking
- **AI Integration**: Live tutoring simulation

## File Structure for Demo

```
demo/
├── README.md                    # This file
├── data/                        # Essential data files
│   ├── lo_index.csv            # Learning objectives
│   ├── content_items.csv       # Content items
│   ├── edges_content.csv       # Content→LO edges
│   └── edges_prereqs.csv       # LO→LO prerequisites
├── src/                        # Python code
│   └── build_and_visualize.py  # Visualization script
├── requirements.txt            # Dependencies
└── output/                     # Generated files
    └── graph_preview.html      # Interactive visualization
```

## Key Metrics

- **Total Nodes**: 408 (138 LOs + 270 content items)
- **Total Edges**: 577 (278 content + 299 prerequisite)
- **Coverage**: 36.3% content coverage, 68.1% LO incoming coverage
- **Quality**: High integrity (0 missing references)
- **Processing**: GPT-4o-mini with multimodal prompts
- **Confidence**: Average score 0.82 for prerequisites, 1.0 for content

## Next Steps for Demo

1. **Copy Essential Files**: Transfer the 4 CSV files to demo/data/
2. **Setup Environment**: Install pyvis, networkx, pandas
3. **Generate Visualization**: Run build_and_visualize.py
4. **Customize UI**: Modify HTML output for demo presentation
5. **Add Interactivity**: Enhance with search, filtering, progress tracking
6. **Deploy**: Host on GitHub Pages, Vercel, or similar

This knowledge graph represents a working prototype of an adaptive learning system that can be extended with real-time tutoring, progress tracking, and personalized learning paths.
