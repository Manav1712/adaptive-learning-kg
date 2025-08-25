# Knowledge Graph Strategy - Production Approach

## Goal
Transform pre-chunked CSV content into production-ready knowledge graphs with ontology-enforced relationships and contextualized retrieval using Zep's advanced graph capabilities.

## Content Structure
Your CSV contains:
- **Pre-chunked content**: Already split by learning objectives (270 episodes across 3 files)
- **Rich metadata**: lo_id, type, book, learning_objective, unit, chapter
- **Structured content**: JSON-formatted problems, solutions, assessments
- **Session types**: concept, exercise, example, try-it

## Current Approach: Ontology-Enforced Knowledge Graphs

**Method**: Build constrained KGs using Zep's `set_ontology()` method with custom entity/edge types, then implement smart retrieval

```python
# Define custom ontology before ingestion
entities = {"Concept": ConceptModel, "Example": ExampleModel, "Exercise": ExerciseModel, "TryIt": TryItModel}
edges = {"PREREQUISITE_OF": ConstrainedEdge, "PART_OF": ConstrainedEdge, "ASSESSED_BY": ConstrainedEdge}
client.graph.set_ontology(entities=entities, edges=edges, graph_ids=[graph_id])

# Apply fact rating to filter noise
fact_rating_instruction = "Rate by calculus learning relevance..."
client.graph.update(graph_id=graph_id, fact_rating_instruction=instruction)
```

**Pros**: 
- Production-ready constraint enforcement
- Reduced edge-type noise (target: 3-5 types vs 200+ without ontology)
- Higher precision relationships (target: 80%+ effectiveness vs 24.6% with schema hints)
- Built-in fact filtering for quality control

**Cons**: 
- Complex ontology design required
- Longer processing times (2-6 hours for 270 episodes on FREE plan)
- Need iterative testing to validate constraint effectiveness

### Zep Implementation Strategy

**1. Graph Structure**
- **LearningObjective nodes**: Each `lo_id` becomes a node with metadata
- **ContentItem nodes**: Each CSV row becomes a node with problem/solution content
- **Relationship edges**: `ALIGNED_WITH` between ContentItem and LearningObjective
- **Prerequisite edges**: `PREREQUISITE_OF` between LearningObjectives

**2. Contextual Retrieval Methods**
- **Hybrid Search**: Combine semantic similarity + BM25 for finding related content
- **Breadth-First Search (BFS)**: Start from current LO to find contextually relevant content
- **Scope-based Search**: Use `edges` scope for specific facts, `nodes` scope for entity overviews

**3. Search Patterns**
```python
# Find related content for current LO
search_results = client.graph.search(
    query="functions domain range",
    scope="edges",
    bfs_origin_node_uuids=[current_lo_uuid]
)

# Find prerequisite LOs
prereq_results = client.graph.search(
    query="prerequisites for functions",
    scope="nodes",
    search_filters={"edge_types": ["PREREQUISITE_OF"]}
)
```

## Experimental Results & Current Status

### Baseline V1 (Initial Approach)
- **Results**: 474 nodes, 493 edges from raw CSV data
- **Issue**: High edge-type noise, generic entity extraction
- **Learning**: Need more constraint on entity/relationship types

### Baseline V2 (Schema Hints)
- **Results**: 270 episodes, 24.6% constraint effectiveness, 169 edge types  
- **Issue**: Schema hints insufficient for production constraints
- **Learning**: Text-based hints don't enforce server-side constraints

### Baseline V3 (Ontology Enforcement) - **IN PROGRESS**
- **Approach**: Custom entity types (Concept/Example/Exercise/TryIt) with property definitions
- **Constraints**: PREREQUISITE_OF/PART_OF/ASSESSED_BY edges with source-target restrictions
- **Enhancements**: Fact rating instruction, type balancing (250 max per type)
- **Status**: 270 episodes processing overnight (2-6 hour completion time)
- **Target**: 80%+ constraint effectiveness, <10 edge types for production readiness

### Next Steps
1. **Evaluate V3 Results**: Check constraint effectiveness and edge type reduction
2. **Production Decision**: 
   - ✅ If successful → Ready for Phase 2 (retrieval pipeline)
   - ⚠️ If unsuccessful → Investigate alternative constraint approaches
3. **Scale to Full Dataset**: Once approach validated with 270 episodes
