# Knowledge Graph Strategy

## Goal
Transform pre-chunked CSV content into knowledge graphs with contextualized retrieval using Zep's graph capabilities.

## Content Structure
Your CSV contains:
- **Pre-chunked content**: Already split by learning objectives
- **Rich metadata**: lo_id, type, book, learning_objective, unit, chapter
- **Structured content**: JSON-formatted problems, solutions, assessments
- **Session types**: concept, exercise, example, try-it

## The Approach: Contextualized Retrieval

**Method**: Build KG from CSV, then implement smart retrieval with context using Zep's hybrid search capabilities

```python
# CSV provides base structure
# Use Zep's hybrid search (semantic + BM25 + BFS)
# Implement contextual search across LOs and content
# Add prerequisite relationships and learning paths
```

**Pros**: Best user experience, intelligent recommendations, leverages Zep's built-in capabilities
**Cons**: Requires understanding of Zep's graph search API

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

## Implementation Plan

### Week 2: Test CSV Integration

**Day 1-2**: Build CSV to KG transformer
**Day 3-4**: Test with small sample (first 10 rows)
**Day 5-7**: Compare with existing chunking approaches

### Testing Process
1. Transform CSV rows to KG nodes/edges
2. Test entity extraction from JSON content
3. Send to Zep and build knowledge graph
4. Measure: entity quality, relationship accuracy, completeness

### Success Criteria
- Learning objectives properly extracted and linked
- Content items (problems, solutions) connected to LOs
- Prerequisites and relationships identified
- Contextual retrieval working (find related content based on current LO)
- Best approach identified for full dataset processing
