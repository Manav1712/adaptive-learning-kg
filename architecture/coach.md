# Coach Agent - Knowledge Graph Design & Implementation

## Overview

The Coach agent serves as the "front door" for the adaptive learning system, working with a lightweight knowledge graph containing only learning objectives (LOs) and prerequisite relationships. It decides session flow (continue/branch/switch) and hands off to the Retriever agent for content delivery.

## Architecture Insights from Manager Meeting

### Two Separate Graph Views
- **Coach Graph**: Lightweight, 600 learning objectives + prerequisite edges
- **Retriever Graph**: Content-rich, includes actual learning materials (current experiments)

### Fixed Nodes, Discovered Edges
- **Nodes**: 600 predefined learning objectives (not extracted from content)
- **Edges**: Prerequisites discovered through content similarity analysis
- **Coach doesn't need to "map" to LOs** - they're already defined!

## Coach Agent Responsibilities

### Primary Function
Maps student input → Learning Objective + decides session flow

### Key Decisions
1. **continue** - stay in current LO/topic  
2. **branch** - explore related/prerequisite LO
3. **switch** - completely new topic/session

### Input Processing
- Student says "What's vertex form?" 
- Coach searches **Coach Graph** (600 LOs) to find matching LO
- No need for complex entity extraction - just LO lookup

## Knowledge Graph Implementation Strategy

### Recommendation: Hybrid Approach
- **Use Zep to STORE** the Coach graph (group graph) and per-user temporal overlay
- **Generate prerequisite edges OFFLINE** (deterministically) and ingest them into Zep
- **Do NOT rely on auto-extraction** for Coach edges; it introduces noise you don't want in a control graph

### Why Zep is Still Useful
- Per-user temporal overlay is built-in (user graphs): perfect for Coach's personalization layer
- Unified search + BFS over the LO graph and overlay
- One place to serve Coach and later hand off to Retriever

### Why Not Auto-Generate Edges in Zep for Coach
- Experiments show edge-type noise and low constraint effectiveness when using text-driven extraction
- For Coach, edges must be stable, interpretable, and directional
- Deterministic generation > LLM extraction

## Implementation Approach

### 1. Fixed LO Nodes
- You already have the 600 LOs
- Prepare metadata: `lo_id`, `title`, `description`, `unit`, `chapter`
- Ingest as "json" or "text" episodes that create one entity per LO
- Single label: `LearningObjective`

### 2. Deterministic Prerequisite Edges
- Compute candidates with text embeddings over LO descriptions (KNN)
- Use lightweight LLM check to decide direction A → B iff "A is a prerequisite for B"
- Optionally use unit/chapter ordering as a prior
- Export edges as: `source_lo_id`, `target_lo_id`, `type=PREREQUISITE_OF`, `confidence`

### 3. Ingest Edges into Zep
- Add as structured "json" episodes expressing explicit facts (PREREQUISITE_OF) between existing LO nodes
- Enforce minimal ontology: one entity type (LearningObjective) and one edge type (PREREQUISITE_OF)
- Optionally set fact rating instruction to downrank anything not matching PREREQUISITE_OF

### 4. Temporal Overlay (Per User)
- As students interact, write overlay edges (MASTERED_BY, STRUGGLES_WITH, RECENTLY_STUDIED) to user graphs
- Keep Coach logic simple: continue/branch/switch using base LO graph + overlay signals

## Coach Graph Querying Examples

### Finding Matching LO
```python
# Coach searches lightweight graph
coach_results = client.graph.search(
    query="vertex form",
    scope="nodes",  # Just LO nodes
    graph_id="coach_graph"  # Lightweight version
)
```

### Finding Prerequisites
```python
# Find prerequisites via edges
prereqs = client.graph.search(
    query="prerequisites",
    scope="edges", 
    search_filters={"edge_types": ["PREREQUISITE_OF"]},
    bfs_origin_node_uuids=[current_lo_uuid]
)
```

## Coach → Retriever Handoff

### Process
1. Coach decides LO + session type (concept/exercise/example)
2. Generates JSON for Retriever: `{"lo_id": "lo_vertex_form", "content_type": "concept"}`
3. Retriever queries **Retriever Graph** for actual content

### JSON Structure
```json
{
  "action": "continue | branch | switch",
  "message_to_student": "Great—vertex form. We'll continue here.",
  "target_los": ["lo_vertex_form"],
  "center_node": "lo_vertex_form",
  "new_session_id": "sess_2"
}
```

## Fallback Strategy

### If Zep Ingestion Becomes a Bottleneck
- Materialize the Coach LO graph offline (e.g., NetworkX/Neo4j)
- Mirror it into Zep periodically for serving and personalization
- Benefit: still use Zep for overlay + retrieval integration, while keeping edge generation under full control

## Multimodal Considerations (Future)

### Implementation Order
1. Start with text-only LO descriptions for embeddings
2. Later compare image transcripts vs direct image embeddings for LO similarity
3. Plug the better signal into the same offline edge pipeline

### Vision Language Models
- Investigate VisionLM for multimodal (text + image) embeddings
- Compare performance of image transcripts vs direct image embeddings
- Consider cost-benefit analysis of multimodal approach

## Quality Assurance

### Sanity Checks Before Use
- Detect and fix cycles in PREREQUISITE_OF
- Cap out-degree/in-degree to avoid hub explosions
- Spot-check a sample with LLM rubric ("Is A truly a prerequisite for B?")

### Edge Validation
- Ensure all edges are directional and meaningful
- Validate prerequisite relationships against curriculum standards
- Check for orphaned nodes or disconnected components

## Next Steps

### Immediate Actions
1. **Research best underlying embedding system** for knowledge graph
2. **Create learning objective knowledge graph** in Zep (offline edge generation)
3. **Run small experiment** comparing transcript-based vs image-based embeddings
4. **Provide environment, tools, and prompt structure** for Coach implementation

### Technical Investigation
- Compare Zep vs custom solution vs GraphRAG for graph generation
- Explore agentic workflows for university tech implementations
- Investigate vision language models for multimodal embeddings

## Conclusion

The Coach agent requires a **precision-focused, deterministic approach** to edge generation rather than the exploratory, content-rich approach used for the Retriever. By generating edges offline and using Zep for storage and personalization, we get the best of both worlds: control over the learning objective structure and powerful temporal reasoning capabilities for student personalization.

**Bottom line**: Use Zep as the serving and personalization layer; generate the Coach edges offline and ingest them explicitly. This gives precision and stability now, while keeping the door open for multimodal upgrades and per-user temporal reasoning.
