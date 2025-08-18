# 3-Tiered Chunking Strategy

## Goal
Test 3 chunking approaches from simple to advanced to find optimal knowledge graph generation.

## Content Structure
OpenStax Calculus has:
- **Sections**: 1.1, 1.2, 1.3, etc.
- **Mixed content**: Definitions, examples, problems, theorems
- **Hierarchical structure**: Chapter → Section → Content

## The 3 Tiers

### Tier 1: Simple Subheading Chunking
**Approach**: Split by section headers (1.1, 1.2, etc.) like Zep's docs

```python
chunks = text.split("## 1.1")  # Simple splitting
```

**Pros**: Fast, simple, mimics Zep's approach
**Cons**: May lose context

### Tier 2: Contextualized Chunking  
**Approach**: Add document context to each chunk

```python
chunk_with_context = f"Chapter 1: Functions and Graphs, Section 1.1\n\n{chunk}"
```

**Pros**: Better context for entity extraction
**Cons**: Slightly more complex

### Tier 3: Custom Entity Types
**Approach**: Define mathematical entity types

```python
entity_types = {
    "MathConcept": MathConcept,
    "PracticeProblem": PracticeProblem,
    "LearningObjective": LearningObjective
}
```

**Pros**: Domain-specific extraction
**Cons**: Most complex setup

## Implementation Plan

### Week 2: Test All 3 Tiers

**Day 1-2**: Build all 3 chunkers
**Day 3-4**: Test each tier with small sample (Section 1.1)
**Day 5-7**: Compare results and pick winner

### Testing Process
1. Extract Section 1.1 from cleaned text
2. Apply Tier 1, 2, and 3 to same content  
3. Send to Zep and compare knowledge graphs
4. Measure: entity quality, relationship accuracy, completeness

### Success Criteria
- Clear mathematical concepts extracted
- Prerequisites and relationships identified
- Practice problems linked to concepts
- Best tier identified for full textbook processing
