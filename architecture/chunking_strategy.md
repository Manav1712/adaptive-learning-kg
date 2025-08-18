# Chunking Strategy for OpenStax Calculus Knowledge Graph

## Why Chunking Strategy Matters

Effective chunking is critical for building high-quality knowledge graphs from educational content. Based on Zep's research, **chunking articles into multiple episodes improved results compared to treating each article as a single episode**. This approach generates:

- **More detailed knowledge graphs** with richer node and edge extraction
- **Better entity relationships** between mathematical concepts
- **Clearer learning pathways** and prerequisite structures
- **Improved concept isolation** for better student personalization

Poor chunking leads to sparse, high-level graphs that miss the nuanced relationships between mathematical concepts, problems, and learning objectives.

---

## OpenStax Content Structure Analysis

### Content Types in Calculus Volume 1
Our cleaned text contains several distinct types of educational content:

#### 1. **Learning Objectives** 
```
Learning Objectives
1.1.1 Use functional notation to evaluate a function.
1.1.2 Determine the domain and range of a function.
1.1.3 Draw the graph of a function.
```

#### 2. **Definitions and Concepts**
```
Definition
A function f consists of a set of inputs, a set of outputs, and a rule for assigning each input to exactly one output.
```

#### 3. **Mathematical Examples**
```
For example, consider the function f, where the domain is the set of all real numbers and the rule is to square the input.
```

#### 4. **Practice Problems**
```
Find the domain and range of f(x) = √(x - 3).
```

#### 5. **Theorems and Proofs**
```
Theorem: Every polynomial function is continuous on (-∞, ∞).
```

### OpenStax Structural Patterns
- **Hierarchical organization**: Chapter → Section → Subsection
- **Mixed content blocks**: Definitions mixed with examples and problems
- **Cross-references**: Frequent references to earlier concepts
- **Progressive complexity**: Concepts build systematically

---

## Chunking Strategy Options

### Strategy A: Content-Type Based Chunking (Recommended)

**Approach**: Separate different types of educational content into focused chunks.

**Rationale**: Each content type serves a different educational purpose and should form distinct knowledge graph entities.

**Implementation**:
```
Chunk 1: Section 1.1 - Learning Objectives
Chunk 2: Section 1.1 - Core Definitions  
Chunk 3: Section 1.1 - Mathematical Examples
Chunk 4: Section 1.1 - Practice Problems
Chunk 5: Section 1.1 - Theorems and Proofs
```

**Advantages**:
- **Precise entity extraction**: Definitions become clean concept nodes
- **Clear relationship mapping**: Problems clearly assess specific concepts
- **Better educational structure**: Learning objectives link to relevant content
- **Optimal for personalization**: Students can focus on specific content types

### Strategy B: Section-Based Chunking

**Approach**: Keep complete sections together as single chunks.

**Implementation**:
```
Chunk 1: Complete Section 1.1 (if under 10k characters)
Chunk 2: Complete Section 1.2
Chunk 3: Complete Section 1.3
```

**Advantages**:
- **Preserves context**: Related concepts stay together
- **Simpler implementation**: Direct section splitting
- **Natural boundaries**: Follows textbook structure

**Disadvantages**:
- **Mixed entity types**: Definitions and problems in same chunk
- **Potential size issues**: Some sections may exceed 10k characters

### Strategy C: Hybrid Approach

**Approach**: Use content-type chunking for complex sections, section-based for simple ones.

**Implementation**: 
- Sections > 8k characters → content-type chunking
- Sections < 8k characters → section-based chunking

---

## Implementation Plan for Week 2

### Day 1-2: Chunk Processing Development

#### Build Content-Type Extractor
```python
class ContentTypeExtractor:
    def extract_learning_objectives(self, text: str) -> str
    def extract_definitions(self, text: str) -> str  
    def extract_examples(self, text: str) -> str
    def extract_problems(self, text: str) -> str
```

#### Build Section Splitter
```python
class SectionSplitter:
    def split_by_sections(self, text: str) -> List[Tuple[str, str]]
    def estimate_chunk_sizes(self, sections: List[str]) -> List[int]
```

### Day 3-4: Strategy Testing

#### Test Setup
1. **Extract Chapter 1, Section 1.1** from cleaned text
2. **Apply all three chunking strategies**
3. **Send one small test chunk to Zep** (~2k characters)
4. **Analyze entity extraction results**

#### Evaluation Criteria
- **Entity Quality**: Are mathematical concepts extracted as clean entities?
- **Relationship Accuracy**: Do problems link to concepts they assess?
- **Learning Structure**: Are prerequisites and progressions captured?
- **Completeness**: Is important educational content preserved?

### Day 5-7: Pipeline Implementation

#### Build Optimal Chunker
Based on test results, implement the winning strategy:

```python
class OptimalChunker:
    def __init__(self, strategy: str):
        self.strategy = strategy
    
    def chunk_textbook(self, text: str) -> List[Dict[str, Any]]:
        # Returns chunks ready for Zep ingestion
        pass
```

#### Integrate with Zep Client
```python
def process_calculus_textbook():
    # 1. Load cleaned text
    text = load_cleaned_calculus()
    
    # 2. Apply optimal chunking
    chunks = optimal_chunker.chunk_textbook(text)
    
    # 3. Send to Zep in batches of 20
    for batch in chunks_of_20(chunks):
        zep_client.graph.add_batch(batch)
    
    # 4. Monitor processing status
    monitor_ingestion_progress()
```

---

## Expected Outcomes

### Knowledge Graph Entities
- **Learning Objectives**: Clear educational goals as nodes
- **Mathematical Concepts**: Definitions and properties as concept nodes  
- **Practice Problems**: Assessment items linked to concepts
- **Examples**: Worked solutions demonstrating concepts
- **Theorems**: Formal mathematical statements

### Knowledge Graph Relationships
- **TEACHES**: Learning objectives → concepts
- **PREREQUISITE_OF**: Concept A → Concept B  
- **ASSESSES**: Problems → concepts
- **DEMONSTRATES**: Examples → concepts
- **FORMALIZES**: Theorems → concepts

### Success Metrics
- **Entity Count**: ~100-200 entities from Chapter 1
- **Relationship Density**: 3-5 relationships per concept
- **Educational Structure**: Clear learning progression
- **Query Capability**: "What are the prerequisites for derivatives?"

---

## Next Steps

### Week 3 Goals
- **Scale to full textbook**: Process all chapters using optimal strategy
- **Add custom entity types**: Define mathematical concept schemas
- **Implement student overlay**: Track individual progress
- **Build retrieval system**: Query knowledge graph for tutoring

### Future Enhancements
- **Multi-modal processing**: Handle mathematical notation and images
- **Cross-textbook linking**: Connect concepts across multiple books
- **Adaptive chunking**: Adjust strategy based on content complexity
- **Quality metrics**: Automated evaluation of chunk effectiveness
