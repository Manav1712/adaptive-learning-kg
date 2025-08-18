# Implementation Plan — Adaptive Learning Platform

## Overview
This document translates the conceptual phases from `phases.md` into concrete development tasks with specific technologies, file structures, and code approaches based on the local-first requirements from `spec.md`.

---

## Phase 0 — Project Setup & Foundation (Week 0)

### 0.1 Project Structure
```
adaptive-learning-kg/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── coach.py
│   │   ├── retrieval.py
│   │   ├── pack_constructor.py
│   │   ├── tutor.py
│   │   ├── grader.py
│   │   ├── mastery_estimator.py
│   │   └── overlay_writer.py
│   ├── data/
│   │   ├── models/
│   │   │   ├── knowledge_graph.py
│   │   │   ├── session.py
│   │   │   └── overlay.py
│   │   └── storage/
│   │       ├── file_storage.py
│   │       └── sqlite_storage.py
│   ├── processing/
│   │   ├── corpus_ingestion.py
│   │   ├── entity_extraction.py
│   │   └── resolver.py
│   ├── retrieval/
│   │   ├── hybrid_search.py
│   │   └── embeddings.py
│   ├── api/
│   │   ├── main.py
│   │   └── routes.py
│   └── utils/
│       ├── config.py
│       └── logging.py
├── data/
│   ├── raw/           # OpenStax source files
│   ├── processed/     # Extracted and cleaned data
│   └── storage/       # JSON files for KG, sessions, etc.
├── web/
│   ├── index.html
│   ├── js/
│   │   ├── d3-viz.js
│   │   └── chat-ui.js
│   └── css/
│       └── styles.css
├── tests/
├── requirements.txt
├── docker-compose.yml
└── README.md
```

### 0.2 Technology Stack
- **Language**: Python 3.11+
- **Web Framework**: FastAPI for API endpoints
- **LLM**: OpenAI GPT-4 (all agents initially)
- **Orchestration**: Direct function calls (simple for local-first)
- **Storage**: JSON + SQLite
- **Vector Search**: Sentence-transformers + FAISS (local)
- **Frontend**: Vanilla JS + D3.js
- **Deployment**: Docker for containerization

### 0.3 Core Data Models
```python
# knowledge_graph.py
@dataclass
class KGNode:
    id: str
    type: str  # "learning_objective", "concept", "problem", "section"
    title: str
    content: str
    metadata: Dict[str, Any]

@dataclass  
class KGEdge:
    source: str
    target: str
    relation: str  # "PREREQUISITE_OF", "ASSESSED_BY", "BELONGS_TO"
    weight: float
    metadata: Dict[str, Any]

# session.py
@dataclass
class Session:
    session_id: str
    student_id: str
    target_los: List[str]
    current_focus: str
    timestamp: datetime
    status: str  # "active", "closed"
    
# overlay.py
@dataclass
class StudentOverlay:
    student_id: str
    mastery_edges: List[MasteryEdge]
    confusion_edges: List[ConfusionEdge]
    last_updated: datetime
```

### 0.4 Development Environment Setup
- [ ] Initialize Python project with Poetry/pip
- [ ] Set up FastAPI application structure
- [ ] Configure OpenAI API integration
- [ ] Set up local development database (SQLite)
- [ ] Create basic test framework structure

---

## Phase 1 — Static Knowledge Graph (Weeks 1-3)

### 1.1 3-Tiered Chunking Strategy (Week 1)
**Goal**: Test 3 chunking approaches from simple to advanced

**Tier 1: Simple Subheading Chunking**
```python
class SimpleChunker:
    def chunk_by_subheadings(self, text: str) -> List[str]:
        """Split by section headers (1.1, 1.2, etc.)"""
        pass
```

**Tier 2: Contextualized Chunking**
```python
class ContextualizedChunker:
    def add_document_context(self, chunk: str, context: dict) -> str:
        """Add chapter/section context to each chunk"""
        return f"Chapter {context['chapter']}, Section {context['section']}\n\n{chunk}"
```

**Tier 3: Custom Entity Types**
```python
class CustomEntityChunker:
    def __init__(self):
        self.math_entity_types = {
            "MathConcept": MathConcept,
            "PracticeProblem": PracticeProblem,
            "LearningObjective": LearningObjective
        }
```

**Deliverables**:
- [x] Text cleaner for OpenStax content (completed)
- [ ] Tier 1: Simple subheading chunker
- [ ] Tier 2: Contextualized chunker  
- [ ] Tier 3: Custom entity types
- [ ] Compare results across all 3 tiers

### 1.2 Zep Integration & Testing (Week 2)
**Goal**: Test all 3 chunking tiers with Zep

**Implementation**:
```python
class ZepTester:
    def test_tier(self, tier_number: int, chunks: List[str]) -> Dict:
        """Test a chunking tier and return KG quality metrics"""
        for chunk in chunks:
            episode = self.zep_client.add_episode(
                name=f"tier_{tier_number}_chunk_{i}",
                episode_body=chunk,
                source=EpisodeType.text
            )
        return self.analyze_kg_quality()
```

**Testing Approach**:
- Test Tier 1: Simple chunks only
- Test Tier 2: Chunks + context
- Test Tier 3: Chunks + context + custom entities
- Compare entity quality and relationships

**Deliverables**:
- [ ] Zep client integration
- [ ] Test framework for all 3 tiers
- [ ] KG quality comparison
- [ ] Best approach selection

### 1.3 Final Implementation (Week 3)
**Goal**: Build production pipeline with best chunking approach

**Implementation**:
```python
class ProductionProcessor:
    def __init__(self, best_tier: int):
        self.chunker = self.get_best_chunker(best_tier)
    
    def process_full_textbook(self) -> KnowledgeGraph:
        """Process entire textbook with winning strategy"""
        chunks = self.chunker.chunk_textbook(cleaned_text)
        return self.send_to_zep_batch(chunks)
```

**Deliverables**:
- [ ] Production-ready chunking pipeline
- [ ] Complete Calculus Volume 1 knowledge graph
- [ ] Basic visualization and validation

---

## Phase 2 — Retrieval Pipeline (Weeks 4-5)

### 2.1 Embedding & Search Infrastructure (Week 4)
**Goal**: Build hybrid search (BM25 + embeddings + graph traversal)

**Implementation**:
```python
class HybridSearch:
    def __init__(self):
        self.bm25_index = BM25Index()
        self.embedding_index = FAISSIndex()
        
    def search(self, query: str, center_node: str = None) -> List[SearchResult]:
        """Combine BM25, vector similarity, and graph expansion"""
        pass
        
    def bfs_expansion(self, seed_nodes: List[str], hops: int = 2) -> List[str]:
        """Graph traversal from target learning objectives"""
        pass
```

**Deliverables**:
- [ ] Pre-computed embeddings for all nodes
- [ ] BM25 index for text search
- [ ] FAISS index for vector similarity
- [ ] Graph traversal (BFS from target LO)

### 2.2 Core Agents Implementation (Week 4-5)
**Goal**: Implement Coach, Retrieval, Pack Constructor, Tutor

**Coach Agent**:
```python
class Coach:
    def process_turn(self, user_input: str, student_id: str, session_id: str) -> CoachDecision:
        """Main entry point for user interactions"""
        pass
        
    def map_to_learning_objective(self, query: str) -> str:
        """Map user query to relevant LO in knowledge graph"""
        pass
        
    def decide_action(self, current_session: Session) -> str:
        """Decide: continue/branch/switch session"""
        pass
```

**Retrieval Agent**:
```python
class RetrievalAgent:
    def retrieve_teaching_pack(self, target_los: List[str], student_id: str) -> RetrievalBundle:
        """Get LO + prerequisites + examples + exercises"""
        pass
        
    def get_prerequisites(self, lo: str) -> List[str]:
        """Find prerequisite learning objectives"""
        pass
        
    def get_examples_and_exercises(self, lo: str) -> Tuple[List[Example], List[Exercise]]:
        """Retrieve worked examples and practice problems"""
        pass
```

**Pack Constructor**:
```python
class PackConstructor:
    def build_teaching_pack(self, bundle: RetrievalBundle) -> TeachingPack:
        """Organize retrieved content into coherent teaching unit"""
        pass
        
    def add_citations(self, content: str, sources: List[Source]) -> str:
        """Add source citations to all content"""
        pass
```

**Tutor Agent**:
```python
class Tutor:
    def generate_explanation(self, pack: TeachingPack) -> str:
        """Create clear explanations from teaching pack"""
        pass
        
    def present_exercises(self, exercises: List[Exercise]) -> str:
        """Format exercises for student interaction"""
        pass
```

### 2.3 Session Management (Week 5)
**Goal**: Handle session state and continuity

**Implementation**:
```python
class SessionManager:
    def create_session(self, student_id: str, target_los: List[str]) -> Session:
        """Initialize new learning session"""
        pass
        
    def load_session(self, session_id: str) -> Session:
        """Load existing session from JSON storage"""
        pass
        
    def save_session(self, session: Session) -> None:
        """Persist session state to file"""
        pass
```

**Deliverables**:
- [ ] JSON-based session persistence
- [ ] Session continuity across interactions
- [ ] Session switching logic

---

## Phase 3 — Personalization Layer (Weeks 6-8)

### 3.1 Grading & Assessment (Week 6)
**Goal**: Implement exercise grading and attempt logging

**Implementation**:
```python
class Grader:
    def grade_attempt(self, exercise: Exercise, student_answer: str) -> GradeResult:
        """Score student responses with feedback"""
        pass
        
    def extract_common_errors(self, incorrect_answers: List[str]) -> List[str]:
        """Identify patterns in student mistakes"""
        pass
```

**Simple Approach**:
- **Exact match** for numeric answers
- **LLM-based grading** for open-ended responses
- **Pattern matching** for algebraic expressions

### 3.2 Mastery Estimation (Week 6-7)
**Goal**: Track student understanding over time

**Implementation**:
```python
class MasteryEstimator:
    def update_mastery(self, student_id: str, lo: str, performance: float) -> None:
        """Update mastery estimates based on performance"""
        pass
        
    def estimate_current_mastery(self, student_id: str, lo: str) -> float:
        """Get current mastery probability for a learning objective"""
        pass
        
    def detect_confusions(self, attempts: List[Attempt]) -> List[ConfusionPattern]:
        """Identify concepts student consistently confuses"""
        pass
```

**Approach**:
- **Simple Bayesian updates** (correct/incorrect ratio)
- **Temporal decay** for older performance
- **Confusion detection** from error patterns

### 3.3 Overlay Management (Week 7-8)
**Goal**: Maintain per-student learning overlay

**Implementation**:
```python
class OverlayWriter:
    def add_mastery_edge(self, student_id: str, lo: str, confidence: float) -> None:
        """Add/update mastery relationship"""
        pass
        
    def add_confusion_edge(self, student_id: str, concepts: List[str], evidence: str) -> None:
        """Record concept confusion patterns"""
        pass
        
    def get_student_overlay(self, student_id: str) -> StudentOverlay:
        """Load complete student overlay from JSON"""
        pass
```

**Deliverables**:
- [ ] Per-student JSON overlay files (`overlay_{student_id}.json`)
- [ ] Temporal tracking of mastery changes
- [ ] Confusion pattern storage

---

## Phase 4 — Adaptive Learning (Weeks 9-10)

### 4.1 Lesson Planning (Week 9)
**Goal**: Plan personalized learning paths

**Implementation**:
```python
class LessonPlanner:
    def plan_next_session(self, student_id: str, target_topic: str) -> LearningPlan:
        """Create personalized learning sequence"""
        pass
        
    def identify_knowledge_gaps(self, overlay: StudentOverlay, target_lo: str) -> List[str]:
        """Find missing prerequisite knowledge"""
        pass
        
    def sequence_prerequisites(self, gaps: List[str]) -> List[str]:
        """Order prerequisite topics optimally"""
        pass
```

### 4.2 Personalized Retrieval (Week 9-10)
**Goal**: Bias retrieval based on student overlay

**Enhanced Retrieval**:
```python
class PersonalizedRetrieval(RetrievalAgent):
    def retrieve_with_overlay(self, target_los: List[str], overlay: StudentOverlay) -> RetrievalBundle:
        """Adjust retrieval based on student mastery/confusion"""
        pass
        
    def boost_weak_prerequisites(self, results: List[SearchResult], overlay: StudentOverlay) -> List[SearchResult]:
        """Prioritize content for student's weak areas"""
        pass
```

### 4.3 Adaptive Tutoring Loop (Week 10)
**Goal**: Close the personalization feedback loop

**Deliverables**:
- [ ] Personalized content selection
- [ ] Prerequisite remediation
- [ ] Adaptive pacing based on mastery

---

## Phase 5 — Scale & Optimization (Week 11)

### 5.1 Performance Optimization
- [ ] Pre-compute frequent retrieval patterns
- [ ] Cache embeddings and search results
- [ ] Optimize graph traversal algorithms
- [ ] Implement result caching for common queries

### 5.2 Monitoring & Observability
- [ ] Performance metrics (response time, accuracy)
- [ ] Usage analytics and learning outcome tracking
- [ ] Error tracking and debugging tools
- [ ] Token usage monitoring for cost optimization

---

## Phase 6 — GraphRAG & Visualization (Week 12)

### 6.1 Community Summarization
**Goal**: Generate high-level topic overviews

**Implementation**:
```python
class GraphRAGSummarizer:
    def generate_community_summaries(self, communities: List[Community]) -> List[Summary]:
        """Create C0/C1 community summaries offline"""
        pass
        
    def create_hierarchical_overviews(self, kg: KG) -> Dict[str, Overview]:
        """Build nested topic overviews"""
        pass
```

### 6.2 D3.js Visualization
**Goal**: Interactive knowledge graph visualization

**Deliverables**:
- [ ] Web interface for KG exploration
- [ ] Student progress visualization overlay
- [ ] Learning path visualization
- [ ] Interactive concept relationship explorer

### 6.3 Answer Composition
**Goal**: Structured responses for broad queries

**Implementation**:
```python
class AnswerComposer:
    def compose_unit_review(self, topic: str, student_id: str) -> StructuredReview:
        """Create comprehensive unit reviews"""
        pass
        
    def blend_global_and_personal(self, summary: Summary, overlay: StudentOverlay) -> PersonalizedSummary:
        """Merge global knowledge with personal learning state"""
        pass
```

---

## Development Priorities

### Immediate Next Steps (Week 0):
1. **Set up project structure** and development environment
2. **Implement basic data models** and file storage classes
3. **Create minimal OpenStax scraper** for one chapter
4. **Build simple entity extraction** pipeline with GPT-4

### Success Criteria for Each Phase:
- **Phase 1**: Static KG with 500+ nodes, visualizable in D3
- **Phase 2**: Working Q&A with citations ("What is vertex form?" → structured response)
- **Phase 3**: Student progress tracking with simple mastery scores
- **Phase 4**: Personalized responses based on student history
- **Phase 5**: Sub-2s response times, stable performance
- **Phase 6**: Rich visualizations and comprehensive reviews

### Data File Structure:
```
data/storage/
├── kg_nodes.json          # Static knowledge graph nodes
├── kg_edges.json          # Static knowledge graph relationships
├── embeddings.json        # Pre-computed vector embeddings
├── sessions/
│   ├── session_001.json   # Individual session logs
│   └── session_002.json
├── overlays/
│   ├── overlay_student_1.json  # Per-student learning state
│   └── overlay_student_2.json
└── summaries.json         # GraphRAG community summaries
```

This implementation plan translates the conceptual phases into concrete development tasks while maintaining the local-first, extensible architecture described in the specification.
