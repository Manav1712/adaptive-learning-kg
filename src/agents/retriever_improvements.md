# Embedding-Based Retrieval: Best Practices & Improvements

## Current Implementation Analysis

### ✅ What's Good
- Uses OpenAI `text-embedding-3-small` (state-of-the-art)
- FAISS for efficient similarity search
- Proper vector normalization for cosine similarity
- Separate indexes for balanced LO/content results

### ⚠️ Areas for Improvement

#### 1. **Text Representation Quality**
**Current**: LOs only embed title (very short, loses context)
```python
base_text = lo_title  # Just "The Tangent Problem and Differential Calculus"
```

**Better**: Include enriched context in embeddings
```python
base_text = f"{lo_title}. Prereqs: {prereq_titles}. Related content: {content_titles}."
```

#### 2. **Hybrid Search (Semantic + Keyword)**
**Current**: Pure semantic search only

**Better**: Combine dense embeddings with sparse (BM25/keyword) search
- Semantic catches conceptual similarity
- Keyword catches exact term matches
- Weighted combination: `final_score = 0.7 * semantic + 0.3 * keyword`

#### 3. **Reranking**
**Current**: Returns raw FAISS similarity scores

**Better**: Use cross-encoder or LLM reranker
- Retrieve top 20-50 candidates
- Rerank with query-document interaction
- Return top-k reranked results

#### 4. **Query Expansion**
**Current**: Uses query as-is

**Better**: Expand queries with synonyms/related terms
- "derivative" → "derivative, differentiation, rate of change, slope"
- Improves recall for conceptual queries

#### 5. **Model Choice**
**Current**: `text-embedding-3-small` (1536 dims, fast, cheap)

**Better Options**:
- `text-embedding-3-large` (3072 dims) - Better quality, 2x cost
- For your dataset size (408 docs), small is fine
- Consider large if quality is critical

## Recommended Architecture

### Tier 1: Current (Good for MVP)
```
Query → OpenAI Embedding → FAISS Search → Top-k Results
```
**Pros**: Fast, cheap, simple
**Cons**: No reranking, limited text representation

### Tier 2: Improved (Recommended)
```
Query → OpenAI Embedding → FAISS Search (top-20) → Cross-Encoder Rerank → Top-k
         ↓
    Query Expansion → BM25 Keyword Search → Hybrid Merge
```
**Pros**: Better accuracy, still fast
**Cons**: More complex, needs reranker model

### Tier 3: Production (Best Quality)
```
Query → Multi-Query Expansion → 
    ├─ OpenAI Embedding → FAISS (top-50)
    ├─ BM25 Keyword Search (top-50)
    └─ Graph-Based Expansion
         ↓
    Hybrid Merge & Deduplication
         ↓
    Cross-Encoder Reranker (top-20)
         ↓
    LLM Reranker for final top-5
```
**Pros**: Maximum accuracy
**Cons**: Expensive, slow, complex

## Specific Improvements for Your Code

### 1. Use Enriched Text for Embeddings
```python
# Instead of just title, use enriched context
base_text = enriched_text  # Include prereqs and related content
```

### 2. Add BM25 Keyword Search
```python
from rank_bm25 import BM25Okapi

# Build BM25 index
tokenized_corpus = [doc.split() for doc in texts]
bm25 = BM25Okapi(tokenized_corpus)

# Search
bm25_scores = bm25.get_scores(query.split())
```

### 3. Add Reranking
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Rerank top candidates
pairs = [[query, doc] for doc in candidates]
rerank_scores = reranker.predict(pairs)
```

### 4. Hybrid Scoring
```python
# Normalize scores to [0, 1]
semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())
bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())

# Weighted combination
final_scores = 0.7 * semantic_norm + 0.3 * bm25_norm
```

## Performance Comparison

| Approach | Accuracy | Speed | Cost/Query | Complexity |
|----------|----------|-------|-----------|------------|
| **Current (Pure Embedding)** | 7/10 | ⚡⚡⚡ Fast | $0.0001 | Low |
| **+ Enriched Text** | 8/10 | ⚡⚡⚡ Fast | $0.0001 | Low |
| **+ Hybrid Search** | 8.5/10 | ⚡⚡ Medium | $0.0001 | Medium |
| **+ Reranking** | 9/10 | ⚡ Medium | $0.001 | Medium |
| **Full Production** | 9.5/10 | ⚡ Slow | $0.01 | High |

## Recommendation

For your use case (educational content retrieval with 138 LOs + 270 content items):

1. **Immediate Fix**: Use `enriched_text` for embeddings instead of just title
2. **Quick Win**: Add BM25 hybrid search (easy, big improvement)
3. **Quality Boost**: Add cross-encoder reranking (moderate effort, significant gain)
4. **Future**: Consider `text-embedding-3-large` if budget allows

Your current approach is **good enough for MVP**, but these improvements would significantly boost accuracy without major complexity.

