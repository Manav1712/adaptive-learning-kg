## Retriever Prototype Pipeline Mind Map

RETRIEVER PROTOTYPE PIPELINE

SETUP PHASE
│
├── Load Knowledge Graph Data
│   ├── Load learning objectives from CSV
│   ├── Load content items from CSV
│   ├── Load prerequisite relationships between LOs
│   └── Load relationships between LOs and content
│
├── Build Graph Maps
│   ├── Map each LO to its prerequisite LOs
│   ├── Map each LO to its linked content items
│   └── These maps help us find related items quickly
│
└── Build Retrieval Corpora
    ├── For Learning Objectives:
    │   ├── base_text: Just the LO title
    │   └── enriched_text: Title plus top 3 prereqs plus top 3 content titles
    │
    └── For Content Items:
        ├── base_text: The actual content text
        └── enriched_text: Content type plus parent LO title plus snippet plus prereqs

INDEX BUILDING PHASE
│
├── Build Embedding Index
│   ├── Load a sentence transformer model
│   ├── Convert each LO title into a vector
│   ├── Convert each content text into a vector
│   ├── Normalize all vectors
│   ├── Build a search index for LOs
│   └── Build a search index for content
│
└── Build BM25 Index
    ├── Tokenize all the enriched text
    ├── Build one unified keyword search index
    └── Keep track of which document is which

RETRIEVAL PHASE (When user asks a question)
│
├── Embedding Method
│   ├── User asks a question
│   ├── Convert question into a vector
│   ├── Search LO index for similar vectors
│   ├── Search content index for similar vectors
│   ├── Return top results ranked by similarity
│   └── Optionally add related items from graph
│
└── BM25 Method
    ├── User asks a question
    ├── Break question into words
    ├── Score all documents by keyword matching
    ├── Sort by score
    ├── Split into top LOs and top content
    ├── Return top results
    └── Optionally add related items from graph

GRAPH EXPANSION (Optional step)
│
├── Look at top 2-3 LO results
├── Add their prerequisite LOs (with slightly lower score)
└── Add their linked content items (with slightly lower score)

EVALUATION PHASE
│
├── Run both methods on test questions
├── Format results into rows
├── Save to CSV file
└── File has columns: query, method, document type, rank, 
    document ID, title, snippet, score, relevance rating, winner


### Summary

**What happens:**
1. Load data from CSV files
2. Build maps showing which LOs connect to which other LOs and content
3. Create two versions of text: simple for embeddings, enriched for keyword search
4. Build search indexes: one for vectors, one for keywords
5. When a question comes in, search both indexes
6. Return top results from each method
7. Optionally add related items from the graph
8. Compare both methods and save results

**The two methods:**
- Embedding method: Finds similar meaning using vectors
- BM25 Summarization method: Finds matching keywords in enriched text

**Graph usage:**
- Not doing deep graph traversal
- Using graph to add context to text (prereqs, related content)
- Optionally expanding results by adding neighbors

**Why not GraphRAG at the moment?**
We're using graph structure as metadata enrichment rather than graph-native retrieval. True GraphRAG would require graph embeddings (node2vec, GraphSAGE), multi-hop traversal algorithms, and subgraph discovery - which adds significant complexity. For this prototype I focused on comparing embedding vs BM25 retrieval methods. 

This keeps the comparison focused on embeddings vs keyword search while using graph relationships to improve both.

## Opinions on LLM Summarization for Retrieval

### Why it's slow
- 138 LOs + 270 content = **408 API calls**
- At ~0.5-1 second per call = **3-7 minutes total**
- Even with caching, first run is painful

### Why it's probably a bad idea
1. **Cost**: ~$0.01-0.02 per summary × 408 = **$4-8 per run**
2. **Latency**: Minutes to build index, not scalable to thousands of docs
3. **Maintenance**: Need to regenerate summaries when content changes
4. **Quality risk**: Summaries might lose key terminology needed for keyword matching
5. **Overkill**: If using embeddings anyway, why not just embed the original text?

### Recommendations

**Option 1: Test on tiny subset (recommended)**
- Limit to first 20 LOs and 30 content items
- Just to prove the concept works
- Document that it's not viable at scale

**Option 2: Skip it entirely**
- Focus on embedding vs BM25 comparison
- Both are instant and cost-free
- Document why LLM summarization was rejected

**Option 3: Use it strategically**
- Only for very long documents (>1000 words)
- Keep short documents as-is
- But your content is already relatively short

### My take
**Skip LLM summarization.** It's interesting academically but impractical:
- Your current comparison (embedding vs BM25) is solid
- Both methods are instant and free
- LLM summaries add cost/latency without clear benefit
- If semantic search works well with embeddings, summaries won't help much

The better investment is manual evaluation of the embedding vs BM25 results you already have.

## Opinions on LLM Summarization for Retrieval

### Why it's slow
- 138 LOs + 270 content = **408 API calls**
- At ~0.5-1 second per call = **3-7 minutes total**
- Even with caching, first run is painful

### Why it's probably a bad idea
1. **Cost**: ~$0.01-0.02 per summary × 408 = **$4-8 per run**
2. **Latency**: Minutes to build index, not scalable to thousands of docs
3. **Maintenance**: Need to regenerate summaries when content changes
4. **Quality risk**: Summaries might lose key terminology needed for keyword matching
5. **Overkill**: If using embeddings anyway, why not just embed the original text?

### Recommendations

**Use it strategically**
- Only for very long documents (>1000 words)
- Keep short documents as-is
- But your content is already relatively short

### My take
**Skip LLM summarization.** It's interesting academically but impractical:
- Your current comparison (embedding vs BM25) is solid
- Both methods are instant and free
- LLM summaries add cost/latency without clear benefit
- If semantic search works well with embeddings, summaries won't help much

