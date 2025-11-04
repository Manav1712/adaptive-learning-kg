"""Retriever Prototype: Embedding vs Summarization

Compare two retrieval strategies on MatchGPT KG data:
- Method 1: Embedding Retrieval - Dense semantic search using FAISS
- Method 2: Summarization Retrieval - Lexical search on summaries using BM25
"""

import os
import warnings

# Disable multiprocessing BEFORE any imports
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
import numpy as np
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def tokenize(text):
    """
    Tokenize text by removing punctuation, lowercasing, and splitting on whitespace.
    
    Args:
        text: Input text string
    
    Returns:
        List of tokens (words)
    """
    text = re.sub(r'[^\w\s]', ' ', str(text).lower())
    return text.split()


def load_data():
    """
    Load all Knowledge Graph CSV files from the processed data directory.
    
    Reads four CSV files:
    - lo_index.csv: Learning objectives with their IDs and titles
    - content_items.csv: Content items (concepts, examples, try_it) linked to LOs
    - edges_prereqs.csv: Prerequisite relationships between LOs
    - edges_content.csv: Content items linked to their parent LOs
    
    Returns:
        tuple: (los_df, content_df, edges_prereqs_df, edges_content_df)
            - los_df: DataFrame with LO metadata
            - content_df: DataFrame with content item metadata
            - edges_prereqs_df: DataFrame with prerequisite edges
            - edges_content_df: DataFrame with LO-to-content edges
    """
    los_df = pd.read_csv(DATA_DIR / "lo_index.csv")
    content_df = pd.read_csv(DATA_DIR / "content_items.csv")
    edges_prereqs_df = pd.read_csv(DATA_DIR / "edges_prereqs.csv")
    edges_content_df = pd.read_csv(DATA_DIR / "edges_content.csv")
    
    print(f"Loaded {len(los_df)} LOs, {len(content_df)} content items")
    print(f"Loaded {len(edges_prereqs_df)} prerequisite edges, {len(edges_content_df)} content edges")
    
    return los_df, content_df, edges_prereqs_df, edges_content_df


def build_adjacency_maps(edges_prereqs_df, edges_content_df):
    """
    Build adjacency maps for fast graph traversal from edge DataFrames.
    
    Creates three dictionaries for efficient lookups:
    - prereq_in_map: Maps LO ID → list of prerequisite LO IDs (incoming edges)
    - prereq_out_map: Maps LO ID → list of LOs that require this LO (outgoing edges, currently unused)
    - content_ids_map: Maps LO ID → list of associated content item IDs
    
    Args:
        edges_prereqs_df: DataFrame with columns (source_lo_id, target_lo_id, score)
        edges_content_df: DataFrame with columns (source_lo_id, target_content_id)
    
    Returns:
        tuple: (prereq_in_map, prereq_out_map, content_ids_map)
            - prereq_in_map: Dict[int, List[int]] - LO → prerequisite LOs
            - prereq_out_map: Dict[int, List[int]] - LO → dependent LOs (unused, kept for future use)
            - content_ids_map: Dict[int, List[str]] - LO → content item IDs
    """
    def _group_to_sorted_ids(df, key_col, id_col, score_col):
        """Helper: Group by key_col and return sorted lists of id_col values."""
        out = {}
        if len(df) == 0:
            return out
        for k, grp in df.groupby(key_col):
            if score_col in grp.columns:
                grp_sorted = grp.sort_values(score_col, ascending=False)
            else:
                grp_sorted = grp
            out[int(k)] = [int(x) for x in grp_sorted[id_col].tolist()]
        return out
    
    prereq_in_map = _group_to_sorted_ids(
        edges_prereqs_df, "target_lo_id", "source_lo_id", "score"
    )
    prereq_out_map = _group_to_sorted_ids(
        edges_prereqs_df, "source_lo_id", "target_lo_id", "score"
    )
    
    content_ids_map = {}
    if len(edges_content_df) > 0:
        for lo_id, grp in edges_content_df.groupby("source_lo_id"):
            content_ids_map[int(lo_id)] = grp["target_content_id"].astype(str).tolist()
    
    return prereq_in_map, prereq_out_map, content_ids_map


def build_corpus(los_df, content_df, prereq_in_map, content_ids_map):
    """
    Build retrieval corpora for Learning Objectives and content items.
    
    Creates two DataFrames, each with:
    - base_text: Minimal text for embedding-based retrieval (LO title or content text)
    - enriched_text: Expanded text with graph context for BM25 retrieval
    
    For LOs:
    - base_text: Just the LO title
    - enriched_text: LO title + top 3 prerequisite titles + top 3 content titles
    
    For content items:
    - base_text: Content text (truncated to 2000 chars)
    - enriched_text: Content type + parent LO title + content snippet + prerequisite info
    
    Args:
        los_df: DataFrame with LO metadata (lo_id, learning_objective)
        content_df: DataFrame with content metadata (content_id, content_type, lo_id_parent, text)
        prereq_in_map: Dict mapping LO ID → prerequisite LO IDs
        content_ids_map: Dict mapping LO ID → associated content item IDs
    
    Returns:
        tuple: (lo_corpus_df, content_corpus_df)
            - lo_corpus_df: DataFrame with columns (doc_id, type, lo_id, base_text, enriched_text, prereq_in_ids, content_ids)
            - content_corpus_df: DataFrame with columns (doc_id, type, lo_id_parent, lo_title, base_text, enriched_text)
    """
    lo_title_map = dict(los_df[["lo_id", "learning_objective"]].itertuples(index=False, name=None))
    content_title_map = {
        row.content_id: f"{row.content_type.capitalize()}: {row.learning_objective}"
        for row in content_df.itertuples(index=False)
    }
    
    def _top_titles(lo_ids, k=3):
        """Get top k LO titles from a list of LO IDs."""
        return [lo_title_map.get(lo, str(lo)) for lo in (lo_ids or [])[:k]]
    
    def _top_content_titles(c_ids, k=3):
        """Get top k content titles from a list of content IDs."""
        return [content_title_map.get(cid, cid) for cid in (c_ids or [])[:k]]
    
    # Build LO corpus
    lo_rows = []
    for row in los_df.itertuples(index=False):
        lo_id = int(row.lo_id)
        lo_title = row.learning_objective
        
        pr_in_ids = prereq_in_map.get(lo_id, [])
        cont_ids = content_ids_map.get(lo_id, [])
        
        pr_in_titles = _top_titles(pr_in_ids, 3)
        cont_titles = _top_content_titles(cont_ids, 3)
        
        base_text = lo_title
        enriched_text = f"{lo_title}. Prereqs: {'; '.join(pr_in_titles) or 'None'}. Supports: {'; '.join(cont_titles) or 'None'}."
        
        lo_rows.append({
            "doc_id": f"lo_{lo_id}",
            "type": "LO",
            "lo_id": lo_id,
            "base_text": base_text,
            "enriched_text": enriched_text,
            "prereq_in_ids": pr_in_ids,
            "content_ids": cont_ids,
        })
    
    lo_corpus_df = pd.DataFrame(lo_rows)
    
    # Build content corpus
    content_rows = []
    for row in content_df.itertuples(index=False):
        cid = row.content_id
        ctype = row.content_type
        parent_lo = int(row.lo_id_parent) if pd.notna(row.lo_id_parent) else None
        lo_title = lo_title_map.get(parent_lo, "") if parent_lo is not None else ""
        text = (row.text or "")[:2000]
        
        pr_in_ids = prereq_in_map.get(parent_lo, []) if parent_lo is not None else []
        pr_in_titles = _top_titles(pr_in_ids, 3)
        
        base_text = text
        prefix = f"{ctype.capitalize()}: {lo_title}".strip(": ")
        enriched_text = f"{prefix}. {text[:200]} Prereqs of LO: {'; '.join(pr_in_titles) or 'None'}."
        
        content_rows.append({
            "doc_id": cid,
            "type": ctype,
            "lo_id_parent": parent_lo,
            "lo_title": lo_title,
            "base_text": base_text,
            "enriched_text": enriched_text,
        })
    
    content_corpus_df = pd.DataFrame(content_rows)
    
    print(f"Built corpus: {len(lo_corpus_df)} LOs, {len(content_corpus_df)} content items")
    
    return lo_corpus_df, content_corpus_df


def build_embedding_index(lo_corpus_df, content_corpus_df):
    """
    Build FAISS vector indexes for efficient embedding-based similarity search.
    
    Uses sentence-transformers to encode LO and content texts into dense vectors,
    then creates FAISS indexes for fast cosine similarity search. Processes texts
    sequentially (one at a time) to avoid multiprocessing issues on macOS.
    
    Args:
        lo_corpus_df: DataFrame with LO corpus (must have 'base_text' column)
        content_corpus_df: DataFrame with content corpus (must have 'base_text' column)
    
    Returns:
        tuple: (embedding_model, lo_index, content_index)
            - embedding_model: SentenceTransformer model instance
            - lo_index: FAISS IndexFlatIP for LOs
            - content_index: FAISS IndexFlatIP for content items
    
    Note:
        Uses 'all-MiniLM-L6-v2' model (384-dim embeddings). Vectors are L2-normalized
        for cosine similarity via inner product. Sequential encoding avoids segfaults
        on macOS systems with multiprocessing issues.
    """
    import torch
    import gc
    
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
    
    lo_texts = lo_corpus_df["base_text"].astype(str).tolist()
    content_texts = content_corpus_df["base_text"].astype(str).tolist()
    
    # Encode one item at a time to completely avoid multiprocessing
    def encode_sequential(texts):
        """
        Encode texts sequentially (one at a time) to avoid multiprocessing issues.
        
        Args:
            texts: List of text strings to encode
        
        Returns:
            numpy array of shape (len(texts), embedding_dim) with float32 vectors
        """
        all_vecs = []
        for i, text in enumerate(texts):
            if i % 20 == 0:
                print(f"  Encoded {i}/{len(texts)}...", end='\r')
            vec = embedding_model.encode([text], convert_to_numpy=True, show_progress_bar=False)
            all_vecs.append(vec[0])
        print(f"  Encoded {len(texts)}/{len(texts)}      ")
        return np.array(all_vecs).astype("float32")
    
    print("Encoding LOs...")
    with torch.no_grad():
        lo_vecs = encode_sequential(lo_texts)
    
    print("Encoding content...")
    with torch.no_grad():
        content_vecs = encode_sequential(content_texts)
    
    # Force cleanup
    gc.collect()
    
    faiss.normalize_L2(lo_vecs)
    faiss.normalize_L2(content_vecs)
    
    dim = lo_vecs.shape[1]
    lo_index = faiss.IndexFlatIP(dim)
    content_index = faiss.IndexFlatIP(dim)
    lo_index.add(lo_vecs)
    content_index.add(content_vecs)
    
    print(f"Built FAISS indexes: {lo_index.ntotal} LOs, {content_index.ntotal} content items")
    
    return embedding_model, lo_index, content_index


def build_bm25_index(lo_corpus_df, content_corpus_df):
    """
    Build BM25 lexical search index over enriched text corpora.
    
    Creates a unified BM25 index from both LO and content enriched_text fields.
    BM25 performs keyword-based ranking using term frequency and inverse document
    frequency, making it suitable for exact keyword matching queries.
    
    Args:
        lo_corpus_df: DataFrame with LO corpus (must have 'doc_id' and 'enriched_text' columns)
        content_corpus_df: DataFrame with content corpus (must have 'doc_id' and 'enriched_text' columns)
    
    Returns:
        tuple: (bm25_index, all_doc_ids)
            - bm25_index: BM25Okapi index instance ready for scoring
            - all_doc_ids: List of doc_ids in same order as indexed documents
                (LOs first, then content items)
    
    Note:
        Text is tokenized by removing punctuation, lowercasing, and splitting on whitespace.
        The enriched_text includes graph context (prereqs, related content) to improve
        keyword matching relevance.
    """
    # Combine LO and content enriched_text
    lo_texts = lo_corpus_df["enriched_text"].astype(str).tolist()
    content_texts = content_corpus_df["enriched_text"].astype(str).tolist()
    all_texts = lo_texts + content_texts
    
    # Build doc_id list
    lo_doc_ids = lo_corpus_df["doc_id"].tolist()
    content_doc_ids = content_corpus_df["doc_id"].tolist()
    all_doc_ids = lo_doc_ids + content_doc_ids
    
    # Tokenize and build index
    tokenized = [tokenize(text) for text in all_texts]
    bm25_index = BM25Okapi(tokenized)
    
    print(f"Built BM25 index: {len(all_texts)} documents")
    
    return bm25_index, all_doc_ids


class SimpleRetrieverAgent:
    """
    Simple retriever agent for comparing embedding-based vs BM25 retrieval methods.
    
    Supports two retrieval strategies:
    - Embedding: Dense semantic search using FAISS and sentence-transformers
    - Summary (BM25): Lexical keyword search using BM25 ranking
    
    Optional graph expansion adds related LOs (prerequisites) and content items
    via 1-hop traversal from top results.
    """
    
    def __init__(self, lo_corpus_df, content_corpus_df,
                 embedding_model=None, lo_index=None, content_index=None,
                 bm25_index=None, bm25_doc_ids=None,
                 prereq_in_map=None, content_ids_map=None):
        """
        Initialize the retriever agent with corpora and indexes.
        
        Args:
            lo_corpus_df: DataFrame with LO corpus (required)
            content_corpus_df: DataFrame with content corpus (required)
            embedding_model: SentenceTransformer model (for embedding method)
            lo_index: FAISS index for LOs (for embedding method)
            content_index: FAISS index for content (for embedding method)
            bm25_index: BM25Okapi index (for BM25 method)
            bm25_doc_ids: List of doc_ids matching bm25_index order (for BM25 method)
            prereq_in_map: Dict mapping LO ID → prerequisite LO IDs (for graph expansion)
            content_ids_map: Dict mapping LO ID → content item IDs (for graph expansion)
        
        Note:
            At least one retrieval method (embedding or BM25) should be provided.
            Graph expansion requires prereq_in_map and content_ids_map.
        """
        self.lo_corpus_df = lo_corpus_df
        self.content_corpus_df = content_corpus_df
        self.embedding_model = embedding_model
        self.lo_index = lo_index
        self.content_index = content_index
        self.bm25_index = bm25_index
        self.bm25_doc_ids = bm25_doc_ids or []
        self.prereq_in_map = prereq_in_map or {}
        self.content_ids_map = content_ids_map or {}
        
        # Build lookup dicts for fast access
        self._lo_row_by_doc = {
            f"lo_{int(r.lo_id)}": i 
            for i, r in enumerate(lo_corpus_df.itertuples(index=False))
        }
        self._ct_row_by_doc = {
            r.doc_id: i 
            for i, r in enumerate(content_corpus_df.itertuples(index=False))
        }
    
    def retrieve(self, query: str, method: str = "embedding", 
                 k_los: int = 5, k_content: int = 5, expand: bool = False) -> dict:
        """
        Retrieve top-k Learning Objectives and content items for a query.
        
        Args:
            query: Search query string
            method: Retrieval method - "embedding" (semantic) or "summary" (BM25 lexical)
            k_los: Number of top Learning Objectives to return (default: 5)
            k_content: Number of top content items to return (default: 5)
            expand: If True, add graph-expanded results (prereqs and related content)
                via 1-hop traversal from top results (default: False)
        
        Returns:
            dict: {
                "los": List of hit dicts with (rank, score, doc_id, lo_id, title, snippet, type),
                "content": List of hit dicts with (rank, score, doc_id, type, lo_title, title, snippet)
            }
        
        Raises:
            AttributeError: If required indexes/models are not initialized for the chosen method
        """
        if method == "embedding":
            return self._retrieve_embedding(query, k_los, k_content, expand)
        elif method == "summary":
            return self._retrieve_bm25(query, max(k_los, k_content) * 2, expand)
        else:
            return {"los": [], "content": []}
    
    def _retrieve_embedding(self, query: str, k_los: int, k_content: int, expand: bool) -> dict:
        """
        Retrieve results using dense embedding similarity search.
        
        Encodes the query into a vector and performs cosine similarity search
        against separate FAISS indexes for LOs and content items. Returns top-k
        results ranked by semantic similarity.
        
        Args:
            query: Search query string
            k_los: Number of top LOs to retrieve
            k_content: Number of top content items to retrieve
            expand: Whether to add graph-expanded results
        
        Returns:
            dict: {"los": List[dict], "content": List[dict]} with hit dictionaries
        
        Note:
            Scores are cosine similarity (0.0-1.0 range). Higher scores indicate
            greater semantic relevance. Separate searches are performed for LOs
            and content to maintain balanced results.
        """
        q = self.embedding_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        
        D_lo, I_lo = self.lo_index.search(q, k_los)
        D_ct, I_ct = self.content_index.search(q, k_content)
        
        lo_hits = []
        for rank, (score, idx) in enumerate(zip(D_lo[0], I_lo[0]), 1):
            row = self.lo_corpus_df.iloc[int(idx)]
            lo_hits.append({
                "rank": rank,
                "score": float(score),
                "doc_id": row["doc_id"],
                "lo_id": row["lo_id"],
                "title": row["base_text"],
                "snippet": row["base_text"],
                "type": "LO",
            })
        
        ct_hits = []
        for rank, (score, idx) in enumerate(zip(D_ct[0], I_ct[0]), 1):
            row = self.content_corpus_df.iloc[int(idx)]
            ct_hits.append({
                "rank": rank,
                "score": float(score),
                "doc_id": row["doc_id"],
                "type": row["type"],
                "lo_title": row["lo_title"],
                "title": row["lo_title"],
                "snippet": row["base_text"][:200],
            })
        
        if expand:
            lo_hits, ct_hits = self._expand_results(lo_hits, ct_hits)
        
        return {"los": lo_hits, "content": ct_hits}
    
    def _retrieve_bm25(self, query: str, k: int, expand: bool) -> dict:
        """
        Retrieve results using BM25 lexical search on enriched text.
        
        Tokenizes the query and scores all documents using BM25 term frequency
        ranking. Returns top-k LOs and top-k content items separately, maintaining
        balanced results across both types.
        
        Args:
            query: Search query string
            k: Target number of results per type (LOs and content)
            expand: Whether to add graph-expanded results
        
        Returns:
            dict: {"los": List[dict], "content": List[dict]} with hit dictionaries
        
        Note:
            BM25 scores are unbounded (typically 0-10+ range). Higher scores indicate
            better keyword matches. The unified index contains both LOs and content,
            so we retrieve more candidates (k*2) and split them into separate lists.
            Uses enriched_text which includes graph context for better keyword matching.
        """
        query_tokens = tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k * 2]  # Get more to split into LO/content
        
        lo_hits = []
        ct_hits = []
        
        for idx in top_indices:
            if len(lo_hits) >= k and len(ct_hits) >= k:
                break
            
            doc_id = self.bm25_doc_ids[idx]
            score = float(scores[idx])
            
            if doc_id.startswith("lo_") and len(lo_hits) < k:
                row_idx = self._lo_row_by_doc.get(doc_id)
                if row_idx is not None:
                    row = self.lo_corpus_df.iloc[row_idx]
                    lo_hits.append({
                        "rank": len(lo_hits) + 1,
                        "score": score,
                        "doc_id": doc_id,
                        "lo_id": row["lo_id"],
                        "title": row["base_text"],
                        "snippet": row["base_text"],
                        "type": "LO",
                    })
            elif not doc_id.startswith("lo_") and len(ct_hits) < k:
                row_idx = self._ct_row_by_doc.get(doc_id)
                if row_idx is not None:
                    row = self.content_corpus_df.iloc[row_idx]
                    ct_hits.append({
                        "rank": len(ct_hits) + 1,
                        "score": score,
                        "doc_id": doc_id,
                        "type": row["type"],
                        "lo_title": row["lo_title"],
                        "title": row["lo_title"],
                        "snippet": row["base_text"][:200],
                    })
        
        if expand:
            lo_hits, ct_hits = self._expand_results(lo_hits, ct_hits)
        
        return {"los": lo_hits, "content": ct_hits}
    
    def _expand_results(self, lo_hits: list, ct_hits: list) -> tuple:
        """
        Add graph-expanded results via 1-hop traversal from top LO results.
        
        Expands the result set by adding:
        - Prerequisite LOs (1-2 per top LO) that are required before learning this LO
        - Related content items (1-2 per top LO) that support learning this LO
        
        Args:
            lo_hits: List of LO hit dictionaries from initial retrieval
            ct_hits: List of content hit dictionaries from initial retrieval
        
        Returns:
            tuple: (expanded_lo_hits, expanded_ct_hits) - Lists with added graph neighbors
        
        Note:
            Only expands from top 2-3 LO results to avoid excessive expansion.
            Expanded results have scores reduced by 10% (multiplied by 0.9) to rank
            them below direct matches. Prevents duplicate additions using set tracking.
            Requires prereq_in_map and content_ids_map to be initialized.
        """
        expanded_lo_ids = set(h["lo_id"] for h in lo_hits)
        expanded_content_ids = set(h["doc_id"] for h in ct_hits)
        
        # Expand from top 2-3 LO results
        for hit in lo_hits[:3]:
            lo_id = hit["lo_id"]
            
            # Add 1-2 prereq LOs
            prereqs = self.prereq_in_map.get(lo_id, [])[:2]
            for pr_id in prereqs:
                if pr_id not in expanded_lo_ids:
                    doc_id = f"lo_{pr_id}"
                    row_idx = self._lo_row_by_doc.get(doc_id)
                    if row_idx is not None:
                        row = self.lo_corpus_df.iloc[row_idx]
                        lo_hits.append({
                            "rank": len(lo_hits) + 1,
                            "score": hit["score"] * 0.9,  # Boost slightly below direct matches
                            "doc_id": doc_id,
                            "lo_id": pr_id,
                            "title": row["base_text"],
                            "snippet": row["base_text"],
                            "type": "LO",
                        })
                        expanded_lo_ids.add(pr_id)
            
            # Add 1-2 linked content items
            content_ids = self.content_ids_map.get(lo_id, [])[:2]
            for cid in content_ids:
                if cid not in expanded_content_ids:
                    row_idx = self._ct_row_by_doc.get(cid)
                    if row_idx is not None:
                        row = self.content_corpus_df.iloc[row_idx]
                        ct_hits.append({
                            "rank": len(ct_hits) + 1,
                            "score": hit["score"] * 0.9,
                            "doc_id": cid,
                            "type": row["type"],
                            "lo_title": row["lo_title"],
                            "title": row["lo_title"],
                            "snippet": row["base_text"][:200],
                        })
                        expanded_content_ids.add(cid)
        
        return lo_hits, ct_hits


def _format_result_row(intent_id, query, method, hit, entity_type):
    """
    Format a single retrieval hit into an evaluation row dictionary.
    
    Args:
        intent_id: Query number (1-indexed)
        query: Original query string
        method: "embedding" or "summary"
        hit: Hit dictionary from retrieval (contains rank, doc_id, score, etc.)
        entity_type: "LO" or content type (concept, example, try_it)
    
    Returns:
        dict: Formatted row dictionary for evaluation DataFrame
    """
    return {
        "intent_id": intent_id,
        "query": query,
        "method": method,
        "entity_type": entity_type,
        "rank": hit["rank"],
        "doc_id": hit["doc_id"],
        "lo_id": hit.get("lo_id", "") or hit.get("lo_title", ""),
        "title": hit["title"],
        "snippet": hit["snippet"][:200],
        "score": hit["score"],
    }


def evaluate_methods(agent, test_queries, output_path=None):
    """
    Compare embedding vs BM25 retrieval methods on a set of test queries.
    
    Runs both retrieval methods on each query and exports results to a CSV
    for manual evaluation. Each row represents one retrieved result with
    metadata including method, rank, scores, and document information.
    
    Args:
        agent: SimpleRetrieverAgent instance with both methods initialized
        test_queries: List of query strings to evaluate
        output_path: Optional path to save CSV. Defaults to results/retriever_eval.csv
    
    Returns:
        pd.DataFrame: Results DataFrame with columns:
            - intent_id: Query number (1-indexed)
            - query: Original query string
            - method: "embedding" or "summary"
            - entity_type: "LO" or content type (concept, example, try_it)
            - rank: Position in results (1-5)
            - doc_id: Document identifier
            - lo_id: LO ID (if applicable)
            - title: Document title
            - snippet: First 200 chars of content
            - score: Retrieval score
            - relevance_score: Empty column for manual 1-5 ratings
            - winner: Empty column for manual method comparison ("embedding", "summary", "tie")
    
    Note:
        For each query, retrieves top 5 LOs and top 5 content items from both methods,
        resulting in ~20 rows per query. After export, manually fill relevance_score
        and winner columns to calculate aggregate metrics.
    """
    results = []
    
    print(f"\nEvaluating {len(test_queries)} queries...")
    for intent_id, query in enumerate(test_queries, 1):
        print(f"  Query {intent_id}/{len(test_queries)}: {query[:60]}...")
        
        # Run embedding method
        emb_results = agent.retrieve(query, method="embedding", k_los=5, k_content=5)
        
        # Run BM25 method
        bm25_results = agent.retrieve(query, method="summary", k_los=5, k_content=5)
        
        # Export embedding results
        for r in emb_results["los"]:
            results.append(_format_result_row(intent_id, query, "embedding", r, "LO"))
        
        for r in emb_results["content"]:
            results.append(_format_result_row(intent_id, query, "embedding", r, r["type"]))
        
        # Export BM25 results
        for r in bm25_results["los"]:
            results.append(_format_result_row(intent_id, query, "summary", r, "LO"))
        
        for r in bm25_results["content"]:
            results.append(_format_result_row(intent_id, query, "summary", r, r["type"]))
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Add manual evaluation columns
    results_df["relevance_score"] = None  # 1-5 rating
    results_df["winner"] = None  # "embedding", "summary", "tie"
    
    # Save to CSV
    if output_path is None:
        output_path = RESULTS_DIR / "retriever_eval.csv"
    
    results_df.to_csv(output_path, index=False)
    print(f"\nExported {len(results_df)} result rows to {output_path}")
    print("\nNext steps:")
    print("1. Open the CSV and manually rate relevance_score (1-5) for each result")
    print("2. For each query, mark winner column: 'embedding', 'summary', or 'tie'")
    print("3. Calculate aggregate metrics: win rate, avg relevance per method")
    
    return results_df


def compare_query(agent, query):
    """
    Display side-by-side comparison of embedding vs BM25 results for a single query.
    
    Prints formatted output showing top-5 LOs and top-5 content items from both
    retrieval methods, with scores and document IDs, for quick visual comparison.
    
    Args:
        agent: SimpleRetrieverAgent instance
        query: Query string to compare
    
    Returns:
        None (prints to stdout)
    """
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}")
    
    emb_results = agent.retrieve(query, method="embedding", k_los=5, k_content=5)
    bm25_results = agent.retrieve(query, method="summary", k_los=5, k_content=5)
    
    print("\nEMBEDDING METHOD:")
    print("-" * 80)
    print("Top LOs:")
    for r in emb_results["los"][:5]:
        print(f"  {r['rank']}. [{r['doc_id']}] (score: {r['score']:.3f}) {r['title']}")
    print("\nTop Content:")
    for r in emb_results["content"][:5]:
        print(f"  {r['rank']}. [{r['doc_id']}] ({r['type']}, score: {r['score']:.3f}) {r['title']}")
    
    print("\n\nBM25 METHOD:")
    print("-" * 80)
    print("Top LOs:")
    for r in bm25_results["los"][:5]:
        print(f"  {r['rank']}. [{r['doc_id']}] (score: {r['score']:.3f}) {r['title']}")
    print("\nTop Content:")
    for r in bm25_results["content"][:5]:
        print(f"  {r['rank']}. [{r['doc_id']}] ({r['type']}, score: {r['score']:.3f}) {r['title']}")


def main():
    """
    Main execution function: build indexes, create agent, and run evaluation.
    
    Orchestrates the full pipeline:
    1. Load Knowledge Graph CSV files
    2. Build adjacency maps for graph traversal
    3. Build retrieval corpora (LO and content)
    4. Build FAISS embedding indexes
    5. Build BM25 lexical index
    6. Create SimpleRetrieverAgent with both methods
    7. Run sanity checks on single queries
    8. Evaluate both methods on 25 test queries
    9. Export results CSV for manual evaluation
    
    Returns:
        None
    
    Note:
        Prints progress updates throughout execution. Final results are saved
        to results/retriever_eval.csv for manual relevance rating and comparison.
    """
    print("=" * 60)
    print("Retriever Prototype Sanity Checks")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    los_df, content_df, edges_prereqs_df, edges_content_df = load_data()
    
    # Build adjacency maps
    print("\n2. Building adjacency maps...")
    prereq_in_map, prereq_out_map, content_ids_map = build_adjacency_maps(
        edges_prereqs_df, edges_content_df
    )
    print(f"  Prereq in map: {len(prereq_in_map)} entries")
    print(f"  Content map: {len(content_ids_map)} entries")
    
    # Build corpus
    print("\n3. Building corpus...")
    lo_corpus_df, content_corpus_df = build_corpus(
        los_df, content_df, prereq_in_map, content_ids_map
    )
    
    # Build embedding index
    print("\n4. Building embedding index...")
    embedding_model, lo_index, content_index = build_embedding_index(
        lo_corpus_df, content_corpus_df
    )
    
    # Build BM25 index
    print("\n5. Building BM25 index...")
    bm25_index, bm25_doc_ids = build_bm25_index(lo_corpus_df, content_corpus_df)
    
    # Create retriever agent
    print("\n6. Creating SimpleRetrieverAgent...")
    agent = SimpleRetrieverAgent(
        lo_corpus_df, content_corpus_df,
        embedding_model=embedding_model,
        lo_index=lo_index,
        content_index=content_index,
        bm25_index=bm25_index,
        bm25_doc_ids=bm25_doc_ids,
        prereq_in_map=prereq_in_map,
        content_ids_map=content_ids_map,
    )
    
    # Test embedding method
    print("\n7. Testing embedding retrieval...")
    test_query = "What is a derivative?"
    results_embedding = agent.retrieve(test_query, method="embedding", k_los=5, k_content=5)
    
    print(f"\nQuery: '{test_query}' (Method: embedding)")
    print("\nTop LOs:")
    for r in results_embedding["los"][:5]:
        print(f"  {r['rank']}. {r['doc_id']} (score: {r['score']:.3f}) - {r['title']}")
    print("\nTop Content:")
    for r in results_embedding["content"][:5]:
        print(f"  {r['rank']}. {r['doc_id']} ({r['type']}, score: {r['score']:.3f}) - {r['title']}")
    
    # Test BM25 method
    print("\n8. Testing BM25 retrieval...")
    results_bm25 = agent.retrieve(test_query, method="summary", k_los=5, k_content=5)
    
    print(f"\nQuery: '{test_query}' (Method: summary/BM25)")
    print("\nTop LOs:")
    for r in results_bm25["los"][:5]:
        print(f"  {r['rank']}. {r['doc_id']} (score: {r['score']:.3f}) - {r['title']}")
    print("\nTop Content:")
    for r in results_bm25["content"][:5]:
        print(f"  {r['rank']}. {r['doc_id']} ({r['type']}, score: {r['score']:.3f}) - {r['title']}")
    
    # Test with expansion
    print("\n9. Testing with graph expansion...")
    results_expanded = agent.retrieve(test_query, method="embedding", k_los=5, k_content=5, expand=True)
    
    print(f"\nQuery: '{test_query}' (Method: embedding, expand=True)")
    print(f"Found {len(results_expanded['los'])} LOs, {len(results_expanded['content'])} content items")
    
    # Generate test queries
    print("\n10. Generating test queries...")
    test_queries = [
        "What is a derivative?",
        "How do I solve a quadratic equation?",
        "Explain the chain rule",
        "How do I find limits?",
        "What is continuity?",
        "Give me practice problems on integrals",
        "How do I use the product rule?",
        "What is the difference between a limit and a derivative?",
        "Explain the tangent problem",
        "How do I find the area under a curve?",
        "What are transcendental functions?",
        "How do I solve optimization problems?",
        "What is the mean value theorem?",
        "Explain L'Hopital's rule",
        "How do I find critical points?",
        "What is a secant line?",
        "How do I calculate rates of change?",
        "What is an integral?",
        "Explain the fundamental theorem of calculus",
        "How do I find antiderivatives?",
        "What is the difference between definite and indefinite integrals?",
        "How do I solve differential equations?",
        "What are piecewise functions?",
        "Explain function transformations",
        "How do I find the domain of a function?",
    ]
    
    print(f"Generated {len(test_queries)} test queries")
    
    # Run evaluation
    print("\n11. Running evaluation...")
    results_df = evaluate_methods(agent, test_queries)
    
    # Show side-by-side comparison for first query
    print("\n12. Side-by-side comparison (first query):")
    compare_query(agent, test_queries[0])
    
    print("\n" + "=" * 60)
    print("Sanity checks complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    try:
        main()
        print("\nScript completed successfully!")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

