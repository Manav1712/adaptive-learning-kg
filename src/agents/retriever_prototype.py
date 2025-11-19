"""Retriever Prototype: Embedding-Based Retrieval

Embedding-based retrieval using dense semantic search on MathGPT KG data.
Uses OpenAI text-embedding-3 with FAISS for efficient similarity search.
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
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


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
    - base_text: Text for embedding-based retrieval (LO title or content text)
    - enriched_text: Expanded text with graph context (kept for reference, not used in embeddings)
    
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


def build_embedding_index(lo_corpus_df, content_corpus_df, embedding_model_name: str = "text-embedding-3-large"):
    """
    Build FAISS vector indexes for efficient embedding-based similarity search.
    
    Uses OpenAI's text-embedding-3 model to encode LO and content texts into dense vectors,
    then creates FAISS indexes for fast cosine similarity search. Processes texts
    in batches via API calls.
    
    Args:
        lo_corpus_df: DataFrame with LO corpus (must have 'base_text' column)
        content_corpus_df: DataFrame with content corpus (must have 'base_text' column)
    
    Returns:
        tuple: (embedding_client, lo_index, content_index)
            - embedding_client: OpenAI client instance (for encoding queries)
            - lo_index: FAISS IndexFlatIP for LOs
            - content_index: FAISS IndexFlatIP for content items
    
    Note:
        Uses OpenAI 'text-embedding-3-small' model (1536-dim embeddings). Vectors are L2-normalized
        for cosine similarity via inner product. Requires OPENAI_API_KEY environment variable.
    """
    import time
    
    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    model = embedding_model_name  # default to highest-quality OpenAI embedding
    
    lo_texts = lo_corpus_df["base_text"].astype(str).tolist()
    content_texts = content_corpus_df["base_text"].astype(str).tolist()
    
    def encode_with_openai(texts, client, model):
        """
        Encode texts using OpenAI API in batches.
        
        Args:
            texts: List of text strings to encode
            client: OpenAI client instance
            model: Model name
        
        Returns:
            numpy array of shape (len(texts), embedding_dim) with float32 vectors
        """
        all_vecs = []
        batch_size = 100  # OpenAI allows up to 2048 texts per request, but we'll batch smaller
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if i % 200 == 0:
                print(f"  Encoded {i}/{len(texts)}...", end='\r')
            
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch
                )
                batch_vecs = [item.embedding for item in response.data]
                all_vecs.extend(batch_vecs)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"  Error encoding batch {i}: {e}")
                # Fallback: encode one at a time for this batch
                for text in batch:
                    try:
                        response = client.embeddings.create(model=model, input=[text])
                        all_vecs.append(response.data[0].embedding)
                        time.sleep(0.1)
                    except Exception as e2:
                        print(f"    Error encoding single text: {e2}")
                        # Use zero vector as fallback (dimension depends on model)
                        dim = 3072 if "large" in model else 1536
                        all_vecs.append([0.0] * dim)
        
        print(f"  Encoded {len(texts)}/{len(texts)}      ")
        return np.array(all_vecs).astype("float32")
    
    print(f"Encoding LOs with OpenAI {model}...")
    lo_vecs = encode_with_openai(lo_texts, client, model)
    
    print(f"Encoding content with OpenAI {model}...")
    content_vecs = encode_with_openai(content_texts, client, model)
    
    faiss.normalize_L2(lo_vecs)
    faiss.normalize_L2(content_vecs)
    
    dim = lo_vecs.shape[1]
    lo_index = faiss.IndexFlatIP(dim)
    content_index = faiss.IndexFlatIP(dim)
    lo_index.add(lo_vecs)
    content_index.add(content_vecs)
    
    print(f"Built FAISS indexes: {lo_index.ntotal} LOs, {content_index.ntotal} content items")
    
    return client, lo_index, content_index


class SimpleRetrieverAgent:
    """
    Simple retriever agent for embedding-based semantic search.
    
    Uses dense semantic search with FAISS and OpenAI text-embedding-3.
    Optional graph expansion adds related LOs (prerequisites) and content items
    via 1-hop traversal from top results.
    """
    
    def __init__(self, lo_corpus_df, content_corpus_df,
                 embedding_model=None, lo_index=None, content_index=None,
                 prereq_in_map=None, content_ids_map=None, embedding_model_name="text-embedding-3-large"):
        """
        Initialize the retriever agent with corpora and indexes.
        
        Args:
            lo_corpus_df: DataFrame with LO corpus (required)
            content_corpus_df: DataFrame with content corpus (required)
            embedding_model: OpenAI client instance (required for embedding method)
            lo_index: FAISS index for LOs (required for embedding method)
            content_index: FAISS index for content (required for embedding method)
            prereq_in_map: Dict mapping LO ID → prerequisite LO IDs (for graph expansion)
            content_ids_map: Dict mapping LO ID → content item IDs (for graph expansion)
            embedding_model_name: Name of the embedding model used (must match index)
        
        Note:
            Graph expansion requires prereq_in_map and content_ids_map.
        """
        self.lo_corpus_df = lo_corpus_df
        self.content_corpus_df = content_corpus_df
        self.embedding_client = embedding_model  # OpenAI client
        self.lo_index = lo_index
        self.content_index = content_index
        self.prereq_in_map = prereq_in_map or {}
        self.content_ids_map = content_ids_map or {}
        self.embedding_model_name = embedding_model_name
        self.rerank_model = "gpt-4o-mini"
        self.rerank_top_k = 10
        
        # Build lookup dicts for fast access
        self._lo_row_by_doc = {
            f"lo_{int(r.lo_id)}": i 
            for i, r in enumerate(lo_corpus_df.itertuples(index=False))
        }
        self._ct_row_by_doc = {
            r.doc_id: i 
            for i, r in enumerate(content_corpus_df.itertuples(index=False))
        }
    
    def retrieve(self, query: str, k_los: int = 5, k_content: int = 5, expand: bool = True, rerank: bool = False) -> dict:
        """
        Retrieve top-k Learning Objectives and content items for a query using embedding-based search.
        
        Args:
            query: Search query string
            k_los: Number of top Learning Objectives to return (default: 5)
            k_content: Number of top content items to return (default: 5)
            expand: If True, add graph-expanded results (prereqs and related content)
                via 1-hop traversal from top results (default: True)
            rerank: If True, use LLM reranking for better precision (default: False, slow/expensive)
        
        Returns:
            dict: {
                "los": List of hit dicts with (rank, score, doc_id, lo_id, title, snippet, type),
                "content": List of hit dicts with (rank, score, doc_id, type, lo_title, title, snippet)
            }
        
        Raises:
            AttributeError: If required indexes/models are not initialized
        """
        return self._retrieve_embedding(query, k_los, k_content, expand, rerank)
    
    def _retrieve_embedding(self, query: str, k_los: int, k_content: int, expand: bool, rerank: bool = False) -> dict:
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
        if self.embedding_client is None or self.lo_index is None or self.content_index is None:
            raise AttributeError("Embedding method requires embedding_client, lo_index, and content_index to be initialized")
        
        # Encode query using OpenAI (must use same model as index)
        response = self.embedding_client.embeddings.create(
            model=self.embedding_model_name,
            input=[query]
        )
        q = np.array(response.data[0].embedding).astype("float32").reshape(1, -1)
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

        # Optional LLM reranking (disabled by default for speed/cost)
        if rerank:
            lo_hits = self._rerank_hits(query, lo_hits, entity_type="LO")
            ct_hits = self._rerank_hits(query, ct_hits, entity_type="content")
        
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

    def _rerank_hits(self, query: str, hits: list, entity_type: str) -> list:
        """
        Refine ordering using an LLM reranker for higher precision.
        """
        if not hits or self.embedding_client is None:
            return hits

        top_hits = hits[: max(1, min(self.rerank_top_k, len(hits)))]
        doc_block = []
        for idx, hit in enumerate(top_hits, 1):
            snippet = (hit.get("snippet") or hit.get("title") or "")[:500]
            doc_block.append(
                f"{idx}. ID: {hit['doc_id']}\nTitle: {hit.get('title','')}\nSnippet: {snippet}\n"
            )
        doc_block_text = "\n".join(doc_block)

        prompt = f"""You are reranking {entity_type} results for a search engine.
Query: "{query}"
Documents:
{doc_block_text}

Return ONLY JSON: {{"reranked_ids": ["doc_id1","doc_id2",...]}} with documents ordered most relevant first."""

        create_params = {
            "model": self.rerank_model,
            "messages": [
                {"role": "system", "content": "You reorder search results. Output strict JSON."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        reranked_ids = []
        try:
            response = self.embedding_client.chat.completions.create(**create_params)
            payload = json.loads(response.choices[0].message.content)
            reranked_ids = payload.get("reranked_ids", [])
        except Exception as exc:
            print(f"  Rerank fallback ({entity_type}): {exc}")

        id_to_hit = {hit["doc_id"]: hit for hit in hits}
        reordered = []
        used = set()
        for doc_id in reranked_ids:
            if doc_id in id_to_hit and doc_id not in used:
                reordered.append(id_to_hit[doc_id])
                used.add(doc_id)

        for hit in hits:
            if hit["doc_id"] not in used:
                reordered.append(hit)

        for idx, hit in enumerate(reordered, 1):
            hit["rank"] = idx
            hit["score"] = max(0.0, 1.0 - 0.05 * (idx - 1))

        return reordered


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
    Evaluate embedding-based retrieval on a set of test queries.
    
    Runs retrieval on each query and exports results to a CSV for manual evaluation.
    Each row represents one retrieved result with metadata including rank, scores,
    and document information.
    
    Args:
        agent: SimpleRetrieverAgent instance with embedding method initialized
        test_queries: List of query strings to evaluate
        output_path: Optional path to save CSV. Defaults to results/retriever_eval.csv
    
    Returns:
        pd.DataFrame: Results DataFrame with columns:
            - intent_id: Query number (1-indexed)
            - query: Original query string
            - method: "embedding"
            - entity_type: "LO" or content type (concept, example, try_it)
            - rank: Position in results (1-5)
            - doc_id: Document identifier
            - lo_id: LO ID (if applicable)
            - title: Document title
            - snippet: First 200 chars of content
            - score: Retrieval score
            - relevance_score: Empty column for manual 1-5 ratings
    
    Note:
        For each query, retrieves top 5 LOs and top 5 content items,
        resulting in ~10 rows per query. After export, manually fill relevance_score
        to calculate aggregate metrics.
    """
    results = []
    
    print(f"\nEvaluating {len(test_queries)} queries...")
    for intent_id, query in enumerate(test_queries, 1):
        print(f"  Query {intent_id}/{len(test_queries)}: {query[:60]}...")
        
        # Run embedding method
        emb_results = agent.retrieve(query, k_los=5, k_content=5)
        
        # Export embedding results
        for r in emb_results["los"]:
            results.append(_format_result_row(intent_id, query, "embedding", r, "LO"))
        
        for r in emb_results["content"]:
            results.append(_format_result_row(intent_id, query, "embedding", r, r["type"]))
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Add manual evaluation column
    results_df["relevance_score"] = None  # 1-5 rating
    
    # Save to CSV
    if output_path is None:
        output_path = RESULTS_DIR / "retriever_eval.csv"
    
    results_df.to_csv(output_path, index=False)
    print(f"\nExported {len(results_df)} result rows to {output_path}")
    print("\nNext steps:")
    print("1. Open the CSV and manually rate relevance_score (1-5) for each result")
    print("2. Calculate aggregate metrics: average relevance, coverage, etc.")
    
    return results_df


def evaluate_methods_simple(agent, test_queries, output_path=None):
    """
    Evaluate embedding-based retrieval with simple output format: one row per query.
    
    Args:
        agent: SimpleRetrieverAgent instance with embedding method initialized
        test_queries: List of query strings to evaluate
        output_path: Optional path to save CSV. Defaults to results/retriever_eval_simple.csv
    
    Returns:
        pd.DataFrame: Results DataFrame with columns:
            - intent_id: Query number (1-indexed)
            - intent: Query string
            - top_5_los: JSON string with top 5 LO results
            - top_5_content: JSON string with top 5 content results
    """
    import json
    
    results = []
    
    print(f"\nEvaluating {len(test_queries)} queries (simple format)...")
    for intent_id, query in enumerate(test_queries, 1):
        print(f"  Query {intent_id}/{len(test_queries)}: {query[:60]}...")
        
        # Run embedding method with LLM reranking enabled
        emb_results = agent.retrieve(query, k_los=5, k_content=5, rerank=True)
        
        # Format results as JSON strings (convert numpy/pandas types to native Python types)
        def safe_int(val):
            """Convert value to int, handling None and numpy types."""
            if val is None or val == "":
                return ""
            return int(val)
        
        def safe_float(val):
            """Convert value to float, handling numpy types."""
            if val is None:
                return 0.0
            return float(val)
        
        # Combine LOs and content into a single list
        combined_results = []
        
        # Add LOs with entity_type marker
        for r in emb_results["los"]:
            combined_results.append({
                "rank": safe_int(r["rank"]),
                "doc_id": str(r["doc_id"]),
                "entity_type": "LO",
                "lo_id": safe_int(r.get("lo_id")),
                "type": None,
                "title": str(r["title"]),
                "score": safe_float(r["score"])
            })
        
        # Add content items with entity_type marker
        for r in emb_results["content"]:
            combined_results.append({
                "rank": safe_int(r["rank"]),
                "doc_id": str(r["doc_id"]),
                "entity_type": "content",
                "lo_id": None,
                "type": str(r["type"]),
                "title": str(r["title"]),
                "score": safe_float(r["score"])
            })
        
        # Sort by score descending to maintain relevance order
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Reassign ranks based on combined order
        for idx, item in enumerate(combined_results, 1):
            item["rank"] = idx
        
        results.append({
            "intent": query,
            "content": json.dumps(combined_results)
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    if output_path is None:
        output_path = RESULTS_DIR / "retriever_eval_simple.csv"
    
    results_df.to_csv(output_path, index=False)
    print(f"\nExported {len(results_df)} result rows to {output_path}")
    print(f"File location: {output_path.absolute()}")
    
    return results_df


def compare_query(agent, query):
    """
    Display embedding-based retrieval results for a single query.
    
    Prints formatted output showing top-5 LOs and top-5 content items with scores
    and document IDs for quick visual inspection.
    
    Args:
        agent: SimpleRetrieverAgent instance
        query: Query string to retrieve
    
    Returns:
        None (prints to stdout)
    """
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}")
    
    results = agent.retrieve(query, k_los=5, k_content=5)
    
    print("\nEMBEDDING RETRIEVAL RESULTS:")
    print("-" * 80)
    print("Top LOs:")
    for r in results["los"][:5]:
        print(f"  {r['rank']}. [{r['doc_id']}] (score: {r['score']:.3f}) {r['title']}")
    print("\nTop Content:")
    for r in results["content"][:5]:
        print(f"  {r['rank']}. [{r['doc_id']}] ({r['type']}, score: {r['score']:.3f}) {r['title']}")


def main():
    """
    Main execution function: build indexes, create agent, and run evaluation.
    
    Orchestrates the full pipeline:
    1. Load Knowledge Graph CSV files
    2. Build adjacency maps for graph traversal
    3. Build retrieval corpora (LO and content)
    4. Build FAISS embedding indexes
    5. Create SimpleRetrieverAgent with embedding method
    6. Run sanity checks on single queries
    7. Evaluate embedding retrieval on 25 test queries
    8. Export results CSV for manual evaluation
    
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
    # Note: prereq_out_map is built but unused (kept for future use)
    
    # Build corpus
    print("\n3. Building corpus...")
    lo_corpus_df, content_corpus_df = build_corpus(
        los_df, content_df, prereq_in_map, content_ids_map
    )
    
    # Build embedding index
    print("\n4. Building embedding index...")
    embedding_model_name = "text-embedding-3-large"  # Use large model for best quality
    embedding_model, lo_index, content_index = build_embedding_index(
        lo_corpus_df, content_corpus_df, embedding_model_name=embedding_model_name
    )
    
    # Create retriever agent
    print("\n5. Creating SimpleRetrieverAgent...")
    agent = SimpleRetrieverAgent(
        lo_corpus_df, content_corpus_df,
        embedding_model=embedding_model,
        lo_index=lo_index,
        content_index=content_index,
        prereq_in_map=prereq_in_map,
        content_ids_map=content_ids_map,
        embedding_model_name=embedding_model_name,
    )
    
    # Test embedding method
    print("\n6. Testing embedding retrieval...")
    test_query = "What is a derivative?"
    results_embedding = agent.retrieve(test_query, k_los=5, k_content=5)
    
    print(f"\nQuery: '{test_query}'")
    print("\nTop LOs:")
    for r in results_embedding["los"][:5]:
        print(f"  {r['rank']}. {r['doc_id']} (score: {r['score']:.3f}) - {r['title']}")
    print("\nTop Content:")
    for r in results_embedding["content"][:5]:
        print(f"  {r['rank']}. {r['doc_id']} ({r['type']}, score: {r['score']:.3f}) - {r['title']}")
    
    # Test with expansion
    print("\n7. Testing with graph expansion...")
    results_expanded = agent.retrieve(test_query, k_los=5, k_content=5, expand=True)
    
    print(f"\nQuery: '{test_query}' (expand=True)")
    print(f"Found {len(results_expanded['los'])} LOs, {len(results_expanded['content'])} content items")
    
    # Load 70 test queries
    print("\n8. Loading 70 test queries...")
    test_queries = [
        # Category 1: Generic Questions (14)
        "What is calculus?",
        "How do I find the derivative of a function?",
        "Explain integration",
        "What are limits?",
        "How do derivatives work?",
        "What is the fundamental theorem of calculus?",
        "Explain continuous functions",
        "What are trigonometric functions?",
        "How do I solve optimization problems?",
        "What is the chain rule?",
        "Explain related rates",
        "What are antiderivatives?",
        "How do I find the area under a curve?",
        "What is the difference between differentiation and integration?",
        # Category 2: Very Nuanced Questions (14)
        "How does the intermediate value theorem relate to the existence of roots in continuous functions?",
        "Why must a function be continuous on a closed interval for Rolle's theorem to apply?",
        "What is the relationship between differentiability and continuity, and can you have one without the other?",
        "How does the mean value theorem connect average rate of change to instantaneous rate of change?",
        "When solving related rates problems, why is it crucial to differentiate implicitly with respect to time?",
        "How does the substitution method in integration relate to the chain rule in differentiation?",
        "Why does L'Hôpital's rule only apply to indeterminate forms like 0/0 or ∞/∞?",
        "What is the geometric interpretation of the second derivative test for concavity?",
        "How do you determine whether an improper integral converges or diverges?",
        "Why is the linearization of a function at a point related to the tangent line?",
        "How does the disk method differ from the shell method when finding volumes of revolution?",
        "What conditions must be met for a function to satisfy the hypotheses of the extreme value theorem?",
        "How does implicit differentiation allow us to find derivatives of relations that aren't functions?",
        "Why does the power rule for integration require n ≠ -1 as a special case?",
        # Category 3: Image-Based Questions (14)
        "Looking at the graph of f(x) = x², what is the slope of the tangent line at x = 1?",
        "In the diagram showing secant lines approaching a tangent, how does this illustrate the limit definition of derivative?",
        "What does the shaded region under the curve represent in the Riemann sum illustration?",
        "How do you interpret the graph showing increasing and decreasing intervals of a function?",
        "In the velocity-time graph shown, what does the area under the curve represent?",
        "Looking at the unit circle diagram, how are sine and cosine values determined?",
        "What does the graph of the derivative tell you about the original function's behavior?",
        "In the illustration of related rates with a ladder sliding down a wall, how are the rates connected?",
        "How does the graph demonstrate that this function has a vertical asymptote?",
        "What do the critical points shown on this graph indicate about local extrema?",
        "In the diagram showing the disk method, how is each disk's volume calculated?",
        "How does the graph illustrate the difference between average and instantaneous rate of change?",
        "What does the concavity shown in this graph tell you about the second derivative?",
        "In the parametric curve illustration, how do x(t) and y(t) trace out the path?",
        # Category 4: Problem-Specific / Symbolic Queries (14)
        "How do I find the derivative of f(x) = 3x² + 5x - 7?",
        "What is the integral of ∫(2x + 3)dx?",
        "Evaluate lim(x→2) (x² - 4)/(x - 2)",
        "Find dy/dx if y = sin(x²)",
        "Solve ∫₀¹ x³dx",
        "What is d/dx[e^(2x)]?",
        "Find the derivative of f(x) = ln(x² + 1)",
        "Evaluate ∫cos(x)sin(x)dx",
        "What is the second derivative of y = x⁴ - 3x² + 2?",
        "Find lim(x→∞) (3x² + 2x)/(x² - 1)",
        "Solve dy/dx = 2xy given y(0) = 1",
        "What is ∫(1/x)dx?",
        "Find the derivative of f(x) = (x² + 1)/(x - 2)",
        "Evaluate ∫₁^e (1/x)dx",
        # Category 5: Objectively Wrong / Non-Existent Queries (14)
        "How do I use calculus to prove the Riemann hypothesis?",
        "What is the derivative of a quantum wave function?",
        "Explain how to integrate using blockchain technology",
        "How does calculus apply to string theory in 11 dimensions?",
        "What is the integral of a neural network activation function?",
        "How do I find the derivative of dark matter?",
        "Explain the calculus of non-Euclidean hyperbolic manifolds",
        "What is the Laplace transform of a social network graph?",
        "How do I use calculus to optimize cryptocurrency mining?",
        "What is the derivative of consciousness?",
        "Explain how to integrate using machine learning algorithms",
        "How does calculus prove the existence of parallel universes?",
        "What is the integral of political ideology?",
        "How do I find the derivative of a DNA sequence?",
    ]
    
    print(f"Generated {len(test_queries)} test queries")
    
    # Run evaluation (simple format: one row per query)
    print("\n9. Running evaluation...")
    results_df = evaluate_methods_simple(agent, test_queries)
    
    # Show results for first query
    print("\n10. Sample query results:")
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

