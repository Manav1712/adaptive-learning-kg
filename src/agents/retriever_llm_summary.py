"""LLM Summarization Retrieval Method

Adds a third retrieval method using LLM-generated summaries:
- Generate summaries for all LOs and content using OpenAI
- Build embedding index from summaries
- Retrieve using semantic search over summaries

This is a separate implementation to compare against embedding and BM25 methods.
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
from typing import List, Dict, Any
import time
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SUMMARIES_DIR = RESULTS_DIR / "summaries"
SUMMARIES_DIR.mkdir(exist_ok=True)


def load_data():
    """Load Knowledge Graph CSV files."""
    los_df = pd.read_csv(DATA_DIR / "lo_index.csv")
    content_df = pd.read_csv(DATA_DIR / "content_items.csv")
    print(f"Loaded {len(los_df)} LOs, {len(content_df)} content items")
    return los_df, content_df


def generate_summary(text: str, doc_type: str, client: OpenAI, model: str = "gpt-3.5-turbo") -> str:
    """
    Generate a concise summary of a document using LLM.
    
    Args:
        text: Document text to summarize
        doc_type: Type of document ("LO" or content type like "concept", "example")
        client: OpenAI client instance
        model: Model to use (default: gpt-3.5-turbo for cost efficiency)
    
    Returns:
        Summary string (2-3 sentences focusing on key concepts)
    """
    if not text or pd.isna(text) or str(text).strip() == "":
        return "No content available."
    
    # Truncate if too long (keep first 2000 chars)
    text_to_summarize = str(text)[:2000]
    
    prompt = f"""Summarize this {doc_type} in 2-3 sentences, focusing on the key concepts and learning objectives. 
Be concise and highlight the most important information for a student learning this material.

{text_to_summarize}"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful educational assistant that creates concise summaries of learning materials."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"  Error generating summary: {e}")
        return f"Error: {str(e)[:100]}"


def generate_all_summaries(los_df: pd.DataFrame, content_df: pd.DataFrame, 
                           cache_file: Path = None, model: str = "gpt-3.5-turbo",
                           limit_los: int = None, limit_content: int = None):
    """
    Generate summaries for all LOs and content items.
    
    Args:
        los_df: DataFrame with LO metadata
        content_df: DataFrame with content metadata
        cache_file: Optional path to cache file to save/load summaries
        model: OpenAI model to use
    
    Returns:
        tuple: (lo_summaries_df, content_summaries_df) with added 'llm_summary' column
    """
    # Check for cached summaries
    if cache_file and cache_file.exists():
        print(f"Loading cached summaries from {cache_file}")
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        lo_summaries = cache_data.get("lo_summaries", {})
        content_summaries = cache_data.get("content_summaries", {})
    else:
        lo_summaries = {}
        content_summaries = {}
    
    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    # Generate LO summaries
    print(f"\nGenerating summaries for {len(los_df)} LOs...")
    lo_summaries_list = []
    for idx, row in los_df.iterrows():
        lo_id = int(row.lo_id)
        lo_text = row.learning_objective
        
        if lo_id in lo_summaries:
            summary = lo_summaries[lo_id]
        else:
            print(f"  LO {idx+1}/{len(los_df)}: {lo_text[:50]}...")
            summary = generate_summary(lo_text, "Learning Objective", client, model)
            lo_summaries[lo_id] = summary
            time.sleep(0.1)  # Rate limiting
        
        lo_summaries_list.append({
            "lo_id": lo_id,
            "doc_id": f"lo_{lo_id}",
            "base_text": lo_text,
            "llm_summary": summary
        })
    
    # Generate content summaries
    print(f"\nGenerating summaries for {len(content_df)} content items...")
    content_summaries_list = []
    for idx, row in content_df.iterrows():
        cid = row.content_id
        ctype = row.content_type
        text = str(row.text or "")[:2000]
        
        if cid in content_summaries:
            summary = content_summaries[cid]
        else:
            print(f"  Content {idx+1}/{len(content_df)}: {ctype} {cid[:20]}...")
            summary = generate_summary(text, ctype, client, model)
            content_summaries[cid] = summary
            time.sleep(0.1)  # Rate limiting
        
        content_summaries_list.append({
            "content_id": cid,
            "doc_id": cid,
            "type": ctype,
            "base_text": text,
            "llm_summary": summary
        })
    
    # Save cache
    if cache_file:
        cache_data = {
            "lo_summaries": lo_summaries,
            "content_summaries": content_summaries
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"\nSaved summaries cache to {cache_file}")
    
    lo_summaries_df = pd.DataFrame(lo_summaries_list)
    content_summaries_df = pd.DataFrame(content_summaries_list)
    
    return lo_summaries_df, content_summaries_df


def build_summary_embedding_index(lo_summaries_df: pd.DataFrame, content_summaries_df: pd.DataFrame):
    """
    Build FAISS embedding index from LLM summaries.
    
    Args:
        lo_summaries_df: DataFrame with LO summaries
        content_summaries_df: DataFrame with content summaries
    
    Returns:
        tuple: (embedding_model, lo_index, content_index)
    """
    import torch
    import gc
    
    print("\nLoading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
    
    lo_summaries = lo_summaries_df["llm_summary"].astype(str).tolist()
    content_summaries = content_summaries_df["llm_summary"].astype(str).tolist()
    
    def encode_sequential(texts):
        """Encode texts one at a time to avoid multiprocessing issues."""
        all_vecs = []
        for i, text in enumerate(texts):
            if i % 20 == 0:
                print(f"  Encoded {i}/{len(texts)}...", end='\r')
            vec = embedding_model.encode([text], convert_to_numpy=True, show_progress_bar=False)
            all_vecs.append(vec[0])
        print(f"  Encoded {len(texts)}/{len(texts)}      ")
        return np.array(all_vecs).astype("float32")
    
    print("Encoding LO summaries...")
    with torch.no_grad():
        lo_vecs = encode_sequential(lo_summaries)
    
    print("Encoding content summaries...")
    with torch.no_grad():
        content_vecs = encode_sequential(content_summaries)
    
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


def retrieve_from_summaries(query: str, embedding_model, lo_index, content_index,
                             lo_summaries_df: pd.DataFrame, content_summaries_df: pd.DataFrame,
                             k_los: int = 5, k_content: int = 5):
    """
    Retrieve results using embeddings of LLM summaries.
    
    Args:
        query: Search query string
        embedding_model: SentenceTransformer model
        lo_index: FAISS index for LO summaries
        content_index: FAISS index for content summaries
        lo_summaries_df: DataFrame with LO summaries
        content_summaries_df: DataFrame with content summaries
        k_los: Number of top LOs to return
        k_content: Number of top content items to return
    
    Returns:
        dict: {"los": List[dict], "content": List[dict]} with hit dictionaries
    """
    q = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    
    D_lo, I_lo = lo_index.search(q, k_los)
    D_ct, I_ct = content_index.search(q, k_content)
    
    lo_hits = []
    for rank, (score, idx) in enumerate(zip(D_lo[0], I_lo[0]), 1):
        row = lo_summaries_df.iloc[int(idx)]
        lo_hits.append({
            "rank": rank,
            "score": float(score),
            "doc_id": row["doc_id"],
            "lo_id": row["lo_id"],
            "title": row["base_text"],
            "snippet": row["llm_summary"],
            "type": "LO",
        })
    
    ct_hits = []
    for rank, (score, idx) in enumerate(zip(D_ct[0], I_ct[0]), 1):
        row = content_summaries_df.iloc[int(idx)]
        ct_hits.append({
            "rank": rank,
            "score": float(score),
            "doc_id": row["doc_id"],
            "type": row["type"],
            "title": row["base_text"][:100],
            "snippet": row["llm_summary"],
        })
    
    return {"los": lo_hits, "content": ct_hits}


def evaluate_llm_summary_method(lo_summaries_df: pd.DataFrame, content_summaries_df: pd.DataFrame,
                                 embedding_model, lo_index, content_index, test_queries: List[str],
                                 output_path: Path = None):
    """
    Evaluate LLM summary retrieval method on test queries.
    
    Args:
        lo_summaries_df: DataFrame with LO summaries
        content_summaries_df: DataFrame with content summaries
        embedding_model: SentenceTransformer model
        lo_index: FAISS index for LO summaries
        content_index: FAISS index for content summaries
        test_queries: List of query strings
        output_path: Path to save CSV results
    
    Returns:
        pd.DataFrame: Results DataFrame
    """
    results = []
    
    print(f"\nEvaluating LLM summary method on {len(test_queries)} queries...")
    for intent_id, query in enumerate(test_queries, 1):
        print(f"  Query {intent_id}/{len(test_queries)}: {query[:60]}...")
        
        hits = retrieve_from_summaries(query, embedding_model, lo_index, content_index,
                                       lo_summaries_df, content_summaries_df, k_los=5, k_content=5)
        
        # Format results
        for r in hits["los"]:
            results.append({
                "intent_id": intent_id,
                "query": query,
                "method": "llm_summary",
                "entity_type": "LO",
                "rank": r["rank"],
                "doc_id": r["doc_id"],
                "lo_id": r.get("lo_id", ""),
                "title": r["title"],
                "snippet": r["snippet"][:200],
                "score": r["score"],
            })
        
        for r in hits["content"]:
            results.append({
                "intent_id": intent_id,
                "query": query,
                "method": "llm_summary",
                "entity_type": r["type"],
                "rank": r["rank"],
                "doc_id": r["doc_id"],
                "lo_id": "",
                "title": r["title"],
                "snippet": r["snippet"][:200],
                "score": r["score"],
            })
    
    results_df = pd.DataFrame(results)
    results_df["relevance_score"] = None
    results_df["winner"] = None
    
    if output_path is None:
        output_path = RESULTS_DIR / "retriever_eval_llm_summary.csv"
    
    results_df.to_csv(output_path, index=False)
    print(f"\nExported {len(results_df)} result rows to {output_path}")
    
    return results_df


def main():
    """Main execution function for LLM summary retrieval."""
    print("=" * 60)
    print("LLM Summary Retrieval Method")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    los_df, content_df = load_data()
    
    # Generate summaries
    print("\n2. Generating LLM summaries...")
    cache_file = SUMMARIES_DIR / "summaries_cache.json"
    lo_summaries_df, content_summaries_df = generate_all_summaries(
        los_df, content_df, cache_file=cache_file, model="gpt-3.5-turbo"
    )
    
    # Build embedding index
    print("\n3. Building embedding index from summaries...")
    embedding_model, lo_index, content_index = build_summary_embedding_index(
        lo_summaries_df, content_summaries_df
    )
    
    # Test queries
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
    
    # Test single query
    print("\n4. Testing retrieval...")
    test_query = "What is a derivative?"
    results = retrieve_from_summaries(test_query, embedding_model, lo_index, content_index,
                                      lo_summaries_df, content_summaries_df, k_los=5, k_content=5)
    
    print(f"\nQuery: '{test_query}' (Method: LLM Summary)")
    print("\nTop LOs:")
    for r in results["los"][:5]:
        print(f"  {r['rank']}. {r['doc_id']} (score: {r['score']:.3f}) - {r['title']}")
    print("\nTop Content:")
    for r in results["content"][:5]:
        print(f"  {r['rank']}. {r['doc_id']} ({r['type']}, score: {r['score']:.3f}) - {r['title']}")
    
    # Evaluate on all queries
    print("\n5. Running evaluation...")
    results_df = evaluate_llm_summary_method(
        lo_summaries_df, content_summaries_df,
        embedding_model, lo_index, content_index,
        test_queries
    )
    
    print("\n" + "=" * 60)
    print("LLM Summary method complete!")
    print("=" * 60)
    print(f"\nResults saved to: {RESULTS_DIR / 'retriever_eval_llm_summary.csv'}")


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

