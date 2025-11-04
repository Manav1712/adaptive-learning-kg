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


def llm_as_retriever(query: str, lo_summaries_df: pd.DataFrame, content_summaries_df: pd.DataFrame,
                     client: OpenAI, model: str = "gpt-3.5-turbo", k_los: int = 5, k_content: int = 5):
    """
    Use LLM directly as retriever - no embeddings, just ask LLM to select relevant documents.
    
    This is a baseline method where we give the LLM the query and all document titles/summaries,
    and ask it to select the most relevant ones.
    
    Args:
        query: Search query string
        lo_summaries_df: DataFrame with LO summaries
        content_summaries_df: DataFrame with content summaries
        client: OpenAI client instance
        model: Model to use
        k_los: Number of top LOs to return
        k_content: Number of top content items to return
    
    Returns:
        dict: {"los": List[dict], "content": List[dict]} with hit dictionaries
    """
    # Build lists of documents with IDs and summaries
    lo_docs = []
    for _, row in lo_summaries_df.iterrows():
        lo_docs.append({
            "id": row["doc_id"],
            "lo_id": row["lo_id"],
            "title": row["base_text"],
            "summary": row.get("llm_summary", row["base_text"])
        })
    
    content_docs = []
    for _, row in content_summaries_df.iterrows():
        content_docs.append({
            "id": row["doc_id"],
            "type": row["type"],
            "title": row["base_text"][:100],
            "summary": row.get("llm_summary", row["base_text"][:200])
        })
    
    # Build prompt for LO selection
    lo_prompt = f"""Given this query: "{query}"

Select the {k_los} most relevant Learning Objectives from this list. Return a JSON object with a key "selected_ids" containing an array of document IDs in order of relevance (most relevant first).

Learning Objectives:
"""
    for doc in lo_docs:
        lo_prompt += f"- ID: {doc['id']}, Title: {doc['title']}, Summary: {doc['summary'][:150]}\n"
    
    lo_prompt += '\nReturn format: {"selected_ids": ["lo_123", "lo_456", ...]}'
    
    # Build prompt for content selection
    content_prompt = f"""Given this query: "{query}"

Select the {k_content} most relevant content items from this list. Return a JSON object with a key "selected_ids" containing an array of document IDs in order of relevance (most relevant first).

Content Items:
"""
    for doc in content_docs:
        content_prompt += f"- ID: {doc['id']}, Type: {doc['type']}, Title: {doc['title']}, Summary: {doc['summary'][:150]}\n"
    
    content_prompt += '\nReturn format: {"selected_ids": ["123_concept_1", "456_example_2", ...]}'
    
    # Call LLM for LO selection
    try:
        lo_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise document retrieval system. Return ONLY valid JSON objects with a 'selected_ids' key containing an array of document IDs."},
                {"role": "user", "content": lo_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        lo_result = json.loads(lo_response.choices[0].message.content)
        lo_selected_ids = lo_result.get("selected_ids", [])
        if not isinstance(lo_selected_ids, list):
            lo_selected_ids = []
    except Exception as e:
        print(f"  Error in LLM LO retrieval: {e}")
        lo_selected_ids = []
    
    # Call LLM for content selection
    try:
        content_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise document retrieval system. Return ONLY valid JSON objects with a 'selected_ids' key containing an array of document IDs."},
                {"role": "user", "content": content_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content_result = json.loads(content_response.choices[0].message.content)
        content_selected_ids = content_result.get("selected_ids", [])
        if not isinstance(content_selected_ids, list):
            content_selected_ids = []
    except Exception as e:
        print(f"  Error in LLM content retrieval: {e}")
        content_selected_ids = []
    
    # Build lookup maps
    lo_map = {doc["id"]: doc for doc in lo_docs}
    content_map = {doc["id"]: doc for doc in content_docs}
    
    # Format LO results
    lo_hits = []
    for rank, doc_id in enumerate(lo_selected_ids[:k_los], 1):
        if doc_id in lo_map:
            doc = lo_map[doc_id]
            lo_hits.append({
                "rank": rank,
                "score": 1.0 - (rank - 1) * 0.1,  # Fake score based on rank
                "doc_id": doc["id"],
                "lo_id": doc["lo_id"],
                "title": doc["title"],
                "snippet": doc["summary"],
                "type": "LO",
            })
    
    # Format content results
    ct_hits = []
    for rank, doc_id in enumerate(content_selected_ids[:k_content], 1):
        if doc_id in content_map:
            doc = content_map[doc_id]
            ct_hits.append({
                "rank": rank,
                "score": 1.0 - (rank - 1) * 0.1,  # Fake score based on rank
                "doc_id": doc["id"],
                "type": doc["type"],
                "title": doc["title"],
                "snippet": doc["summary"],
            })
    
    return {"los": lo_hits, "content": ct_hits}


def evaluate_llm_as_retriever(lo_summaries_df: pd.DataFrame, content_summaries_df: pd.DataFrame,
                               client: OpenAI, test_queries: List[str], model: str = "gpt-3.5-turbo",
                               output_path: Path = None):
    """
    Evaluate LLM-as-retriever baseline method on test queries.
    
    Args:
        lo_summaries_df: DataFrame with LO summaries
        content_summaries_df: DataFrame with content summaries
        client: OpenAI client instance
        test_queries: List of query strings
        model: Model to use
        output_path: Path to save CSV results
    
    Returns:
        pd.DataFrame: Results DataFrame
    """
    results = []
    
    print(f"\nEvaluating LLM-as-retriever baseline on {len(test_queries)} queries...")
    for intent_id, query in enumerate(test_queries, 1):
        print(f"  Query {intent_id}/{len(test_queries)}: {query[:60]}...")
        
        hits = llm_as_retriever(query, lo_summaries_df, content_summaries_df, client, model, k_los=5, k_content=5)
        
        # Format results
        for r in hits["los"]:
            results.append({
                "intent_id": intent_id,
                "query": query,
                "method": "llm_as_retriever",
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
                "method": "llm_as_retriever",
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
        output_path = RESULTS_DIR / "retriever_eval_llm_as_retriever.csv"
    
    results_df.to_csv(output_path, index=False)
    print(f"\nExported {len(results_df)} result rows to {output_path}")
    
    return results_df


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
    
    # Test LLM-as-retriever baseline
    print("\n4. Testing LLM-as-retriever baseline...")
    test_query = "What is a derivative?"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)
    
    llm_results = llm_as_retriever(test_query, lo_summaries_df, content_summaries_df, client, k_los=5, k_content=5)
    
    print(f"\nQuery: '{test_query}' (Method: LLM-as-retriever baseline)")
    print("\nTop LOs:")
    for r in llm_results["los"][:5]:
        print(f"  {r['rank']}. {r['doc_id']} (score: {r['score']:.3f}) - {r['title']}")
    print("\nTop Content:")
    for r in llm_results["content"][:5]:
        print(f"  {r['rank']}. {r['doc_id']} ({r['type']}, score: {r['score']:.3f}) - {r['title']}")
    
    # Test summary-based retrieval
    print("\n5. Testing summary-based retrieval...")
    test_query = "What is a derivative?"
    results = retrieve_from_summaries(test_query, embedding_model, lo_index, content_index,
                                      lo_summaries_df, content_summaries_df, k_los=5, k_content=5)
    
    print(f"\nQuery: '{test_query}' (Method: LLM Summary + Embeddings)")
    print("\nTop LOs:")
    for r in results["los"][:5]:
        print(f"  {r['rank']}. {r['doc_id']} (score: {r['score']:.3f}) - {r['title']}")
    print("\nTop Content:")
    for r in results["content"][:5]:
        print(f"  {r['rank']}. {r['doc_id']} ({r['type']}, score: {r['score']:.3f}) - {r['title']}")
    
    # Evaluate LLM-as-retriever baseline
    print("\n6. Running LLM-as-retriever baseline evaluation...")
    llm_as_retriever_results = evaluate_llm_as_retriever(
        lo_summaries_df, content_summaries_df, client, test_queries
    )
    
    # Evaluate on all queries
    print("\n7. Running summary-based evaluation...")
    results_df = evaluate_llm_summary_method(
        lo_summaries_df, content_summaries_df,
        embedding_model, lo_index, content_index,
        test_queries
    )
    
    print("\n" + "=" * 60)
    print("LLM Summary method complete!")
    print("=" * 60)
    print(f"\nResults saved to:")
    print(f"  - LLM-as-retriever baseline: {RESULTS_DIR / 'retriever_eval_llm_as_retriever.csv'}")
    print(f"  - Summary-based: {RESULTS_DIR / 'retriever_eval_llm_summary.csv'}")


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

