"""LLM Full-Context Retrieval

Uses a high-quality LLM (GPT-4 Turbo or Claude) with the entire dataset in context.
This approach puts all documents in a single prompt and asks the LLM to select
the most relevant ones.

Pros:
- Can leverage full document text (not just summaries)
- High-quality reasoning from advanced models
- No information loss from truncation

Cons:
- Expensive (large context window)
- Slow (multiple API calls per query)
- Context window limits (may need batching for very large datasets)
"""

import os
import warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
from pathlib import Path
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    """Load Knowledge Graph CSV files."""
    los_df = pd.read_csv(DATA_DIR / "lo_index.csv")
    content_df = pd.read_csv(DATA_DIR / "content_items.csv")
    print(f"Loaded {len(los_df)} LOs, {len(content_df)} content items")
    return los_df, content_df


def count_tokens_approx(text: str) -> int:
    """
    Rough token count estimate (OpenAI uses ~4 chars per token).
    
    Args:
        text: Input text string
    
    Returns:
        Estimated token count
    """
    return len(text) // 4


def format_document_for_prompt(doc: Dict[str, Any], doc_type: str) -> str:
    """
    Format a single document for inclusion in the LLM prompt.
    
    Args:
        doc: Document dictionary with id, title, text, etc.
        doc_type: "LO" or content type
    
    Returns:
        Formatted string representation of the document
    """
    if doc_type == "LO":
        lo_text = doc.get("text") or doc.get("title") or ""
        return f"ID: {doc['id']}\nTitle: {doc['title']}\nContent:\n{lo_text}\n"
    else:
        full_text = doc.get("text") or ""
        return (
            f"ID: {doc['id']}\n"
            f"Type: {doc.get('type', 'content')}\n"
            f"Title: {doc.get('title', doc.get('lo_title', ''))}\n"
            f"Content:\n{full_text}\n"
        )


def chunk_documents_for_llm(
    documents: List[Dict[str, Any]],
    doc_type: str,
    max_context_tokens: int,
    query_tokens: int,
    prompt_overhead_tokens: int = 2000,
) -> List[List[Dict[str, Any]]]:
    """
    Split documents into batches that fit within the context window without truncation.

    Args:
        documents: List of document dictionaries.
        doc_type: "LO" or "content".
        max_context_tokens: Maximum tokens allowed per LLM call.
        query_tokens: Estimated tokens for the user query.
        prompt_overhead_tokens: Reserved tokens for instructions/system message.

    Returns:
        List of batches, where each batch is a list of document dicts.
    """
    available_tokens = max_context_tokens - query_tokens - prompt_overhead_tokens
    if available_tokens <= 0:
        raise ValueError(
            "Not enough tokens available for documents after reserving space for query and instructions."
        )

    batches: List[List[Dict[str, Any]]] = []
    current_batch: List[Dict[str, Any]] = []
    tokens_used = 0

    for doc in documents:
        doc_text = format_document_for_prompt(doc, doc_type)
        doc_tokens = count_tokens_approx(doc_text)

        if doc_tokens > available_tokens:
            raise ValueError(
                f"Document {doc.get('id')} of type {doc_type} exceeds available context window "
                "even when placed alone. Increase max_context_tokens or reduce document size."
            )

        if tokens_used + doc_tokens > available_tokens and current_batch:
            batches.append(current_batch)
            current_batch = []
            tokens_used = 0

        current_batch.append(doc)
        tokens_used += doc_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def build_selection_prompt(query: str, doc_block: str, doc_type: str, k: int) -> str:
    """
    Build a selection prompt for the LLM given a block of documents.

    Args:
        query: User query string.
        doc_block: Concatenated document text for the batch.
        doc_type: "LO" or "content".
        k: Number of items to select.

    Returns:
        Formatted prompt string.
    """
    if doc_type == "LO":
        label = "Learning Objectives"
        example_id = '"lo_123"'
    else:
        label = "Content Items"
        example_id = '"content_123_example"'

    return f"""Given this user query: "{query}"

Select the {k} most relevant {label} from the list below. Consider semantic relevance, topic alignment,
and educational value. Return ONLY a JSON object with this exact format:
{{"selected_ids": [{example_id}, ...]}}

The selected_ids array should contain exactly {k} document IDs (or all available IDs if fewer documents are provided)
in order of relevance (most relevant first).

{label}:
{doc_block}

Query: "{query}"

Return JSON object:"""


def call_llm_selection(
    client: OpenAI,
    model: str,
    prompt: str,
    doc_type: str,
) -> List[str]:
    """
    Call the LLM to select document IDs from a prompt.

    Args:
        client: OpenAI client instance.
        model: Model name.
        prompt: User prompt containing documents and instructions.
        doc_type: "LO" or "content" for logging.

    Returns:
        List of selected document IDs.
    """
    system_prompt = (
        "You are a precise document retrieval system for educational content. "
        "Return ONLY valid JSON objects with a 'selected_ids' key containing an array of document IDs. "
        "Do not include any explanation or additional text."
    )

    create_params: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    if not model.startswith("gpt-5"):
        create_params["temperature"] = 0.0

    try:
        response = client.chat.completions.create(**create_params)
        result = json.loads(response.choices[0].message.content)
        selected_ids = result.get("selected_ids", [])
        if isinstance(selected_ids, list):
            return selected_ids
    except Exception as exc:
        print(f"  Error during {doc_type} selection call: {exc}")
        import traceback

        traceback.print_exc()

    return []


def multi_batch_selection(
    query: str,
    documents: List[Dict[str, Any]],
    doc_type: str,
    k: int,
    client: OpenAI,
    model: str,
    max_context_tokens: int,
) -> List[str]:
    """
    Run selection across multiple batches to avoid truncating document text.

    Args:
        query: User query string.
        documents: Document corpus for the given type.
        doc_type: "LO" or "content".
        k: Number of top documents to return.
        client: OpenAI client.
        model: Model name.
        max_context_tokens: Maximum tokens per LLM call.

    Returns:
        List of top-k document IDs.
    """
    query_tokens = count_tokens_approx(query)
    batches = chunk_documents_for_llm(
        documents,
        doc_type,
        max_context_tokens=max_context_tokens,
        query_tokens=query_tokens,
    )

    print(f"  {doc_type.upper()} selection will run across {len(batches)} batch(es).")

    id_scores: Dict[str, float] = {}
    first_seen: Dict[str, Any] = {}

    for batch_index, batch_docs in enumerate(batches, 1):
        doc_texts = [format_document_for_prompt(doc, doc_type) for doc in batch_docs]
        doc_block = "\n".join(doc_texts)
        batch_k = min(k, len(batch_docs))
        prompt = build_selection_prompt(query, doc_block, doc_type, batch_k)
        selected_ids = call_llm_selection(client, model, prompt, doc_type)

        valid_ids = {doc["id"] for doc in batch_docs}
        for rank, doc_id in enumerate(selected_ids[:batch_k], 1):
            if doc_id not in valid_ids:
                continue
            weight = batch_k - rank + 1
            id_scores[doc_id] = id_scores.get(doc_id, 0.0) + weight
            if doc_id not in first_seen:
                first_seen[doc_id] = (batch_index, rank)

    if not id_scores:
        return []

    sorted_ids = sorted(
        id_scores.items(),
        key=lambda item: (-item[1], first_seen.get(item[0], (float("inf"), float("inf")))),
    )

    return [doc_id for doc_id, _ in sorted_ids[:k]]


def llm_full_context_retriever(
    query: str,
    lo_corpus: List[Dict[str, Any]],
    content_corpus: List[Dict[str, Any]],
    client: OpenAI,
    model: str = "gpt-4o",  # GPT-4o for best quality
    k_los: int = 5,
    k_content: int = 5,
    max_context_tokens: int = 120000,  # GPT-4o has 128K, leave margin
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Use high-quality LLM with full document context to retrieve relevant documents.
    
    This method puts all documents (or as many as fit) in the context window
    and asks the LLM to select the most relevant ones. Uses full text, not summaries.
    
    Args:
        query: Search query string
        lo_corpus: List of LO document dicts with keys: id, title, text (or base_text)
        content_corpus: List of content document dicts with keys: id, type, title, text
        client: OpenAI client instance
        model: Model to use (gpt-5 recommended for best quality, gpt-4o as fallback, claude-3-5-sonnet-20241022 for Anthropic)
        k_los: Number of top LOs to return
        k_content: Number of top content items to return
        max_context_tokens: Maximum tokens to use for context (leave margin for response)
    
    Returns:
        dict: {"los": List[dict], "content": List[dict]} with hit dictionaries
    """
    # Retrieve LOs using multi-batch strategy
    lo_selected_ids = multi_batch_selection(
        query=query,
        documents=lo_corpus,
        doc_type="LO",
        k=k_los,
        client=client,
        model=model,
        max_context_tokens=max_context_tokens,
    )
    
    # Retrieve content using multi-batch strategy
    content_selected_ids = multi_batch_selection(
        query=query,
        documents=content_corpus,
        doc_type="content",
        k=k_content,
        client=client,
        model=model,
        max_context_tokens=max_context_tokens,
    )
    
    # Build lookup maps
    lo_map = {doc["id"]: doc for doc in lo_corpus}
    content_map = {doc["id"]: doc for doc in content_corpus}
    
    # Format LO results
    lo_hits = []
    for rank, doc_id in enumerate(lo_selected_ids[:k_los], 1):
        if doc_id in lo_map:
            doc = lo_map[doc_id]
            lo_hits.append({
                "rank": rank,
                "score": 1.0 - (rank - 1) * 0.1,  \
                "doc_id": doc["id"],
                "lo_id": doc.get("lo_id", ""),
                "title": doc["title"],
                "snippet": doc.get("text", doc.get("base_text", ""))[:200],
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
                "type": doc.get("type", ""),
                "lo_title": doc.get("lo_title", ""),
                "title": doc.get("title", doc.get("lo_title", "")),
                "snippet": doc.get("text", "")[:200],
            })
    
    return {"los": lo_hits, "content": ct_hits}


def build_corpus_for_llm(los_df: pd.DataFrame, content_df: pd.DataFrame) -> tuple:
    """
    Build corpus dictionaries optimized for LLM full-context retrieval.
    
    Args:
        los_df: DataFrame with LO metadata
        content_df: DataFrame with content metadata
    
    Returns:
        tuple: (lo_corpus, content_corpus) as lists of dicts
    """
    lo_corpus = []
    for _, row in los_df.iterrows():
        lo_corpus.append({
            "id": f"lo_{int(row.lo_id)}",
            "lo_id": int(row.lo_id),
            "title": row.learning_objective,
            "text": row.learning_objective,  # For LOs, title is the text
            "base_text": row.learning_objective,
        })
    
    content_corpus = []
    lo_title_map = dict(los_df[["lo_id", "learning_objective"]].itertuples(index=False, name=None))
    
    for _, row in content_df.iterrows():
        parent_lo = int(row.lo_id_parent) if pd.notna(row.lo_id_parent) else None
        lo_title = lo_title_map.get(parent_lo, "") if parent_lo is not None else ""
        
        full_text = str(row.text or "")
        content_corpus.append({
            "id": row.content_id,
            "type": row.content_type,
            "lo_id_parent": parent_lo,
            "lo_title": lo_title,
            "title": lo_title,
            "text": full_text,
            "base_text": full_text,
        })
    
    return lo_corpus, content_corpus


def evaluate_llm_full_context(lo_corpus, content_corpus, test_queries, output_path=None, model="gpt-4-turbo-preview"):
    """
    Evaluate LLM full-context retrieval on test queries.
    
    Args:
        lo_corpus: List of LO document dicts
        content_corpus: List of content document dicts
        test_queries: List of query strings
        output_path: Optional path to save CSV
        model: Model to use (gpt-5 recommended for best quality, gpt-4o as fallback, claude-3-5-sonnet-20241022 for Anthropic)
    
    Returns:
        pd.DataFrame: Results DataFrame
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    results = []
    
    print(f"\nEvaluating LLM full-context retrieval ({model}) on {len(test_queries)} queries...")
    for intent_id, query in enumerate(test_queries, 1):
        print(f"\nQuery {intent_id}/{len(test_queries)}: {query[:60]}...")
        
        hits = llm_full_context_retriever(
            query, lo_corpus, content_corpus, client, model=model, k_los=5, k_content=5
        )
        
        # Format results
        for r in hits["los"]:
            results.append({
                "intent_id": intent_id,
                "query": query,
                "method": f"llm_full_context_{model}",
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
                "method": f"llm_full_context_{model}",
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
    
    if output_path is None:
        output_path = RESULTS_DIR / f"retriever_eval_llm_full_context_{model.replace('-', '_')}.csv"
    
    results_df.to_csv(output_path, index=False)
    print(f"\nExported {len(results_df)} result rows to {output_path}")
    print("\nNext steps:")
    print("1. Open the CSV and manually rate relevance_score (1-5) for each result")
    print("2. Calculate aggregate metrics: average relevance, coverage, etc.")
    
    return results_df


def main():
    """Main execution function."""
    print("=" * 60)
    print("LLM Full-Context Retrieval")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    los_df, content_df = load_data()
    
    # Build corpus
    print("\n2. Building corpus for LLM retrieval...")
    lo_corpus, content_corpus = build_corpus_for_llm(los_df, content_df)
    print(f"  Built corpus: {len(lo_corpus)} LOs, {len(content_corpus)} content items")
    
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
    ]
    
    print(f"\n3. Testing with {len(test_queries)} queries...")
    print("   Note: This will be slow and expensive due to large context windows")
    print("   Using GPT-5 (128K context window)")
    
    # Evaluate
    results_df = evaluate_llm_full_context(
        lo_corpus,
        content_corpus,
        test_queries,
        model="gpt-5"  # GPT-5 is available
    )
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    try:
        main()
        print("\nScript completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

