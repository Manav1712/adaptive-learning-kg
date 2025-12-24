# Jupyter Notebook Pipeline Specification (Single-Cell) — Adaptive Learning Coach + Retriever + Planner + Tutor
## Build the final system by editing ONE notebook cell, step by step

---

## Overview

This spec is written so an engineer can start from the existing notebook cell in:

- `src/workflow_demo/coach 9th oct (2) (1).ipynb`

…paste that **single code cell** into a new Jupyter notebook cell, and then apply the ordered edits below to arrive at the **final system**:

- Coach maintains **plan state** and evaluates every new student message **with that plan in mind**
- Retrieval uses **offline-precomputed corpus embeddings** (runtime only embeds the query and does cosine similarity)
- Image handling is a special case of retrieval: **OCR text + image embeddings in parallel**, merge candidates
- Planner returns the **simplified plan JSON** (current_plan + future_plan)
- Tutor follows the plan **without confirmation**, stays on topic, and hands off to coach on off-topic requests

**Assumption (important):** all preprocessing is already complete before running this notebook. The KG CSVs, LO embeddings, and image embeddings exist on disk. This spec covers **runtime behavior only** — loading artifacts, embedding queries, retrieval, planning, and tutoring.

**Constraint:** everything must remain in **one Jupyter cell** (for now).

---

## Step 0 — Baseline (Load the existing notebook cell)

**Input:** the first code cell from `src/workflow_demo/coach 9th oct (2) (1).ipynb`  
**Process:** copy the code from the first cell into a single Jupyter cell and run it once to confirm the baseline works  
**Output:** a working "Seamless Learning Assistant (Pure LLM Architecture)" loop

**Important (security):** the baseline cell contains a hard-coded OpenAI key. Before running, replace it with environment variable usage:

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

**Baseline section markers (so you know what to replace later):** the notebook cell has headings like:

- `# ---------- LLM Helper Functions ----------`
- `# ---------- Session Memory Structure ----------`
- `# ---------- OpenStax Learning Objectives & FAQ Topics ----------`
- `# ---------- Tutoring Planner Tool ----------`
- `# ---------- COACH AGENT ----------`
- `# ---------- TUTOR BOT ----------`
- `# ---------- FAQ BOT ----------`
- `# ---------- MAIN LOOP ----------`

In the steps below, we'll explicitly say which section to edit/replace.

---

## What we are changing (high-level)

We will upgrade the baseline to:

1. **File-based KG**: load LOs and prereq edges from CSVs at runtime
2. **Retriever**: use embeddings to retrieve Top‑K LO candidates (text + image)
3. **Planner**: generate a **simplified plan** JSON from candidates
4. **Tutor**: teach from **current_plan only**, no confirmation, off-plan → ask to end session and switch
5. **Coach**: keep `active_plan` and run a plan-guard check on every student message

---

## Runtime Data Files

You can keep these paths as-is inside the single cell (relative to project root):

```
demo/
  lo_index.csv
  edges_prereqs.csv

demo/runtime_artifacts/
  lo_embeddings.npy
  lo_row_index.csv

src/workflow_demo/image_corpus/
  image_metadata.csv
  image_embeddings.npy
  derivatives/...
```

If you use different paths, update the constants in Step 1.

---

## Dependencies (exact versions)

Install these packages before running. The versions below are tested and known to work together.

**Core runtime (required):**

```
openai==1.107.2
numpy==1.26.4
pandas==2.3.2
sentence-transformers==5.1.2
pillow==11.3.0
pydantic==2.11.9
python-dotenv==1.1.1
```

**To install:**

```bash
pip install openai==1.107.2 numpy==1.26.4 pandas==2.3.2 sentence-transformers==5.1.2 pillow==11.3.0 pydantic==2.11.9 python-dotenv==1.1.1
```

**Or use the project's `requirements.txt`:**

```bash
pip install -r requirements.txt
```

**What each package does:**

| Package | Purpose |
|---------|---------|
| `openai` | LLM API calls (GPT-5.1, GPT-4o, embeddings) |
| `numpy` | Embedding matrices, cosine similarity |
| `pandas` | Load KG CSVs (`lo_index.csv`, `edges_prereqs.csv`, `image_metadata.csv`) |
| `sentence-transformers` | CLIP model for image embeddings |
| `pillow` | Image loading for CLIP |
| `pydantic` | Data validation (optional but recommended) |
| `python-dotenv` | Load `OPENAI_API_KEY` from `.env` file |

**Environment variable:**

```bash
export OPENAI_API_KEY="sk-..."
```

Or create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

---

## Step 1 — Add configuration + imports (single cell)

**Input:** none  
**Process:** add constants + imports near the top of the cell (after `DEBUG_MODE`)  
**Output:** the rest of the cell can use consistent config

Add all imports at the top of the cell:

```python
import os
import json
import base64
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from PIL import Image

# Models
# - Use GPT‑5.1 for all text-only reasoning (coach routing, plan-guard, planner, tutor when no image)
# - Use GPT‑4o for any call that includes an image (OCR and tutor-with-image)
CHAT_MODEL = os.getenv("WORKFLOW_DEMO_LLM_MODEL", "gpt-5.1")
VISION_MODEL = os.getenv("WORKFLOW_DEMO_VISION_MODEL", "gpt-4o")
TEXT_EMBED_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "text-embedding-3-large")

# Retrieval params
TOP_K_TEXT = 6
TOP_K_IMAGE = 3
TOP_K_MERGED = 6

# Paths
REPO_ROOT = Path.cwd()
KG_DIR = REPO_ROOT / "demo"
ARTIFACT_DIR = KG_DIR / "runtime_artifacts"
IMAGE_CORPUS_DIR = REPO_ROOT / "src" / "workflow_demo" / "image_corpus"

# Retry (Coach-only LLM calls)
MAX_RETRIES = 3
RETRY_BASE_SECONDS = 1.0
RETRY_MAX_SECONDS = 12.0
```

### 1.1 Update the baseline model variables + helper to use `CHAT_MODEL`

**Input:** baseline variables `MODEL_NAME = "gpt-4o"` and helper `_chat_json(...)`  
**Process:** make all chat-completion calls use `CHAT_MODEL` consistently  
**Output:** you can switch models by env var without editing code

In the baseline cell:

- Replace:

```python
MODEL_NAME = "gpt-4o"
```

- With:

```python
MODEL_NAME = CHAT_MODEL
```

And keep `_chat_json` using `model=MODEL_NAME`.

### 1.2 Verify dependencies

Ensure all packages from the **Dependencies** section above are installed. If running in a fresh notebook environment, you can install inline:

```python
# Uncomment to install in-notebook (first run only)
# !pip install -q openai==1.107.2 numpy==1.26.4 pandas==2.3.2 sentence-transformers==5.1.2 pillow==11.3.0 pydantic==2.11.9 python-dotenv==1.1.1
```

---

## Step 2 — Define the simplified plan JSON contract (Planner output)

**Input:** none  
**Process:** add a strict JSON schema description and a validator function  
**Output:** planner output can be validated before tutoring starts

### 2.1 Contract (must match exactly)

Planner must output:

```json
{
  "subject": "calculus",
  "mode": "conceptual_review",
  "current_plan": [
    {
      "lo_id": 42,
      "title": "The Chain Rule",
      "proficiency": 0.35,
      "notes": "Student is new to this topic. Start with intuition about nested functions before introducing notation.",
      "is_primary": true,
      "how_to_teach": "Start with the intuition: when one function is inside another, the rate of change depends on both. Use the notation dy/dx = (dy/du)(du/dx). Walk through simple examples like (x^2 + 1)^3 before moving to trigonometric compositions.",
      "why_to_teach": "The chain rule is essential for differentiating composite functions, which appear constantly in real-world applications. Without it, students cannot handle nested expressions or implicit differentiation."
    },
    {
      "lo_id": 41,
      "title": "Differentiation Rules",
      "proficiency": 0.65,
      "notes": "Prerequisite — student has solid understanding, quick review only.",
      "is_primary": false,
      "how_to_teach": "Cover power rule, product rule, and quotient rule systematically. Emphasize pattern recognition and when to apply each rule.",
      "why_to_teach": "These foundational rules are prerequisites for the chain rule and all advanced differentiation techniques."
    }
  ],
  "future_plan": [
    {
      "lo_id": 43,
      "title": "Implicit Differentiation",
      "proficiency": 0.0,
      "notes": "Natural next step after mastering the chain rule.",
      "is_primary": false,
      "how_to_teach": "Show how to differentiate equations where y is not isolated. Emphasize the chain rule connection.",
      "why_to_teach": "Extends chain rule to equations that cannot be solved for y explicitly."
    }
  ],
  "book": "Calculus Volume 1",
  "unit": "Derivatives",
  "chapter": "Differentiation Rules"
}
```

**Type Constraints:**
- `subject`: string, must be one of `"calculus"`, `"algebra"`, `"trigonometry"`
- `mode`: string, must be one of `"conceptual_review"`, `"examples"`, `"practice"`
- `lo_id`: integer, must exist in `lo_index.csv`
- `proficiency`: float, range `[0.0, 1.0]` (0.0 = new/struggling, 1.0 = mastered)
- `notes`: string, can be empty, typically 10-100 characters
- `is_primary`: boolean, exactly one `true` in `current_plan`, always `false` in `future_plan`
- `how_to_teach`: string, non-empty, pedagogical guidance (typically 50-300 characters)
- `why_to_teach`: string, non-empty, rationale for teaching this LO (typically 50-200 characters)
- `book`, `unit`, `chapter`: strings, non-empty, from the primary LO's metadata

**Rules**
- `current_plan`: length 1–3
- exactly 1 item in `current_plan` has `is_primary=true`
- `future_plan`: exactly 1 item
- all items use the **same** `mode` (the mode is the session mode)

### 2.2 Validator (add to cell)

```python
def validate_plan(plan: dict) -> None:
    """Validate simplified plan JSON contract.
    
    Raises ValueError with descriptive message if validation fails.
    """
    if not isinstance(plan, dict):
        raise ValueError("plan must be a dict")
    
    # Check required top-level keys
    required_keys = ["subject", "mode", "current_plan", "future_plan", "book", "unit", "chapter"]
    for k in required_keys:
        if k not in plan:
            raise ValueError(f"plan missing key: {k}")
    
    # Validate subject and mode are from allowed sets
    allowed_subjects = {"calculus", "algebra", "trigonometry"}
    if plan["subject"] not in allowed_subjects:
        raise ValueError(f"subject must be one of {allowed_subjects}, got: {plan['subject']}")
    
    allowed_modes = {"conceptual_review", "examples", "practice"}
    if plan["mode"] not in allowed_modes:
        raise ValueError(f"mode must be one of {allowed_modes}, got: {plan['mode']}")
    
    # Validate current_plan length
    if not isinstance(plan["current_plan"], list):
        raise ValueError("current_plan must be a list")
    if not (1 <= len(plan["current_plan"]) <= 3):
        raise ValueError(f"current_plan must have 1-3 items, got {len(plan['current_plan'])}")
    
    # Validate future_plan length
    if not isinstance(plan["future_plan"], list):
        raise ValueError("future_plan must be a list")
    if len(plan["future_plan"]) != 1:
        raise ValueError(f"future_plan must have exactly 1 item, got {len(plan['future_plan'])}")
    
    # Validate each LO item
    required_lo_fields = ["lo_id", "title", "proficiency", "notes", "is_primary", "how_to_teach", "why_to_teach"]
    all_items = plan["current_plan"] + plan["future_plan"]
    
    for i, item in enumerate(all_items):
        if not isinstance(item, dict):
            raise ValueError(f"Plan item {i} must be a dict")
        for field in required_lo_fields:
            if field not in item:
                raise ValueError(f"Plan item {i} missing required field: {field}")
        
        # Validate types
        if not isinstance(item["lo_id"], int):
            raise ValueError(f"Plan item {i}: lo_id must be int, got {type(item['lo_id'])}")
        if not isinstance(item["proficiency"], (int, float)) or not (0.0 <= item["proficiency"] <= 1.0):
            raise ValueError(f"Plan item {i}: proficiency must be float in [0.0, 1.0], got {item['proficiency']}")
        if not isinstance(item["is_primary"], bool):
            raise ValueError(f"Plan item {i}: is_primary must be bool, got {type(item['is_primary'])}")
    
    # Check exactly one primary in current_plan
    primaries = [x for x in plan["current_plan"] if x.get("is_primary") is True]
    if len(primaries) != 1:
        raise ValueError(f"current_plan must contain exactly one primary LO (is_primary=true), found {len(primaries)}")
    
    # Check all items use same mode (rule: all items use the same mode)
    # Note: mode is set at top-level, but we verify no conflicting mode fields in items
    for i, item in enumerate(all_items):
        if "mode" in item and item["mode"] != plan["mode"]:
            raise ValueError(f"Plan item {i} has mode '{item['mode']}' but top-level mode is '{plan['mode']}'. All items must use the same mode.")
```

---

## Step 3 — Load the knowledge graph (CSV → lookups)

**Input:** `demo/lo_index.csv`, `demo/edges_prereqs.csv`  
**Process:** load both files and build fast lookups  
**Output:** `lo_by_id` and `prereqs_by_lo`

### 3.1 KG fields used by downstream components

From `lo_index.csv`:
- `lo_id` (int)
- `learning_objective` (string) → becomes `title`
- `book`, `unit`, `chapter` (strings)
- *(recommended)* `how_to_teach`, `why_to_teach` (strings)

From `edges_prereqs.csv`:
- `source_lo_id` (prereq)
- `target_lo_id` (dependent)

### 3.2 CSV Schema

**`lo_index.csv` format:**

Required columns:
- `lo_id` (integer): Unique learning objective identifier
- `learning_objective` (string, non-empty): Full title of the LO

Optional columns:
- `book` (string): e.g., "Calculus Volume 1"
- `unit` (string): e.g., "Derivatives"
- `chapter` (string): e.g., "Differentiation Rules"
- `how_to_teach` (string): Pedagogical guidance for teaching this LO
- `why_to_teach` (string): Rationale for why this LO is important

**File encoding:** UTF-8

Example row:
```csv
lo_id,learning_objective,book,unit,chapter,how_to_teach,why_to_teach
42,"The Chain Rule","Calculus Volume 1","Derivatives","Differentiation Rules","Start with the intuition: when one function is inside another, the rate of change depends on both.","The chain rule is essential for differentiating composite functions."
```

**`edges_prereqs.csv` format:**

Required columns:
- `source_lo_id` (integer): Prerequisite LO ID
- `target_lo_id` (integer): Dependent LO ID (the LO that requires the prerequisite)

**File encoding:** UTF-8

Example row:
```csv
source_lo_id,target_lo_id
41,42
40,42
```

This means: LO 41 and LO 40 are prerequisites for LO 42.

**Note:** Both files must exist in `demo/` directory (as specified in Step 1).

### 3.3 Loader (add to cell)

**When to call:** Call `load_kg()` once at startup (before Step 5 retrieval or Step 6 planning).

**Usage:** The returned `lo_by_id` and `prereqs_by_lo` are used in:
- Step 5: `build_candidate()` function to add KG fields (book, unit, chapter, how_to_teach, why_to_teach, prereq_lo_ids) to retrieval candidates
- Step 6: Planner uses these fields when generating the simplified plan

```python
def load_kg() -> tuple[dict[int, dict], dict[int, list[int]]]:
    """Load knowledge graph from CSVs and build lookup dictionaries.
    
    Returns:
        tuple: (lo_by_id, prereqs_by_lo)
            - lo_by_id: dict mapping lo_id -> LO metadata dict with keys:
              lo_id, title, book, unit, chapter, how_to_teach, why_to_teach
            - prereqs_by_lo: dict mapping lo_id -> list of prerequisite lo_ids
    
    Raises:
        FileNotFoundError: If CSV files are missing
        ValueError: If CSV format is invalid, required columns missing, or data integrity checks fail
    
    Edge cases:
    - If file not found, pd.read_csv() raises FileNotFoundError
    - If file is empty, pd.read_csv() raises pd.errors.EmptyDataError
    - If required columns missing, validation will raise ValueError
    """
    # Load CSVs
    lo_df = pd.read_csv(KG_DIR / "lo_index.csv")
    prereq_df = pd.read_csv(KG_DIR / "edges_prereqs.csv")
    
    # Validate required columns
    required_lo_cols = ["lo_id", "learning_objective"]
    missing = [c for c in required_lo_cols if c not in lo_df.columns]
    if missing:
        raise ValueError(f"lo_index.csv missing required columns: {missing}")
    
    required_edge_cols = ["source_lo_id", "target_lo_id"]
    missing = [c for c in required_edge_cols if c not in prereq_df.columns]
    if missing:
        raise ValueError(f"edges_prereqs.csv missing required columns: {missing}")
    
    # Build lo_by_id lookup with validation
    lo_by_id: dict[int, dict] = {}
    for _, row in lo_df.iterrows():
        # Validate and convert lo_id
        try:
            lo_id = int(row["lo_id"])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid lo_id (must be integer): {row['lo_id']}")
        
        # Check for duplicates
        if lo_id in lo_by_id:
            raise ValueError(f"Duplicate lo_id in lo_index.csv: {lo_id}")
        
        # Validate learning_objective (required, non-empty)
        learning_obj = str(row["learning_objective"]).strip()
        if not learning_obj:
            raise ValueError(f"Empty learning_objective for lo_id {lo_id}")
        
        lo_by_id[lo_id] = {
            "lo_id": lo_id,
            "title": learning_obj,
            "book": str(row.get("book", "")).strip(),
            "unit": str(row.get("unit", "")).strip(),
            "chapter": str(row.get("chapter", "")).strip(),
            "how_to_teach": str(row.get("how_to_teach", "")).strip(),
            "why_to_teach": str(row.get("why_to_teach", "")).strip(),
        }
    
    # Build prereqs_by_lo lookup with validation
    prereqs_by_lo: dict[int, list[int]] = {}
    valid_lo_ids = set(lo_by_id.keys())
    
    for _, row in prereq_df.iterrows():
        # Validate and convert IDs
        try:
            src = int(row["source_lo_id"])
            tgt = int(row["target_lo_id"])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid lo_id in edges_prereqs.csv: source={row.get('source_lo_id')}, target={row.get('target_lo_id')}")
        
        # Validate references exist
        if src not in valid_lo_ids:
            raise ValueError(f"edges_prereqs.csv references non-existent source_lo_id: {src}")
        if tgt not in valid_lo_ids:
            raise ValueError(f"edges_prereqs.csv references non-existent target_lo_id: {tgt}")
        
        # Prevent self-loops
        if src == tgt:
            raise ValueError(f"edges_prereqs.csv contains self-loop: lo_id {src}")
        
        prereqs_by_lo.setdefault(tgt, []).append(src)
    
    return lo_by_id, prereqs_by_lo
```

**Concrete output examples**

**Example 1: `lo_by_id` lookup**

Input: `lo_by_id[42]`  
Output:

```json
{
  "lo_id": 42,
  "title": "The Chain Rule",
  "book": "Calculus Volume 1",
  "unit": "Derivatives",
  "chapter": "Differentiation Rules",
  "how_to_teach": "Start with the intuition: when one function is inside another, the rate of change depends on both. Use the notation dy/dx = (dy/du)(du/dx). Walk through simple examples like (x^2 + 1)^3 before moving to trigonometric compositions.",
  "why_to_teach": "The chain rule is essential for differentiating composite functions, which appear constantly in real-world applications. Without it, students cannot handle nested expressions or implicit differentiation."
}
```

**Example 2: `prereqs_by_lo` lookup**

Input: `prereqs_by_lo[42]`  
Output: `[41, 40]`  // List of prerequisite LO IDs for LO 42

Input: `prereqs_by_lo[10]`  
Output: `[]`  // Empty list if LO 10 has no prerequisites

**Data type notes:**
- `lo_id` values are integers (not strings)
- Empty optional fields (book, unit, chapter, how_to_teach, why_to_teach) are stored as empty strings `""`
- `prereqs_by_lo` values are lists of integers (prerequisite lo_ids)
- If an LO has no prerequisites, the list is empty `[]`

---

## Step 4 — Load offline embedding artifacts (runtime only loads)

**Input:** embedding `.npy` matrices + row index CSVs  
**Process:** load into memory at startup  
**Output:** in-memory matrices for cosine similarity search

**When to call:** Call `load_text_artifacts()` and `load_image_artifacts()` once at startup (before Step 5 retrieval).

**Usage:** The returned embeddings and metadata are used in Step 5 for cosine similarity search. These artifacts are precomputed offline (as stated in Overview) and loaded into memory at runtime.

**Memory note:** These matrices are loaded entirely into RAM. For a typical KG with ~150 LOs and ~300 images, expect ~2-3 MB total. For larger KGs, scale accordingly.

### 4.1 Expected artifact formats

**Text artifacts:**

`lo_embeddings.npy`:
- Shape: `(num_los, dim)` where `dim` is the embedding dimension (3072 for `text-embedding-3-large`)
- Dtype: `float32`
- Normalization: Each row must be L2-normalized (unit vector, norm = 1.0)
- Row order: Must match `lo_row_index.csv` row order (row 0 → first LO, row 1 → second LO, etc.)

`lo_row_index.csv`:
- Columns: `row_index` (int, required), `lo_id` (int, required)
- Encoding: UTF-8
- Must be sorted by `row_index` in ascending order (0, 1, 2, ...)
- Each `lo_id` must exist in `lo_index.csv` (loaded in Step 3)
- Example:
  ```csv
  row_index,lo_id
  0,42
  1,41
  2,40
  ```

**Image artifacts:**

`image_embeddings.npy`:
- Shape: `(num_images, 512)` where 512 is the CLIP embedding dimension (fixed for `clip-ViT-B-32`)
- Dtype: `float32`
- Normalization: Each row must be L2-normalized (unit vector, norm = 1.0)
- Row order: Must match `image_metadata.csv` row order

`image_metadata.csv`:
- Required columns: `image_id` (int or string), `path` (string), `lo_id` (int)
- Optional columns: `description` (string), `keywords` (string)
- Encoding: UTF-8
- Each `lo_id` must exist in `lo_index.csv` (loaded in Step 3)
- Example:
  ```csv
  image_id,path,lo_id,description,keywords
  1,"derivatives/chain_rule_example.png",42,"Visual example of chain rule","chain rule,derivative,composite"
  ```

### 4.2 Loader (add to cell)

```python
def load_text_artifacts() -> tuple[np.ndarray, list[int]]:
    """Load text embedding artifacts.
    
    Returns:
        tuple: (lo_embeddings, lo_ids_by_row)
            - lo_embeddings: numpy array shape (num_los, dim), float32, L2-normalized rows
            - lo_ids_by_row: list of lo_ids matching embedding row order
    
    Raises:
        FileNotFoundError: If artifact files are missing
        ValueError: If files are invalid or don't match expected format
    
    Edge cases:
    - If file not found, np.load() raises FileNotFoundError
    - If file is corrupted, np.load() raises ValueError
    - If file is empty or has wrong shape, validation will raise ValueError
    """
    # Load embeddings
    emb = np.load(ARTIFACT_DIR / "lo_embeddings.npy").astype("float32")
    
    # Validate shape
    if len(emb.shape) != 2:
        raise ValueError(f"lo_embeddings.npy must be 2D array, got shape {emb.shape}")
    
    # Verify L2-normalization
    row_norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(row_norms, 1.0, atol=1e-5):
        raise ValueError("lo_embeddings.npy rows are not L2-normalized (expected norm = 1.0)")
    
    # Load row index
    rows = pd.read_csv(ARTIFACT_DIR / "lo_row_index.csv")
    
    # Validate required columns
    required_cols = ["row_index", "lo_id"]
    missing = [c for c in required_cols if c not in rows.columns]
    if missing:
        raise ValueError(f"lo_row_index.csv missing required columns: {missing}")
    
    # Sort and extract lo_ids
    rows = rows.sort_values("row_index")
    lo_ids_by_row = [int(x) for x in rows["lo_id"].tolist()]
    
    # Validate row count matches
    if emb.shape[0] != len(lo_ids_by_row):
        raise ValueError(f"lo_embeddings.npy has {emb.shape[0]} rows but lo_row_index.csv has {len(lo_ids_by_row)} rows")
    
    return emb, lo_ids_by_row


def load_image_artifacts() -> tuple[np.ndarray, pd.DataFrame]:
    """Load image embedding artifacts.
    
    Returns:
        tuple: (image_embeddings, image_metadata)
            - image_embeddings: numpy array shape (num_images, 512), float32, L2-normalized rows
            - image_metadata: DataFrame with columns image_id, path, lo_id, (optional) description, keywords
    
    Raises:
        FileNotFoundError: If artifact files are missing
        ValueError: If files are invalid or don't match expected format
    
    Edge cases:
    - If file not found, np.load() raises FileNotFoundError
    - If file is corrupted, np.load() raises ValueError
    - If file is empty or has wrong shape, validation will raise ValueError
    """
    # Load embeddings
    emb = np.load(IMAGE_CORPUS_DIR / "image_embeddings.npy").astype("float32")
    
    # Validate shape
    if len(emb.shape) != 2:
        raise ValueError(f"image_embeddings.npy must be 2D array, got shape {emb.shape}")
    if emb.shape[1] != 512:
        raise ValueError(f"image_embeddings.npy must have 512 columns (CLIP dimension), got {emb.shape[1]}")
    
    # Verify L2-normalization
    row_norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(row_norms, 1.0, atol=1e-5):
        raise ValueError("image_embeddings.npy rows are not L2-normalized (expected norm = 1.0)")
    
    # Load metadata
    meta = pd.read_csv(IMAGE_CORPUS_DIR / "image_metadata.csv")
    
    # Validate required columns
    required_cols = ["image_id", "path", "lo_id"]
    missing = [c for c in required_cols if c not in meta.columns]
    if missing:
        raise ValueError(f"image_metadata.csv missing required columns: {missing}")
    
    # Validate row count matches
    if emb.shape[0] != len(meta):
        raise ValueError(f"image_embeddings.npy has {emb.shape[0]} rows but image_metadata.csv has {len(meta)} rows")
    
    return emb, meta
```

**Concrete output examples**

**Example 1: `load_text_artifacts()`**

Input: Files `demo/runtime_artifacts/lo_embeddings.npy` (shape: 138, 3072) and `demo/runtime_artifacts/lo_row_index.csv`  
Output:
- `lo_embeddings`: numpy array shape `(138, 3072)`, dtype `float32`, L2-normalized rows
- `lo_ids_by_row`: `[42, 41, 40, 39, 38, 37, 36, ...]`  // List of 138 lo_ids matching embedding row order

**Example 2: `load_image_artifacts()`**

Input: Files `src/workflow_demo/image_corpus/image_embeddings.npy` (shape: 270, 512) and `src/workflow_demo/image_corpus/image_metadata.csv`  
Output:
- `image_embeddings`: numpy array shape `(270, 512)`, dtype `float32`, L2-normalized rows
- `image_metadata`: DataFrame with 270 rows:
  ```
  image_id,path,lo_id,description,keywords
  1,derivatives/chain_rule_example.png,42,Visual example of chain rule,chain rule,derivative
  2,derivatives/product_rule.png,41,Product rule diagram,product rule,derivative
  ...
  ```

---

## Step 5 — Runtime retrieval (text + image) with cosine similarity

**Input:** `student_text`, optional `ocr_text`, optional `image_path`  
**Process:** retrieve Top‑K candidates from text embeddings and image embeddings in parallel; merge  
**Output:** `merged_candidates` to feed the planner

**When to call:** Call `retrieve_candidates()` from the coach when a new tutoring request is received (after OCR if image provided).

**Usage:** The returned merged candidates are passed to the planner (Step 6) to generate the simplified plan. The function handles both text-only queries and image queries (with optional OCR text).

### 5.1 Cosine similarity (exact)

If vectors are L2-normalized, cosine similarity is the dot product:

\[
\text{score} = q \cdot v
\]

So:

```python
scores = lo_embeddings @ q_vec
top_idx = np.argsort(scores)[::-1][:TOP_K_TEXT]
```

### 5.2 Text query embedding (runtime)

Add:

```python
def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype("float32")

def embed_text_query(text: str) -> np.ndarray:
    """Embed text query using OpenAI embeddings API.
    
    Edge cases:
    - If text is empty, raises ValueError
    - If API call fails, raises RuntimeError (handle at coach level)
    """
    if not text.strip():
        raise ValueError("Query text cannot be empty")
    resp = client.embeddings.create(model=TEXT_EMBED_MODEL, input=text)
    vec = np.array(resp.data[0].embedding, dtype="float32")
    return l2_normalize(vec)
```

### 5.3 Image query embedding (runtime, CLIP)

Initialize the CLIP model once at startup (add after imports in Step 1):

```python
CLIP_MODEL_NAME = "clip-ViT-B-32"
clip_model = SentenceTransformer(CLIP_MODEL_NAME)
```

Then add the embedding function (add to retrieval section):

```python
def embed_image_query(image_path: str) -> np.ndarray:
    """Embed image query using CLIP model.
    
    Edge cases:
    - If image file not found, Image.open() raises FileNotFoundError
    - If image is corrupted, Image.open() raises PIL.UnidentifiedImageError
    - Handle these at coach level
    """
    img = Image.open(image_path).convert("RGB")
    vec = clip_model.encode([img], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec.astype("float32")
```

### 5.4 Candidate building + merge (add to cell)

```python
def build_candidate(lo_id: int, score: float, source: str, lo_by_id: dict, prereqs_by_lo: dict) -> dict:
    lo = lo_by_id[lo_id]
    return {
        "lo_id": lo_id,
        "title": lo["title"],
        "score": float(score),
        "source": source,
        "book": lo.get("book"),
        "unit": lo.get("unit"),
        "chapter": lo.get("chapter"),
        "how_to_teach": lo.get("how_to_teach") or "",
        "why_to_teach": lo.get("why_to_teach") or "",
        "prereq_lo_ids": [int(x) for x in prereqs_by_lo.get(lo_id, [])],
    }


def merge_candidates(text_cands: list, image_cands: list) -> list:
    best = {}
    for c in (text_cands + image_cands):
        lo_id = int(c["lo_id"])
        if lo_id not in best or c["score"] > best[lo_id]["score"]:
            best[lo_id] = c
    merged = sorted(best.values(), key=lambda x: -x["score"])
    for c in merged:
        c["source"] = "merged"
    return merged[:TOP_K_MERGED]


def retrieve_candidates(
    student_text: str,
    ocr_text: str | None,
    image_path: str | None,
    lo_embeddings: np.ndarray,
    lo_ids_by_row: list[int],
    lo_by_id: dict[int, dict],
    prereqs_by_lo: dict[int, list[int]],
    image_embeddings: np.ndarray,
    image_meta: pd.DataFrame
) -> list[dict]:
    query = "\\n".join([x for x in [student_text, ocr_text] if x]).strip()
    
    if not query and not image_path:
        raise ValueError("At least one of student_text, ocr_text, or image_path must be provided")

    # Text retrieval
    text_cands = []
    if query:
        q = embed_text_query(query)
        scores = lo_embeddings @ q
        top_idx = np.argsort(scores)[::-1][:TOP_K_TEXT]
        text_cands = [
            build_candidate(lo_ids_by_row[i], float(scores[i]), "text", lo_by_id, prereqs_by_lo)
            for i in top_idx
            if lo_ids_by_row[i] in lo_by_id
        ]

    # Image retrieval (optional)
    image_cands = []
    if image_path:
        q_img = embed_image_query(image_path)
        s_img = image_embeddings @ q_img
        top_i = np.argsort(s_img)[::-1][:TOP_K_IMAGE]
        for i in top_i:
            lo_id = int(image_meta.iloc[i]["lo_id"])
            if lo_id in lo_by_id:
                image_cands.append(build_candidate(lo_id, float(s_img[i]), "image", lo_by_id, prereqs_by_lo))

    merged = merge_candidates(text_cands, image_cands)
    
    if not merged:
        raise ValueError("No candidates found. Check that lo_by_id contains valid LO entries.")

    # Debug output (DEBUG_MODE should be defined in Step 1 or baseline)
    if globals().get("DEBUG_MODE", False):
        print("\\n=== RETRIEVAL DEBUG ===")
        print("Query:", query[:120])
        print("Text Top:")
        for c in text_cands[:TOP_K_TEXT]:
            print(f"  [{c['score']:.3f}] {c['title']} (LO {c['lo_id']})")
        if image_cands:
            print("Image Top:")
            for c in image_cands[:TOP_K_IMAGE]:
                print(f"  [{c['score']:.3f}] {c['title']} (LO {c['lo_id']})")
        print("Merged Top:")
        for c in merged:
            print(f"  [{c['score']:.3f}] {c['title']} (LO {c['lo_id']})")
        print("========================\\n")

    return merged
```

### 5.5 Exactly what gets passed into the Planner (KG context + scores)

Planner candidates must include:
- retrieval signal: `score`, `source`
- identity: `lo_id`, `title`
- KG context: `book`, `unit`, `chapter`, `prereq_lo_ids`
- teaching guidance: `how_to_teach`, `why_to_teach`
- adaptivity: `proficiency` + `suggested_notes` (added in Step 6)

---

## Step 6 — Planner LLM (merged candidates → simplified plan JSON)

**Input:** `student_request`, `mode`, merged candidates, proficiency map  
**Process:** call LLM with strict prompt; validate plan JSON  
**Output:** `active_plan` stored in coach state

**When to call:** Call `planner_llm()` from the coach after retrieving candidates (Step 5) and before starting the tutor session.

**Usage:** The returned plan is stored in coach state as `active_plan` and passed to the tutor (Step 7). The planner enriches candidates with proficiency scores and generates a structured plan matching the simplified plan schema (Step 2).

**Retry logic:** The planner uses `_chat_json()` which does not have retry logic (retry logic is coach-only per Step 1). If the LLM call fails, the function will raise an exception. The coach should handle retries at a higher level if needed.

### 6.1 Proficiency + notes (coach-owned)

**Note:** The `lo_mastery` dict format:
- Keys: `lo_id` as strings (e.g., `"42"`, `"41"`)
- Values: Proficiency scores as floats in range `[0.0, 1.0]`
  - `0.0` = new/struggling (no prior knowledge)
  - `1.0` = mastered (complete understanding)
- Example: `{"42": 0.35, "41": 0.65, "40": 0.0}`

Add:

```python
def proficiency_note(p: float) -> str:
    """Generate teaching guidance note based on proficiency score.
    
    Args:
        p: Proficiency score in range [0.0, 1.0]
    
    Returns:
        Short teaching guidance string for the planner to use in notes field
    """
    if p >= 0.85:
        return "High mastery — move quickly, focus on nuances."
    if p >= 0.65:
        return "Solid understanding — emphasize applications."
    if p >= 0.40:
        return "Developing — include a worked example and a quick check."
    return "New/struggling — start from fundamentals and go step-by-step."
```

### 6.2 Planner input payload (exact)

```python
def build_planner_input(student_request: str, mode: str, merged: list[dict], lo_mastery: dict[str, float]) -> dict:
    """Build planner input payload with enriched candidates.
    
    Args:
        student_request: Student's learning request text
        mode: Teaching mode (must be one of: "conceptual_review", "examples", "practice")
        merged: List of merged candidate dicts from Step 5 (each with lo_id, title, score, etc.)
        lo_mastery: Dict mapping lo_id (as string) to proficiency score (0.0-1.0)
    
    Returns:
        Dict with keys: student_request, mode, candidates (enriched with proficiency and suggested_notes)
    
    Raises:
        ValueError: If merged is empty or mode is invalid
    """
    if not merged:
        raise ValueError("merged candidates list cannot be empty")
    
    allowed_modes = {"conceptual_review", "examples", "practice"}
    if mode not in allowed_modes:
        raise ValueError(f"mode must be one of {allowed_modes}, got: {mode}")
    
    cands = []
    for c in merged:
        prof = float(lo_mastery.get(str(c["lo_id"]), 0.0))
        cands.append({**c, "proficiency": prof, "suggested_notes": proficiency_note(prof)})
    return {"student_request": student_request, "mode": mode, "candidates": cands}
```

### 6.3 Planner prompt (exact)

```python
PLANNER_SYSTEM_PROMPT = \"\"\"You are the Tutoring Planner.

You will be given INPUT_JSON with:
- student_request
- mode (conceptual_review|examples|practice)
- candidates: merged Top-K LO candidates, each with how_to_teach, why_to_teach, prereq_lo_ids, and proficiency

Your job: create a simplified plan:
- current_plan: exactly 1 PRIMARY LO + up to 2 dependent LOs
- future_plan: exactly 1 LO for next time

Rules:
1) Exactly one LO in current_plan has is_primary=true.
2) current_plan length: 1 to 3.
3) future_plan length: exactly 1.
4) Use the provided mode for all LOs.
5) Copy how_to_teach and why_to_teach from the selected candidates (do not invent).
6) Use proficiency + suggested_notes to write notes (keep notes short).
7) book/unit/chapter must come from the PRIMARY LO.

Return ONLY valid JSON matching the plan schema (no extra keys).\"\"\"
```

### 6.4 Planner call (add to cell)

```python
def planner_llm(student_request: str, mode: str, merged: list[dict], lo_mastery: dict[str, float]) -> dict:
    """Generate simplified tutoring plan from merged candidates.
    
    Args:
        student_request: Student's learning request text
        mode: Teaching mode (conceptual_review|examples|practice)
        merged: List of merged candidate dicts from Step 5
        lo_mastery: Dict mapping lo_id (as string) to proficiency score (0.0-1.0)
    
    Returns:
        Plan dict matching simplified plan schema (Step 2), validated
    
    Raises:
        ValueError: If merged is empty, mode is invalid, or plan validation fails
        RuntimeError: If LLM API call fails (no retry logic - handled by coach if needed)
    
    Edge cases:
    - If merged is empty, raises ValueError before calling LLM
    - If LLM returns invalid JSON, _chat_json will raise an exception
    - If plan validation fails, raises ValueError with descriptive message
    - Handle API failures at coach level
    """
    if not merged:
        raise ValueError("Cannot create plan: no candidates provided")
    payload = build_planner_input(student_request, mode, merged, lo_mastery)
    user = "INPUT_JSON:\\n" + json.dumps(payload, indent=2)
    plan = _chat_json(PLANNER_SYSTEM_PROMPT, user, temperature=0.0)
    validate_plan(plan)
    return plan
```

**Concrete example:**

**Input:**
```python
student_request = "I want to learn derivatives"
mode = "conceptual_review"
merged = [
    {
        "lo_id": 42,
        "title": "The Chain Rule",
        "score": 0.92,
        "source": "merged",
        "book": "Calculus Volume 1",
        "unit": "Derivatives",
        "chapter": "Differentiation Rules",
        "how_to_teach": "Start with the intuition: when one function is inside another...",
        "why_to_teach": "The chain rule is essential for differentiating composite functions...",
        "prereq_lo_ids": [41, 40]
    },
    {
        "lo_id": 41,
        "title": "Differentiation Rules",
        "score": 0.78,
        "source": "merged",
        "book": "Calculus Volume 1",
        "unit": "Derivatives",
        "chapter": "Differentiation Rules",
        "how_to_teach": "Cover power rule, product rule, and quotient rule...",
        "why_to_teach": "These foundational rules are prerequisites...",
        "prereq_lo_ids": []
    }
]
lo_mastery = {"42": 0.35, "41": 0.65, "40": 0.0}
```

**Output (plan dict):**
```json
{
  "subject": "calculus",
  "mode": "conceptual_review",
  "current_plan": [
    {
      "lo_id": 42,
      "title": "The Chain Rule",
      "proficiency": 0.35,
      "notes": "Student is new to this topic. Start with intuition about nested functions before introducing notation.",
      "is_primary": true,
      "how_to_teach": "Start with the intuition: when one function is inside another...",
      "why_to_teach": "The chain rule is essential for differentiating composite functions..."
    },
    {
      "lo_id": 41,
      "title": "Differentiation Rules",
      "proficiency": 0.65,
      "notes": "Prerequisite — student has solid understanding, quick review only.",
      "is_primary": false,
      "how_to_teach": "Cover power rule, product rule, and quotient rule...",
      "why_to_teach": "These foundational rules are prerequisites..."
    }
  ],
  "future_plan": [
    {
      "lo_id": 43,
      "title": "Implicit Differentiation",
      "proficiency": 0.0,
      "notes": "Natural next step after mastering the chain rule.",
      "is_primary": false,
      "how_to_teach": "Show how to differentiate equations where y is not isolated...",
      "why_to_teach": "Extends chain rule to equations that cannot be solved for y explicitly."
    }
  ],
  "book": "Calculus Volume 1",
  "unit": "Derivatives",
  "chapter": "Differentiation Rules"
}
```



---

## Step 7 — Tutor contract (no confirmation; follow current_plan only)

**Input:** `current_plan` only (+ optional image, conversation history)  
**Process:** tutor teaches from plan; off-plan → ask to end + hand off to coach  
**Output:** strict JSON with `switch_topic_request` when off-plan

**When to call:** Call `tutor_llm()` or `tutor_llm_with_optional_image()` from the coach during an active tutoring session (after plan is created in Step 6). Use `tutor_llm_with_optional_image` when student provides an image, otherwise use `tutor_llm`.

**Usage:** The returned dict contains `message_to_student` (displayed to user) and `switch_topic_request` (used by coach to handle topic switches). The tutor follows the `current_plan` without asking for confirmation.

**Error handling:** Tutor calls use `_chat_json()` which does not have retry logic (retry logic is coach-only). If the LLM call fails, the function will raise an exception. Handle empty/null responses and API failures at the coach level.

### 7.1 Exact coach → tutor handoff payload

**Important:** only pass `current_plan` (coach keeps `future_plan`).

**Edge cases:**
- If `current_plan` is empty, raise `ValueError` before calling tutor
- If `conversation_history` is empty, pass empty list `[]`
- `conversation_history` format: list of dicts with `speaker` ("student" or "tutor") and `text` (string)
- `image` should be `null` if no image, or the image path string if provided

**Concrete example:**

```json
{
  "mode": "conceptual_review",
  "subject": "calculus",
  "current_plan": [
    {
      "lo_id": 42,
      "title": "The Chain Rule",
      "proficiency": 0.35,
      "notes": "Student is new to this topic. Start with intuition about nested functions before introducing notation.",
      "is_primary": true,
      "how_to_teach": "Start with the intuition: when one function is inside another, the rate of change depends on both. Use the notation dy/dx = (dy/du)(du/dx). Walk through simple examples like (x^2 + 1)^3 before moving to trigonometric compositions.",
      "why_to_teach": "The chain rule is essential for differentiating composite functions, which appear constantly in real-world applications. Without it, students cannot handle nested expressions or implicit differentiation."
    },
    {
      "lo_id": 41,
      "title": "Differentiation Rules",
      "proficiency": 0.65,
      "notes": "Prerequisite — student has solid understanding, quick review only.",
      "is_primary": false,
      "how_to_teach": "Cover power rule, product rule, and quotient rule systematically. Emphasize pattern recognition and when to apply each rule.",
      "why_to_teach": "These foundational rules are prerequisites for the chain rule and all advanced differentiation techniques."
    }
  ],
  "conversation_history": [
    {"speaker": "student", "text": "I want to learn derivatives"}
  ],
  "image": null
}
```

### 7.2 Tutor system prompt (exact)

```python
TUTOR_SYSTEM_PROMPT = \"\"\"You are the learning tutor.

You will be given:
- subject, mode
- current_plan: list of LOs to teach (exactly one is_primary=true)
- conversation_history: the tutoring session history so far
- optional image: the student's image (if provided)

Rules (must follow):
1) NO PLAN CONFIRMATION. Start teaching immediately.
2) STAY ON PLAN. Only teach content related to current_plan LO titles.
3) OFF-PLAN HANDOFF: If the student asks for a different topic than current_plan:
   - Ask: "That's a different topic. Would you like to end this session and work on that instead?"
   - Set needs_topic_confirmation=true and switch_topic_request to the student's exact wording.
   - Do NOT teach the new topic yet.

Return ONLY valid JSON:
{
  "message_to_student": "string",
  "end_activity": boolean,
  "silent_end": boolean,
  "needs_topic_confirmation": boolean,
  "switch_topic_request": null or "student text",
  "session_summary": {
    "topics_covered": ["..."],
    "student_understanding": "excellent|good|satisfactory|needs_practice|struggling",
    "notes": "optional"
  }
}\"\"\"
```

### 7.3 Tutor call (add)

```python
def tutor_llm(handoff_payload: dict) -> dict:
    """Call tutor LLM with handoff payload.
    
    Args:
        handoff_payload: Dict with keys: mode, subject, current_plan, conversation_history, image
    
    Returns:
        Dict with keys: message_to_student, end_activity, silent_end, needs_topic_confirmation,
        switch_topic_request, session_summary
    
    Edge cases:
    - If current_plan is empty, raise ValueError before calling
    - If LLM returns invalid JSON, _chat_json will raise an exception
    """
    if not handoff_payload.get("current_plan"):
        raise ValueError("handoff_payload must contain non-empty current_plan")
    user = "INPUT_JSON:\\n" + json.dumps(handoff_payload, indent=2)
    return _chat_json(TUTOR_SYSTEM_PROMPT, user, temperature=0.0)
```

### 7.4 Passing the raw image to the Tutor (native vision)

If you want the tutor to "see" the image, you must pass it as an `image_url` message part (data URL for local files).

Add:

```python
def image_path_to_openai_part(image_path: str) -> dict:
    """Convert image file path to OpenAI vision API format.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Dict with type "image_url" and base64-encoded data URL
    
    Edge cases:
    - If file not found, Path.read_bytes() will raise FileNotFoundError
    - Assumes image format can be detected (defaults to PNG in data URL)
    """
    b = Path(image_path).read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

from typing import Optional

def tutor_llm_with_optional_image(handoff_payload: dict, image_path: Optional[str]) -> dict:
    """Call tutor LLM with optional image support.
    
    Args:
        handoff_payload: Dict with keys: mode, subject, current_plan, conversation_history, image
        image_path: Optional path to student's image file
    
    Returns:
        Dict with keys: message_to_student, end_activity, silent_end, needs_topic_confirmation,
        switch_topic_request, session_summary
    
    Edge cases:
    - If image_path is None or empty, falls back to tutor_llm() (text-only)
    - If image file not found, raises FileNotFoundError
    - If LLM response is empty/null, _coerce_json will raise an exception
    """
    if not image_path:
        return tutor_llm(handoff_payload)
    parts = [
        {"type": "text", "text": "INPUT_JSON:\\n" + json.dumps(handoff_payload, indent=2)},
        image_path_to_openai_part(image_path),
    ]
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        temperature=0,
        messages=[{"role": "system", "content": TUTOR_SYSTEM_PROMPT}, {"role": "user", "content": parts}],
    )
    return _coerce_json(resp.choices[0].message.content)
```

---

## Step 8 — Coach state machine (plan persistence + per-turn plan evaluation)

**Input:** each student message (+ optional image)  
**Process:** maintain `active_plan`; decide plan vs tutor vs topic switch  
**Output:** assistant message each turn; update `active_plan` / memory

**When to call:** The coach state machine runs in the main loop (`run_seamless_assistant_v2()`), processing each student message turn-by-turn.

**Usage:** This orchestrates the entire system: maintains plan state (`active_plan`, `lo_mastery`), routes to retriever/planner/tutor based on current state, and handles topic switches via plan-guard evaluation.

**Error handling:** Failures in retrieval, planning, or tutoring should be handled gracefully. If LLM calls fail, provide fallback messages to the student. The main loop should continue even if individual steps fail (log errors but don't crash).

### 8.1 Coach state (add globals near main loop)

```python
active_plan = None  # dict matching simplified plan schema
lo_mastery = {}     # {"42": 0.7, ...} — see Step 6.1 for format details
current_image_path = None
current_ocr_text = ""
```

### 8.2 Plan-guard: coach-level "is this still on-plan?"

Requirement: the coach evaluates *every* new student message in conjunction with the current plan.

Add a small classifier that returns strict JSON:

```python
PLAN_GUARD_PROMPT = \"\"\"You are a plan-guard. Decide if the student's message is within the current tutoring plan.

Return ONLY JSON:
{
  "decision": "continue|switch_topic|end_session",
  "switch_topic_request": null or "student text"
}

Rules:
- If the student is ending ("thanks", "done", "quit") => end_session
- If the student requests a different topic than the plan => switch_topic
- Else => continue
\"\"\"

def plan_guard(active_plan: dict, student_text: str) -> dict:
    """Evaluate if student message is within current plan.
    
    Args:
        active_plan: Plan dict matching simplified plan schema (must not be None)
        student_text: Student's message text
    
    Returns:
        Dict with keys: decision ("continue"|"switch_topic"|"end_session"), 
        switch_topic_request (str or None)
    
    Edge cases:
    - If active_plan is None, raises ValueError (caller should check first)
    - If current_plan is empty, returns decision="end_session"
    - If LLM call fails, raises RuntimeError (handle at coach level)
    
    """
    if active_plan is None:
        raise ValueError("active_plan cannot be None")
    payload = {
        "current_plan_titles": [x.get("title") for x in active_plan.get("current_plan", [])],
        "student_text": student_text,
    }
    user = "INPUT_JSON:\\n" + json.dumps(payload, indent=2)
    return _chat_json(PLAN_GUARD_PROMPT, user, temperature=0.0)
```

### 8.3 OCR (minimal)

For now, OCR can be a simple vision call that returns `{extracted_text, query}`. You can replace this later.

```python
OCR_SYSTEM_PROMPT = \"\"\"Extract any visible text/math from the image. Return JSON:
{ "extracted_text": "...", "query": "short retrieval query", "confidence": 0.0 }\"\"\"

def ocr_image(image_path: str, user_text: str) -> dict:
    """Extract text and generate query from image using vision model.
    
    Args:
        image_path: Path to image file
        user_text: Optional user-provided text context
    
    Returns:
        Dict with keys: extracted_text (str), query (str), confidence (float)
    
    Edge cases:
    - If image file not found, Path.read_bytes() raises FileNotFoundError
    - If image is corrupted, raises PIL.UnidentifiedImageError
    - If LLM response is invalid, _coerce_json will raise an exception
    - Handle these at coach level
    """
    # Build OpenAI vision input as base64 data url
    b = Path(image_path).read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    img_part = {"type":"image_url","image_url":{"url": f"data:image/png;base64,{b64}"}}
    user = [
        {"type":"text","text": f"User text: {user_text or ''}"},
        img_part
    ]
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        temperature=0,
        messages=[{"role":"system","content": OCR_SYSTEM_PROMPT}, {"role":"user","content": user}],
    )
    return _coerce_json(resp.choices[0].message.content)
```

### 8.4 Image detection (so a user can provide an image path in the REPL)

Add this helper (works in notebook REPL style):

```python
def looks_like_image_path(token: str) -> bool:
    """Check if a token looks like an image file path.
    
    Args:
        token: String token to check
    
    Returns:
        True if token is an existing file with image extension
    
    Edge cases:
    - If path doesn't exist, returns False
    - Case-insensitive extension matching
    """
    p = Path(token)
    return p.exists() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}

def detect_image_in_text(user_input: str) -> tuple[str | None, str]:
    """Detect image path in user input and extract remaining text.
    
    Args:
        user_input: User's input string (may contain image path)
    
    Returns:
        Tuple of (image_path or None, remaining_text)
    
    Edge cases:
    - If no image path found, returns (None, user_input)
    - If multiple tokens match, returns first match
    - Resolves relative paths to absolute paths
    """
    tokens = user_input.split()
    for t in tokens:
        if looks_like_image_path(t):
            remaining = " ".join([x for x in tokens if x != t]).strip()
            return str(Path(t).resolve()), remaining
    return None, user_input
```

### 8.5 Per-turn algorithm (replace the `# ---------- MAIN LOOP ----------` logic)

Replace the baseline “in_bot_session” branching with this:

1. Detect (optional) image input and run OCR
2. If `active_plan is None` → retrieve → plan → tutor immediately
3. If `active_plan exists` → build tutor payload with `current_plan` and call tutor
4. Coach always runs `plan_guard(active_plan, student_text)` before tutoring
5. If off-plan: coach asks confirmation to end session and switch topics; if confirmed → clear plan and replan

**Exact boundary payloads**

- Coach → Retriever: `(student_text, ocr_text, image_path)`
- Retriever → Planner: `{student_request, mode, candidates[] with KG fields + proficiency}`
- Coach → Tutor: `{subject, mode, current_plan, conversation_history, image}`

### 8.6 Minimal reference implementation (paste to replace the baseline `run_seamless_assistant()` body)

This is the *shape* of the final orchestration. You will need to connect it to your existing debug prints.

```python
def run_seamless_assistant_v2():
    global active_plan, lo_mastery, current_image_path, current_ocr_text

    lo_by_id, prereqs_by_lo = load_kg()
    lo_embeddings, lo_ids_by_row = load_text_artifacts()
    image_embeddings, image_meta = load_image_artifacts()

    tutor_history = []  # messages inside tutoring session
    pending_switch_request = None  # when not None, we are awaiting yes/no

    print("Adaptive Learning Coach (v2) — type 'quit' to exit\\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        # Handle switch confirmation (yes/no) if we previously asked
        if pending_switch_request is not None:
            norm = user_input.strip().lower()
            if norm in {"yes", "y", "yep", "sure", "ok", "okay"}:
                # End current plan and treat the pending request as the next request
                text_only = pending_switch_request
                pending_switch_request = None
                active_plan = None
                tutor_history = []
                print("Assistant: Got it — switching topics.\\n")
            elif norm in {"no", "n", "nope"}:
                pending_switch_request = None
                print("Assistant: Okay — let's stay on the current topic.\\n")
                continue
            else:
                print("Assistant: Please answer 'yes' or 'no'.\\n")
                continue

        # Image detection + OCR
        img, text_only = detect_image_in_text(user_input)
        if img:
            current_image_path = img
            ocr = ocr_image(img, text_only)
            current_ocr_text = (ocr.get("extracted_text") or "") + "\\n" + (ocr.get("query") or "")

        # If we have a plan, coach evaluates the message with the plan in mind
        if active_plan is not None:
            # Validate active_plan is not None (plan_guard will raise if None)
            guard = plan_guard(active_plan, text_only)
            if guard.get("decision") == "end_session":
                active_plan = None
                tutor_history = []
                print("Assistant: Sounds good — session ended. What would you like to work on next?\\n")
                continue
            if guard.get("decision") == "switch_topic":
                pending_switch_request = guard.get("switch_topic_request") or text_only
                print("Assistant: That's a different topic. Would you like to end this session and work on that instead?\\n")
                continue

        # Plan if needed
        if active_plan is None:
            mode = "conceptual_review"  # MVP default; you can ask the user instead
            merged = retrieve_candidates(
                student_text=text_only,
                ocr_text=current_ocr_text,
                image_path=current_image_path,
                lo_embeddings=lo_embeddings,
                lo_ids_by_row=lo_ids_by_row,
                lo_by_id=lo_by_id,
                prereqs_by_lo=prereqs_by_lo,
                image_embeddings=image_embeddings,
                image_meta=image_meta,
            )
            active_plan = planner_llm(text_only, mode, merged, lo_mastery)
            if DEBUG_MODE:
                print("PLAN:", json.dumps(active_plan, indent=2))

        # Tutor (current_plan only)
        tutor_payload = {
            "subject": active_plan.get("subject"),
            "mode": active_plan.get("mode"),
            "current_plan": active_plan.get("current_plan"),
            "conversation_history": tutor_history + [{"speaker": "student", "text": text_only}],
            "image": None,
        }
        bot = tutor_llm_with_optional_image(tutor_payload, current_image_path)
        # Validate tutor response structure
        if not isinstance(bot, dict) or "message_to_student" not in bot:
            print("Assistant: Sorry, I encountered an error. Let's try again.\\n")
            continue
        msg = (bot.get("message_to_student") or "").strip()
        if msg:
            print(f"Assistant: {msg}\\n")
            tutor_history.append({"speaker": "student", "text": text_only})
            tutor_history.append({"speaker": "assistant", "text": msg})
```

### 8.7 Runtime boundary payloads (quick reference)

| Boundary | Keys / Args | See Details |
|----------|-------------|-------------|
| Coach → Retriever | `student_text`, `ocr_text`, `image_path` | Step 5.4 `retrieve_candidates()` |
| Retriever → Planner | `student_request`, `mode`, `candidates[]` | Step 6.2 `build_planner_input()` |
| Coach → Tutor | `subject`, `mode`, `current_plan`, `conversation_history`, `image` | Step 7.1 (full example) |

---

## Step 9 — Fully worked example 

This section shows exactly what happens at each step when a student asks for help.

---

### 9.1 Example A: First tutoring request (no prior plan)

**Student types:** `Help me understand the chain rule`

---

**Step 1: Image detection**

No image detected. `image_path = None`, `ocr_text = ""`.

---

**Step 2: Plan guard**

`active_plan` is `None`, so skip plan guard.

---

**Step 3: Retrieval**

Call `retrieve_candidates(student_text="Help me understand the chain rule", ocr_text="", image_path=None, ...)`.

**Debug output (printed to console):**

```
=== RETRIEVAL DEBUG ===
Query: Help me understand the chain rule
Text Top:
  [0.891] The Chain Rule (LO 42)
  [0.823] Differentiation Rules (LO 41)
  [0.756] The Derivative as a Function (LO 40)
  [0.712] Implicit Differentiation (LO 43)
  [0.698] Defining the Derivative (LO 39)
  [0.654] A Preview of Calculus (LO 38)
Merged Top:
  [0.891] The Chain Rule (LO 42)
  [0.823] Differentiation Rules (LO 41)
  [0.756] The Derivative as a Function (LO 40)
  [0.712] Implicit Differentiation (LO 43)
  [0.698] Defining the Derivative (LO 39)
  [0.654] A Preview of Calculus (LO 38)
========================
```

**Merged candidates:** Top 2 shown (format matches Step 5.4 `build_candidate()` output + proficiency fields):

| lo_id | title | score | proficiency | suggested_notes |
|-------|-------|-------|-------------|-----------------|
| 42 | The Chain Rule | 0.891 | 0.0 | New/struggling |
| 41 | Differentiation Rules | 0.823 | 0.65 | Solid understanding |

---

**Step 4: Planner LLM call**

Input: `{student_request, mode, candidates}` (see Step 6.2)

**Planner output** (matches Step 2 schema):

| Field | Value |
|-------|-------|
| subject | calculus |
| mode | conceptual_review |
| current_plan | LO 42 (primary, proficiency 0.0) + LO 41 (prereq, proficiency 0.65) |
| future_plan | LO 43 (Implicit Differentiation) |
| book/unit/chapter | Calculus Volume 1 / Derivatives / Differentiation Rules |

---

**Step 5: Tutor handoff**

Payload matches Step 7.1 schema with `current_plan` from planner output (excludes `future_plan`).

---

**Step 6: Tutor response**

**Tutor output:**

```json
{
  "message_to_student": "Great question! The chain rule helps us differentiate composite functions — that's when one function is 'inside' another.\n\nThink of it like this: if you have f(g(x)), you're applying f to the output of g. To find how fast the whole thing changes, you need to consider both how fast g changes AND how fast f reacts to that change.\n\nThe formula is: (f(g(x)))' = f'(g(x)) · g'(x)\n\nLet's try a simple example: find the derivative of (x² + 1)³.\n\nHere, the 'outer' function is u³ and the 'inner' function is u = x² + 1.\n\nCan you tell me what you think the derivative of the outer function u³ is, treating u as the variable?",
  "end_activity": false,
  "silent_end": false,
  "needs_topic_confirmation": false,
  "switch_topic_request": null,
  "session_summary": {
    "topics_covered": ["Chain rule introduction", "Composite function concept"],
    "student_understanding": "satisfactory",
    "notes": "Student engaged, asking foundational question"
  }
}
```

**Printed to student:** The `message_to_student` field from above.

---

### 9.2 Example B: Off-plan request (topic switch)

**Context:** Student is in a tutoring session on "The Chain Rule". They now type:

**Student types:** `when is my exam?`

---

**Step 1: Plan guard**

`active_plan` exists. Call `plan_guard(active_plan, "when is my exam?")`.

**Plan guard output:**

```json
{
  "decision": "switch_topic",
  "switch_topic_request": "when is my exam?"
}
```

---

**Step 2: Coach asks for confirmation**

**Printed to student:**

```
Assistant: That's a different topic. Would you like to end this session and work on that instead?
```

`pending_switch_request = "when is my exam?"`.

---

**Step 3: Student confirms**

**Student types:** `yes`

---

**Step 4: Coach clears plan and re-routes**

`active_plan = None`. The coach recognizes this as an FAQ intent and routes to the FAQ flow.

**Printed to student:**

```
Assistant: Got it — switching topics.
```

(FAQ flow begins for exam schedule.)

---

### 9.3 Example C: Image input with OCR + CLIP retrieval

**Student types:** `/path/to/graph.png what is this showing?`

---

**Step 1: Image detection**

`image_path = "/path/to/graph.png"`, `text_only = "what is this showing?"`.

---

**Step 2: OCR**

Call `ocr_image("/path/to/graph.png", "what is this showing?")`.

**OCR output:**

```json
{
  "extracted_text": "y = sin(x²)\ndy/dx = ?",
  "query": "derivative of sine of x squared",
  "confidence": 0.92
}
```

`current_ocr_text = "y = sin(x²)\ndy/dx = ?\nderivative of sine of x squared"`.

---

**Step 3: Retrieval (parallel text + image)**

**Text retrieval** uses query: `"what is this showing?\ny = sin(x²)\ndy/dx = ?\nderivative of sine of x squared"`.

**Image retrieval** uses CLIP embedding of the raw image.

**Debug output:**

```
=== RETRIEVAL DEBUG ===
Query: what is this showing?
y = sin(x²)
dy/dx = ?
derivative of sine of x squared
Text Top:
  [0.867] The Chain Rule (LO 42)
  [0.801] Derivatives of Trigonometric Functions (LO 44)
  [0.756] Differentiation Rules (LO 41)
Image Top:
  [0.823] The Chain Rule (LO 42)
  [0.789] Derivatives of Trigonometric Functions (LO 44)
Merged Top:
  [0.867] The Chain Rule (LO 42)
  [0.801] Derivatives of Trigonometric Functions (LO 44)
  [0.756] Differentiation Rules (LO 41)
========================
```

The chain rule is identified as the primary topic because the image shows a composite function (sine of x²).

---

## Acceptance criteria

1. Single cell runs end-to-end in Jupyter
2. `active_plan` persists across turns and is used to interpret new student messages
3. Retrieval loads embeddings from disk and computes cosine similarity + Top‑K
4. OCR + image embedding retrieval both run (when image present) and merge candidates
5. Planner produces simplified plan JSON that passes `validate_plan`
6. Tutor starts teaching immediately and stays on `current_plan` (no confirmation)
