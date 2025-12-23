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

The baseline is “pure LLM routing + toy OpenStax lists”. We will upgrade it to:

1. **File-based KG**: load LOs and prereq edges from CSVs at runtime
2. **Retriever**: use embeddings to retrieve Top‑K LO candidates (text + image)
3. **Planner**: generate a **simplified plan** JSON from candidates
4. **Tutor**: teach from **current_plan only**, no confirmation, off-plan → ask to end session and switch
5. **Coach**: keep `active_plan` and run a plan-guard check on every student message

---

## Runtime Data Files (expected)

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

## Step 1 — Add configuration + imports (single cell)

**Input:** none  
**Process:** add constants + imports near the top of the cell (after `DEBUG_MODE`)  
**Output:** the rest of the cell can use consistent config

Add:

```python
import os
from pathlib import Path
import base64
import numpy as np
import pandas as pd

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

### 1.2 Install missing packages (only if needed)

If your notebook env doesn’t have these installed, run (still within the single cell, at the very top):

```python
# Optional: install deps in-notebook
# !pip install -q numpy pandas sentence-transformers pillow
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
  "subject": "calculus|algebra|trigonometry",
  "mode": "conceptual_review|examples|practice",
  "current_plan": [
    {
      "lo_id": 42,
      "title": "The Chain Rule",
      "proficiency": 0.35,
      "notes": "short note",
      "is_primary": true,
      "how_to_teach": "string",
      "why_to_teach": "string"
    }
  ],
  "future_plan": [
    {
      "lo_id": 45,
      "title": "Implicit Differentiation",
      "proficiency": 0.0,
      "notes": "",
      "is_primary": false,
      "how_to_teach": "string",
      "why_to_teach": "string"
    }
  ],
  "book": "string",
  "unit": "string",
  "chapter": "string"
}
```

**Rules**
- `current_plan`: length 1–3
- exactly 1 item in `current_plan` has `is_primary=true`
- `future_plan`: exactly 1 item
- all items use the **same** `mode` (the mode is the session mode)

### 2.2 Validator (add to cell)

```python
def validate_plan(plan: dict) -> None:
    if not isinstance(plan, dict):
        raise ValueError("plan must be a dict")
    for k in ["subject", "mode", "current_plan", "future_plan"]:
        if k not in plan:
            raise ValueError(f"plan missing key: {k}")
    if not (1 <= len(plan["current_plan"]) <= 3):
        raise ValueError("current_plan must have 1-3 items")
    primaries = [x for x in plan["current_plan"] if x.get("is_primary") is True]
    if len(primaries) != 1:
        raise ValueError("current_plan must contain exactly one primary LO")
    if len(plan["future_plan"]) != 1:
        raise ValueError("future_plan must contain exactly one LO")
```

---

## Step 3 — Load the knowledge graph (CSV → lookups)

**Input:** `demo/lo_index.csv`, `demo/edges_prereqs.csv`  
**Process:** load both files and build fast lookups  
**Output:** `lo_by_id` and `prereqs_by_lo`

### 3.1 Required KG fields surfaced to downstream components

From `lo_index.csv`:
- `lo_id` (int)
- `learning_objective` (string) → becomes `title`
- `book`, `unit`, `chapter` (strings)
- *(recommended)* `how_to_teach`, `why_to_teach` (strings)

From `edges_prereqs.csv`:
- `source_lo_id` (prereq)
- `target_lo_id` (dependent)

### 3.2 Loader (add to cell)

```python
def load_kg():
    lo_df = pd.read_csv(KG_DIR / "lo_index.csv")
    prereq_df = pd.read_csv(KG_DIR / "edges_prereqs.csv")

    lo_by_id = {}
    for _, row in lo_df.iterrows():
        lo_id = int(row["lo_id"])
        lo_by_id[lo_id] = {
            "lo_id": lo_id,
            "title": str(row["learning_objective"]),
            "book": str(row.get("book", "")),
            "unit": str(row.get("unit", "")),
            "chapter": str(row.get("chapter", "")),
            "how_to_teach": str(row.get("how_to_teach", "")),
            "why_to_teach": str(row.get("why_to_teach", "")),
        }

    prereqs_by_lo = {}
    for _, row in prereq_df.iterrows():
        src = int(row["source_lo_id"])
        tgt = int(row["target_lo_id"])
        prereqs_by_lo.setdefault(tgt, []).append(src)

    return lo_by_id, prereqs_by_lo
```

**Concrete output example**

Input: `lo_by_id[42]`  
Output:

```json
{
  "lo_id": 42,
  "title": "The Chain Rule",
  "book": "Calculus Volume 1",
  "unit": "Derivatives",
  "chapter": "Differentiation Rules",
  "how_to_teach": "...",
  "why_to_teach": "..."
}
```

---

## Step 4 — Load offline embedding artifacts (runtime only loads)

**Input:** embedding `.npy` matrices + row index CSVs  
**Process:** load them once at startup  
**Output:** in-memory matrices for cosine similarity search

### 4.1 Expected artifact formats

Text:
- `lo_embeddings.npy`: `(num_los, dim)` float32 **L2-normalized row-wise**
- `lo_row_index.csv`: `row_index, lo_id`

Image:
- `image_embeddings.npy`: `(num_images, 512)` float32 **L2-normalized row-wise**
- `image_metadata.csv`: at least `image_id, path, lo_id, description, keywords`

### 4.2 Loader (add to cell)

```python
def load_text_artifacts():
    emb = np.load(ARTIFACT_DIR / "lo_embeddings.npy").astype("float32")
    rows = pd.read_csv(ARTIFACT_DIR / "lo_row_index.csv").sort_values("row_index")
    lo_ids_by_row = [int(x) for x in rows["lo_id"].tolist()]
    if emb.shape[0] != len(lo_ids_by_row):
        raise ValueError("lo_embeddings rows != lo_row_index rows")
    return emb, lo_ids_by_row


def load_image_artifacts():
    emb = np.load(IMAGE_CORPUS_DIR / "image_embeddings.npy").astype("float32")
    meta = pd.read_csv(IMAGE_CORPUS_DIR / "image_metadata.csv")
    if emb.shape[0] != len(meta):
        raise ValueError("image_embeddings rows != image_metadata rows")
    return emb, meta
```

---

## Step 5 — Runtime retrieval (text + image) with cosine similarity

**Input:** `student_text`, optional `ocr_text`, optional `image_path`  
**Process:** retrieve Top‑K candidates from text embeddings and image embeddings in parallel; merge  
**Output:** `merged_candidates` to feed the planner

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
    resp = client.embeddings.create(model=TEXT_EMBED_MODEL, input=text)
    vec = np.array(resp.data[0].embedding, dtype="float32")
    return l2_normalize(vec)
```

### 5.3 Image query embedding (runtime, CLIP)

Add:

```python
from sentence_transformers import SentenceTransformer
from PIL import Image

CLIP_MODEL_NAME = "clip-ViT-B-32"
clip_model = SentenceTransformer(CLIP_MODEL_NAME)

def embed_image_query(image_path: str) -> np.ndarray:
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


def retrieve_candidates(student_text: str, ocr_text: str, image_path: str, lo_embeddings, lo_ids_by_row, lo_by_id, prereqs_by_lo, image_embeddings, image_meta):
    query = "\\n".join([x for x in [student_text, ocr_text] if x]).strip()

    # Text retrieval
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

    if DEBUG_MODE:
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

### 6.1 Proficiency + notes (coach-owned)

Add:

```python
def proficiency_note(p: float) -> str:
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
def build_planner_input(student_request: str, mode: str, merged: list, lo_mastery: dict) -> dict:
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
def planner_llm(student_request: str, mode: str, merged: list, lo_mastery: dict) -> dict:
    payload = build_planner_input(student_request, mode, merged, lo_mastery)
    user = "INPUT_JSON:\\n" + json.dumps(payload, indent=2)
    plan = _chat_json(PLANNER_SYSTEM_PROMPT, user, temperature=0.0)
    validate_plan(plan)
    return plan
```



---

## Step 7 — Tutor contract (no confirmation; follow current_plan only)

**Input:** `current_plan` only (+ optional image, conversation history)  
**Process:** tutor teaches from plan; off-plan → ask to end + hand off to coach  
**Output:** strict JSON with `switch_topic_request` when off-plan

### 7.1 Exact coach → tutor handoff payload

**Important:** only pass `current_plan` (coach keeps `future_plan`).

```json
{
  "mode": "conceptual_review",
  "subject": "calculus",
  "current_plan": [ ... ],
  "conversation_history": [ {"speaker":"student","text":"..."} ],
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
    user = "INPUT_JSON:\\n" + json.dumps(handoff_payload, indent=2)
    return _chat_json(TUTOR_SYSTEM_PROMPT, user, temperature=0.0)
```

### 7.4 Passing the raw image to the Tutor (native vision)

If you want the tutor to “see” the image, you must pass it as an `image_url` message part (data URL for local files).

Add:

```python
def image_path_to_openai_part(image_path: str) -> dict:
    b = Path(image_path).read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

from typing import Optional

def tutor_llm_with_optional_image(handoff_payload: dict, image_path: Optional[str]) -> dict:
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

### 8.1 Coach state (add globals near main loop)

```python
active_plan = None  # dict matching simplified plan schema
lo_mastery = {}     # {"42": 0.7, ...}
current_image_path = None
current_ocr_text = ""
```

### 8.1 Plan-guard: coach-level “is this still on-plan?”

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
    payload = {
        "current_plan_titles": [x.get("title") for x in active_plan.get("current_plan", [])],
        "student_text": student_text,
    }
    user = "INPUT_JSON:\\n" + json.dumps(payload, indent=2)
    return _chat_json(PLAN_GUARD_PROMPT, user, temperature=0.0)
```

### 8.2 OCR (minimal)

For now, OCR can be a simple vision call that returns `{extracted_text, query}`. You can replace this later.

```python
OCR_SYSTEM_PROMPT = \"\"\"Extract any visible text/math from the image. Return JSON:
{ "extracted_text": "...", "query": "short retrieval query", "confidence": 0.0 }\"\"\"

def ocr_image(image_path: str, user_text: str) -> dict:
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

### 8.3 Image detection (so a user can provide an image path in the REPL)

Add this helper (works in notebook REPL style):

```python
def looks_like_image_path(token: str) -> bool:
    p = Path(token)
    return p.exists() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}

def detect_image_in_text(user_input: str) -> tuple[str | None, str]:
    tokens = user_input.split()
    for t in tokens:
        if looks_like_image_path(t):
            remaining = " ".join([x for x in tokens if x != t]).strip()
            return str(Path(t).resolve()), remaining
    return None, user_input
```

### 8.4 Per-turn algorithm (replace the `# ---------- MAIN LOOP ----------` logic)

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

### 8.5 Minimal reference implementation (paste to replace the baseline `run_seamless_assistant()` body)

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
        msg = (bot.get("message_to_student") or "").strip()
        if msg:
            print(f"Assistant: {msg}\\n")
            tutor_history.append({"speaker": "student", "text": text_only})
            tutor_history.append({"speaker": "assistant", "text": msg})
```

### 8.6 Exact runtime boundary payloads (copy/paste reference)

**Coach → Retriever (function args):**

- `student_text`: the student’s latest text
- `ocr_text`: OCR-extracted text/query (may be empty)
- `image_path`: raw image path (may be None)

**Retriever → Planner (INPUT_JSON):**

```json
{
  "student_request": "Help me with the chain rule",
  "mode": "conceptual_review",
  "candidates": [
    {
      "lo_id": 42,
      "title": "The Chain Rule",
      "score": 0.8921,
      "source": "merged",
      "book": "Calculus Volume 1",
      "unit": "Derivatives",
      "chapter": "Differentiation Rules",
      "prereq_lo_ids": [41, 10],
      "how_to_teach": "...",
      "why_to_teach": "...",
      "proficiency": 0.35,
      "suggested_notes": "New/struggling — start from fundamentals..."
    }
  ]
}
```

**Coach → Tutor (INPUT_JSON):**

```json
{
  "subject": "calculus",
  "mode": "conceptual_review",
  "current_plan": [
    {
      "lo_id": 42,
      "title": "The Chain Rule",
      "proficiency": 0.35,
      "notes": "New/struggling — start from fundamentals...",
      "is_primary": true,
      "how_to_teach": "...",
      "why_to_teach": "..."
    }
  ],
  "conversation_history": [
    {"speaker": "student", "text": "Help me with the chain rule"}
  ],
  "image": null
}
```

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

**Merged candidates (passed to planner):**

```json
[
  {
    "lo_id": 42,
    "title": "The Chain Rule",
    "score": 0.891,
    "source": "merged",
    "book": "Calculus Volume 1",
    "unit": "Derivatives",
    "chapter": "Differentiation Rules",
    "prereq_lo_ids": [41, 40],
    "how_to_teach": "Start with the intuition: when one function is inside another, the rate of change depends on both. Use the notation dy/dx = (dy/du)(du/dx). Walk through simple examples like (x^2 + 1)^3 before moving to trigonometric compositions.",
    "why_to_teach": "The chain rule is essential for differentiating composite functions, which appear constantly in real-world applications. Without it, students cannot handle nested expressions or implicit differentiation.",
    "proficiency": 0.0,
    "suggested_notes": "New/struggling — start from fundamentals and go step-by-step."
  },
  {
    "lo_id": 41,
    "title": "Differentiation Rules",
    "score": 0.823,
    "source": "merged",
    "book": "Calculus Volume 1",
    "unit": "Derivatives",
    "chapter": "Differentiation Rules",
    "prereq_lo_ids": [39, 40],
    "how_to_teach": "Cover power rule, product rule, and quotient rule systematically. Emphasize pattern recognition and when to apply each rule.",
    "why_to_teach": "These foundational rules are prerequisites for the chain rule and all advanced differentiation techniques.",
    "proficiency": 0.65,
    "suggested_notes": "Solid understanding — emphasize applications."
  }
]
```

---

**Step 4: Planner LLM call**

**Input to planner:**

```json
{
  "student_request": "Help me understand the chain rule",
  "mode": "conceptual_review",
  "candidates": [ ... as above ... ]
}
```

**Planner output (validated by `validate_plan`):**

```json
{
  "subject": "calculus",
  "mode": "conceptual_review",
  "current_plan": [
    {
      "lo_id": 42,
      "title": "The Chain Rule",
      "proficiency": 0.0,
      "notes": "Student is new to this topic. Start with intuition about nested functions before introducing notation.",
      "is_primary": true,
      "how_to_teach": "Start with the intuition: when one function is inside another, the rate of change depends on both. Use the notation dy/dx = (dy/du)(du/dx). Walk through simple examples like (x^2 + 1)^3 before moving to trigonometric compositions.",
      "why_to_teach": "The chain rule is essential for differentiating composite functions, which appear constantly in real-world applications."
    },
    {
      "lo_id": 41,
      "title": "Differentiation Rules",
      "proficiency": 0.65,
      "notes": "Prerequisite — student has solid understanding, quick review only.",
      "is_primary": false,
      "how_to_teach": "Cover power rule, product rule, and quotient rule systematically.",
      "why_to_teach": "These foundational rules are prerequisites for the chain rule."
    }
  ],
  "future_plan": [
    {
      "lo_id": 43,
      "title": "Implicit Differentiation",
      "proficiency": 0.0,
      "notes": "Natural next step after mastering the chain rule.",
      "is_primary": false,
      "how_to_teach": "Show how to differentiate equations where y is not isolated.",
      "why_to_teach": "Extends chain rule to equations that cannot be solved for y explicitly."
    }
  ],
  "book": "Calculus Volume 1",
  "unit": "Derivatives",
  "chapter": "Differentiation Rules"
}
```

---

**Step 5: Tutor handoff**

**Payload sent to tutor:**

```json
{
  "subject": "calculus",
  "mode": "conceptual_review",
  "current_plan": [
    {
      "lo_id": 42,
      "title": "The Chain Rule",
      "proficiency": 0.0,
      "notes": "Student is new to this topic. Start with intuition about nested functions before introducing notation.",
      "is_primary": true,
      "how_to_teach": "Start with the intuition: when one function is inside another, the rate of change depends on both. Use the notation dy/dx = (dy/du)(du/dx). Walk through simple examples like (x^2 + 1)^3 before moving to trigonometric compositions.",
      "why_to_teach": "The chain rule is essential for differentiating composite functions, which appear constantly in real-world applications."
    },
    {
      "lo_id": 41,
      "title": "Differentiation Rules",
      "proficiency": 0.65,
      "notes": "Prerequisite — student has solid understanding, quick review only.",
      "is_primary": false,
      "how_to_teach": "Cover power rule, product rule, and quotient rule systematically.",
      "why_to_teach": "These foundational rules are prerequisites for the chain rule."
    }
  ],
  "conversation_history": [
    {"speaker": "student", "text": "Help me understand the chain rule"}
  ],
  "image": null
}
```

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

**Printed to student:**

```
Assistant: Great question! The chain rule helps us differentiate composite functions — that's when one function is 'inside' another.

Think of it like this: if you have f(g(x)), you're applying f to the output of g. To find how fast the whole thing changes, you need to consider both how fast g changes AND how fast f reacts to that change.

The formula is: (f(g(x)))' = f'(g(x)) · g'(x)

Let's try a simple example: find the derivative of (x² + 1)³.

Here, the 'outer' function is u³ and the 'inner' function is u = x² + 1.

Can you tell me what you think the derivative of the outer function u³ is, treating u as the variable?
```

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


