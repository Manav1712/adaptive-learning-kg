"""
Minimal strict semantic LLM evaluator

Purpose:
- Judge each edge with a binary label per the relation type using an LLM
- Emit compact JSONL per edge and a minimal summary JSON

Inputs:
- --lo-index: CSV with LO metadata (expects columns: lo_id, learning_objective, unit, chapter, book)
- --content-items: CSV with content metadata (expects columns: content_id, content_type, lo_id_parent, text, learning_objective, unit, chapter, book)
- --edges-in: CSV with edges to judge (supports two schemas):
  - Prereqs: source_lo_id, target_lo_id, relation
  - Content links: source_lo_id, target_content_id, relation

Outputs:
- --jsonl-out: JSONL file with one record per edge
- --summary-out: JSON file with aggregate metrics per relation

Behavior:
- System prompt requires strict JSON only: {"label":"correct|incorrect","reason":"<=200 chars"}
- Temperature 0.0, no scores/confidence
- Prompts send only compact summaries (title + short aggregate_text slices)
- On any parse or API error, default to incorrect with short reason
- Unknown relations default to incorrect with reason "unsupported relation"
- Acceptance policy: label == "correct"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import re
import random

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ----------------------------
# Config
# ----------------------------


@dataclass
class EvalConfig:
	model: str = "gpt-4o-mini"
	temperature: float = 0.0
	max_retries: int = 2
	# IO
	input_lo_index: str = "data/processed/lo_index.csv"
	input_content_items: str = "data/processed/content_items.csv"
	input_edges: str = "data/processed/edges_prereqs.csv"
	jsonl_out: str = "data/processed/llm_edge_checks.jsonl"
	summary_out: str = "data/processed/llm_edge_checks_summary.json"
	# Cost control
	limit: Optional[int] = None


# ----------------------------
# Prompt builders (minimal)
# ----------------------------


def build_prereq_prompt(src: dict, tgt: dict) -> str:
    return (
        "Task: Determine if the source learning objective is a prerequisite for the target.\n"
        "Definitions:\n"
        "- prerequisite: knowledge in source is required for a learner to achieve the target at its intended depth "
        "(not just to recall, but to apply/analyze as appropriate).\n"
        "- supports: knowledge in source is helpful but not strictly required.\n"
        "- incorrect: the source is unrelated, equivalent, or part-of the target.\n\n"
        "Respond with strict JSON only: {\"label\":\"prerequisite|supports|incorrect\",\"reason\":\"<=200 chars\"}\n\n"
        f"Source LO title: {src.get('learning_objective','')}\n"
        f"Source summary: {str(src.get('aggregate_text',''))[:800]}\n\n"
        f"Target LO title: {tgt.get('learning_objective','')}\n"
        f"Target summary: {str(tgt.get('aggregate_text',''))[:800]}\n"
    )


def build_content_prompt(rel: str, lo: dict, content: dict) -> str:
	purpose = {
		"explained_by": "teaches or explains the LO clearly",
		"exemplified_by": "shows worked examples directly illustrating the LO",
		"practiced_by": "provides opportunities to practice skills in the LO",
	}.get(rel, "teaches the LO")
	return (
		f"Task: Determine if the content {purpose}.\n"
		"Definitions:\n"
		"- correct: content directly fulfills this role for the LO.\n"
		"- supports: content is related and helpful for the LO, even if not a perfect match.\n"
		"- incorrect: content is completely unrelated, misleading, or harmful to learning the LO.\n\n"
		"Note: Cross-LO connections are valuable - content can support multiple related learning objectives.\n"
		"Only mark as incorrect if the content is truly irrelevant or would confuse learners.\n\n"
		"Respond with strict JSON only (no code fences, no extra text): {\"label\":\"correct|supports|incorrect\",\"reason\":\"<=200 chars, avoid double quotes in the text\"}\n\n"
		f"LO title: {lo.get('learning_objective','')}\n"
		f"LO summary: {str(lo.get('aggregate_text',''))[:800]}\n\n"
		f"Content title: {content.get('title','')}\n"
		f"Content summary: {str(content.get('aggregate_text',''))[:800]}\n"
	)


# ----------------------------
# Data loading and lookups
# ----------------------------


def load_lo_lookup(path: str, content_items_df: pd.DataFrame) -> Dict[str, dict]:
	"""Build LO lookup with an aggregate_text from any content rows that reference the LO.

	Inputs: lo_index.csv and content_items dataframe (lo_id_parent, text)
	Output: dict lo_id -> {learning_objective, aggregate_text, unit, chapter, book}
	"""
	lo_df = pd.read_csv(path)
	# aggregate content per LO
	texts: Dict[str, List[str]] = {}
	for _, row in content_items_df.iterrows():
		lo_id_parent = str(row.get("lo_id_parent") or "")
		if not lo_id_parent:
			continue
		text = str(row.get("text") or "")
		if text:
			texts.setdefault(lo_id_parent, []).append(text)
	lookup: Dict[str, dict] = {}
	for _, r in lo_df.iterrows():
		lid = str(r.get("lo_id") or "")
		agg = "\n\n".join(texts.get(lid, []))
		lookup[lid] = {
			"lo_id": lid,
			"learning_objective": str(r.get("learning_objective") or ""),
			"aggregate_text": agg,
			"unit": str(r.get("unit") or ""),
			"chapter": str(r.get("chapter") or ""),
			"book": str(r.get("book") or ""),
		}
	return lookup


def load_content_lookup(path: str) -> Dict[str, dict]:
	"""Build content lookup with compact summaries.

	Output: content_id -> {title, aggregate_text, content_type, lo_id_parent}
	"""
	cdf = pd.read_csv(path)
	lookup: Dict[str, dict] = {}
	for _, r in cdf.iterrows():
		cid = str(r.get("content_id") or "")
		lookup[cid] = {
			"content_id": cid,
			"title": str(r.get("learning_objective") or ""),  # content has no separate title; use LO text
			"aggregate_text": str(r.get("text") or ""),
			"content_type": str(r.get("content_type") or ""),
			"lo_id_parent": str(r.get("lo_id_parent") or ""),
		}
	return lookup


# ----------------------------
# LLM call and strict parse
# ----------------------------


def _sanitize_and_extract_json(text: str) -> Optional[dict]:
	"""Attempt to coerce non-JSON-ish outputs into a JSON object.

	Strategies:
	- Strip markdown code fences
	- Take substring between first '{' and last '}'
	- Normalize smart quotes
	- Fallback regex extraction for label/reason with either single or double quotes
	"""
	if not isinstance(text, str):
		return None

	# Strip common code fences
	s = text.strip()
	if s.startswith("```"):
		# remove leading and trailing triple backticks blocks
		s = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", s, flags=re.MULTILINE)

	# Take JSON-looking slice
	if "{" in s and "}" in s:
		start = s.find("{")
		end = s.rfind("}")
		if 0 <= start < end:
			candidate = s[start : end + 1]
			# Normalize smart quotes
			candidate = candidate.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
			candidate = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
			# Prevent bare newlines in JSON strings by leaving them (they're valid when escaped by API). We'll rely on json.loads.
			try:
				return json.loads(candidate)
			except Exception:
				pass

	# Regex fallback: extract label/reason even if single-quoted
	label_match = re.search(r"label\s*[:=]\s*['\"]([a-zA-Z_]+)['\"]", s)
	reason_match = re.search(r"reason\s*[:=]\s*['\"]([\s\S]{0,400}?)['\"]\s*[}\]]?", s)
	if label_match and reason_match:
		label = label_match.group(1)
		reason = reason_match.group(1)
		# Replace double quotes within reason to keep valid JSON
		reason = reason.replace('"', "'")
		return {"label": label, "reason": reason}

	return None


def call_llm(prompt: str, cfg: EvalConfig, is_prereq: bool = False) -> Tuple[str, str]:
	"""Call the chat model with strict JSON-only system instruction.

	Returns (label, reason) with defaults on any error.
	"""
	client = OpenAI()

	messages = [
		{"role": "system", "content": (
			"You are a strict evaluator. Return ONLY a single JSON object with keys label and reason. "
			"Do not include any text before or after the JSON. Do not use markdown code fences. "
			"Labels must be one of: prerequisite, correct, supports, incorrect. "
			"Keep reason under 200 chars, plain text; avoid double quotes inside the reason."
		)},
		{"role": "user", "content": prompt},
	]
	
	# Add small base delay between API calls to avoid rate limits
	time.sleep(0.25)
	
	text: str = "{}"
	max_retries = max(4, int(cfg.max_retries))  # Ensure at least 4 attempts total
	
	for attempt in range(max_retries):
		try:
			resp = client.chat.completions.create(
				model=cfg.model,
				temperature=float(cfg.temperature),
				messages=messages,
				max_tokens=200,  # Limit response length to keep responses compact
				response_format={"type": "json_object"},
			)
			text = resp.choices[0].message.content if getattr(resp, "choices", None) else "{}"
			
			# Strict parse first
			try:
				obj = json.loads(text)
			except Exception:
				obj = _sanitize_and_extract_json(text)
				if obj is None:
					# Backoff with jitter then retry
					if attempt < max_retries - 1:
						wait = min(8.0, (2 ** attempt) * 0.5) + random.uniform(0.0, 0.25)
						print(f"[retry {attempt + 1}] JSON parse failed, retrying...")
						time.sleep(wait)
						continue
					return "incorrect", "parse_error"

			# Validate object
			label = str(obj.get("label", "incorrect")).strip().lower()
			reason = str(obj.get("reason", ""))[:200]
			# Replace any stray newlines and double quotes inside reason to be safe
			reason = reason.replace('"', "'").replace("\n", " ").strip()
			
			allowed = {"prerequisite", "correct", "supports", "incorrect"}
			if label not in allowed:
				if attempt < max_retries - 1:
					wait = min(8.0, (2 ** attempt) * 0.5) + random.uniform(0.0, 0.25)
					print(f"[retry {attempt + 1}] invalid_label, retrying...")
					time.sleep(wait)
					continue
				return "incorrect", "invalid_label"
			
			return label, reason
				
		except Exception as e:
			if attempt < max_retries - 1:
				wait = min(8.0, (2 ** attempt) * 0.75) + random.uniform(0.0, 0.5)
				print(f"[retry {attempt + 1}] API error: {str(e)[:100]}, retrying...")
				time.sleep(wait)
				continue
			return "incorrect", "api_error"
	
	return "incorrect", "max_retries_exceeded"


# ----------------------------
# Progress reporting
# ----------------------------


def log_progress(processed: int, total: int, stats: Dict[str, int], started_at: float, is_prereq_mode: bool = False) -> None:
	"""Log progress with timing and accuracy info."""
	elapsed = max(0.0, time.time() - started_at)
	rate = (processed / elapsed) if elapsed > 0 else 0.0
	eta_sec = int((total - processed) / rate) if rate > 0 else 0
	
	if is_prereq_mode:
		# Prereq mode: show prerequisite, supports, incorrect
		prereq = stats.get("prerequisite", 0)
		supports = stats.get("supports", 0) 
		incorrect = stats.get("incorrect", 0)
		pct_prereq = (prereq / max(1, processed)) * 100
		print(f"[eval] {processed}/{total} edges | {prereq}prereq {supports}supports {incorrect}✗ ({pct_prereq:.1f}% prereq) | {elapsed:.1f}s elapsed | ETA {eta_sec//60}:{eta_sec%60:02d}", flush=True)
	else:
		# Content mode: show correct, supports, incorrect
		correct = stats.get("correct", 0)
		supports = stats.get("supports", 0)
		incorrect = stats.get("incorrect", 0)
		pct_correct = (correct / max(1, processed)) * 100
		print(f"[eval] {processed}/{total} edges | {correct}✓ {supports}supports {incorrect}✗ ({pct_correct:.1f}% correct) | {elapsed:.1f}s elapsed | ETA {eta_sec//60}:{eta_sec%60:02d}", flush=True)


# ----------------------------
# Edge judging
# ----------------------------


def judge_edges(edges_df: pd.DataFrame, lo_lookup: Dict[str, dict], content_lookup: Dict[str, dict], cfg: EvalConfig) -> List[dict]:
	"""Judge edges with relation-specific prompts. Defaults to incorrect on any issue."""
	results: List[dict] = []
	allowed_content_rels = {"explained_by", "exemplified_by", "practiced_by"}

	def mk_row_common() -> dict:
		return {
			"source_lo_id": None,
			"target_lo_id": None,
			"content_id": None,
			"relation": None,
			"llm_label": "incorrect",
			"llm_reason": "",
		}

	total = len(edges_df)
	if cfg.limit is not None:
		total = min(total, int(cfg.limit))
	
	# Detect mode from first edge
	first_rel = str(edges_df.iloc[0].get("relation", "")).strip() if len(edges_df) > 0 else ""
	is_prereq_mode = (first_rel == "prerequisite")
	
	print(f"[eval] Starting evaluation of {total} edges...")
	started_at = time.time()
	stats = {"prerequisite": 0, "correct": 0, "supports": 0, "incorrect": 0}

	count = 0
	for _, r in edges_df.iterrows():
		if cfg.limit is not None and count >= int(cfg.limit):
			break
		rel = str(r.get("relation") or "").strip()
		out = mk_row_common()
		out["relation"] = rel

		# Prerequisite LO->LO
		if rel == "prerequisite" and ("source_lo_id" in r and "target_lo_id" in r):
			src_id = str(r.get("source_lo_id") or "")
			tgt_id = str(r.get("target_lo_id") or "")
			out["source_lo_id"] = src_id
			out["target_lo_id"] = tgt_id
			src = lo_lookup.get(src_id)
			tgt = lo_lookup.get(tgt_id)
			if not src or not tgt:
				out["llm_label"], out["llm_reason"] = "incorrect", "missing_lo"
				stats["incorrect"] += 1
			else:
				prompt = build_prereq_prompt(src, tgt)
				label, reason = call_llm(prompt, cfg, is_prereq=True)
				out["llm_label"], out["llm_reason"] = label, reason
				stats[label] = stats.get(label, 0) + 1
			results.append(out)
			count += 1
			
			# Progress report every 10 edges or at the end
			if count % 10 == 0 or count == total:
				log_progress(count, total, stats, started_at, is_prereq_mode=True)
			continue

		# Content links LO->Content
		if rel in allowed_content_rels and ("source_lo_id" in r and "target_content_id" in r):
			src_id = str(r.get("source_lo_id") or "")
			cid = str(r.get("target_content_id") or "")
			out["source_lo_id"] = src_id
			out["content_id"] = cid
			lo_obj = lo_lookup.get(src_id)
			content_obj = content_lookup.get(cid)
			if not lo_obj or not content_obj:
				out["llm_label"], out["llm_reason"] = "incorrect", "missing_lo_or_content"
				stats["incorrect"] += 1
			else:
				prompt = build_content_prompt(rel, lo_obj, content_obj)
				label, reason = call_llm(prompt, cfg, is_prereq=False)
				out["llm_label"], out["llm_reason"] = label, reason
				stats[label] = stats.get(label, 0) + 1
			results.append(out)
			count += 1
			
			# Progress report every 10 edges or at the end
			if count % 10 == 0 or count == total:
				log_progress(count, total, stats, started_at, is_prereq_mode=False)
			continue

		# Unsupported
		out["llm_label"], out["llm_reason"] = "incorrect", "unsupported_relation"
		stats["incorrect"] += 1
		results.append(out)
		count += 1
		
		# Progress report every 10 edges or at the end
		if count % 10 == 0 or count == total:
			log_progress(count, total, stats, started_at, is_prereq_mode)

	return results


# ----------------------------
# Summary
# ----------------------------


def summarize(results: List[dict]) -> Dict[str, dict]:
	"""Compute minimal summary per relation: n, correct, incorrect, p_correct."""
	from collections import defaultdict

	stats = defaultdict(lambda: {"n": 0, "prerequisite": 0, "correct": 0, "supports": 0, "incorrect": 0, "p_correct": 0.0, "p_prereq": 0.0})
	for row in results:
		rel = str(row.get("relation") or "")
		label = str(row.get("llm_label") or "incorrect")
		entry = stats[rel]
		entry["n"] += 1
		entry[label] = entry.get(label, 0) + 1
		
	for rel, entry in stats.items():
		n = max(1, int(entry["n"]))
		# For prereq relations, calculate prerequisite percentage
		if "prerequisite" in entry:
			prereq_count = entry.get("prerequisite", 0)
			entry["p_prereq"] = round(float(prereq_count) / float(n), 4)
		# For content relations, calculate correct percentage  
		if "correct" in entry:
			correct_count = entry.get("correct", 0)
			entry["p_correct"] = round(float(correct_count) / float(n), 4)
	return dict(stats)


# ----------------------------
# IO helpers
# ----------------------------


def ensure_parent(path: str) -> None:
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def read_edges_csv(path: str) -> pd.DataFrame:
	# tolerate both prereq and content schemas
	df = pd.read_csv(path)
	needed_any = [
		{"source_lo_id", "target_lo_id", "relation"},
		{"source_lo_id", "target_content_id", "relation"},
	]
	if not any(req.issubset(set(df.columns)) for req in needed_any):
		raise ValueError("edges-in CSV missing required columns")
	return df


def write_jsonl(path: str, rows: List[dict]) -> None:
	ensure_parent(path)
	with open(path, "w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ----------------------------
# CLI
# ----------------------------


def main(argv: Optional[Iterable[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Strict LLM edge evaluator (binary)")
	parser.add_argument("--lo-index", type=str, default="data/processed/lo_index.csv")
	parser.add_argument("--content-items", type=str, default="data/processed/content_items.csv")
	parser.add_argument("--edges-in", type=str, required=True)
	parser.add_argument("--jsonl-out", type=str, default="data/processed/llm_edge_checks.jsonl")
	parser.add_argument("--summary-out", type=str, default="data/processed/llm_edge_checks_summary.json")
	parser.add_argument("--model", type=str, default="gpt-4o-mini")
	parser.add_argument("--temperature", type=float, default=0.0)
	parser.add_argument("--max-retries", type=int, default=2)
	parser.add_argument("--limit", type=int, default=None)
	args = parser.parse_args(list(argv) if argv is not None else None)

	cfg = EvalConfig(
		model=args.model,
		temperature=float(args.temperature),
		max_retries=int(args.max_retries),
		input_lo_index=args.lo_index,
		input_content_items=args.content_items,
		input_edges=args.edges_in,
		jsonl_out=args.jsonl_out,
		summary_out=args.summary_out,
		limit=args.limit,
	)

	content_df = pd.read_csv(cfg.input_content_items)
	lo_lookup = load_lo_lookup(cfg.input_lo_index, content_df)
	content_lookup = load_content_lookup(cfg.input_content_items)
	edges_df = read_edges_csv(cfg.input_edges)

	rows = judge_edges(edges_df, lo_lookup, content_lookup, cfg)
	write_jsonl(cfg.jsonl_out, rows)
	ensure_parent(cfg.summary_out)
	with open(cfg.summary_out, "w", encoding="utf-8") as f:
		json.dump(summarize(rows), f, ensure_ascii=False, indent=2)
	print(f"Wrote {cfg.jsonl_out} ({len(rows)} rows)")
	print(f"Wrote {cfg.summary_out}")
	return 0


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main())