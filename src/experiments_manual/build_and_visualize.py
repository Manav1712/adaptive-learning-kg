"""
Build and Visualize Graph (Offline CSV → Interactive HTML)

- Reads edges from:
  - data/processed/edges_content.csv (LO → Content)
  - data/processed/edges_prereqs.csv (LO → LO)
- Reads node metadata from:
  - data/processed/lo_index.csv
  - data/processed/content_items.csv

Outputs:
- data/processed/graph_preview.html  (interactive PyVis network)

Optional args:
--edges-content, --edges-prereqs, --lo-index, --content-items, --out
--max-nodes: limit to first N nodes for a lighter preview
--focus-unit / --focus-chapter: filter by unit/chapter values
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Set

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def filter_by_scope(lo_df: pd.DataFrame, content_df: pd.DataFrame, unit: Optional[str], chapter: Optional[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if unit:
        lo_df = lo_df[lo_df["unit"].astype(str) == str(unit)].copy()
        content_df = content_df[content_df["unit"].astype(str) == str(unit)].copy()
    if chapter:
        lo_df = lo_df[lo_df["chapter"].astype(str) == str(chapter)].copy()
        content_df = content_df[content_df["chapter"].astype(str) == str(chapter)].copy()
    return lo_df, content_df


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build interactive HTML graph preview from CSV edges")
    parser.add_argument("--edges-content", default="data/processed/edges_content.csv")
    parser.add_argument("--edges-prereqs", default="data/processed/edges_prereqs.csv")
    parser.add_argument("--lo-index", default="data/processed/lo_index.csv")
    parser.add_argument("--content-items", default="data/processed/content_items.csv")
    parser.add_argument("--out", default="data/processed/graph_preview.html")
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--focus-unit", type=str, default=None)
    parser.add_argument("--focus-chapter", type=str, default=None)
    args = parser.parse_args(argv)

    lo_df = load_csv(args.lo_index)
    content_df = load_csv(args.content_items)
    lo_df, content_df = filter_by_scope(lo_df, content_df, args.focus_unit, args.focus_chapter)

    # Edges (optional if missing)
    edges_content = pd.read_csv(args.edges_content) if os.path.exists(args.edges_content) else pd.DataFrame()
    edges_prereqs = pd.read_csv(args.edges_prereqs) if os.path.exists(args.edges_prereqs) else pd.DataFrame()

    # Build node maps for labels
    lo_label = lo_df.set_index("lo_id")["learning_objective"].astype(str).to_dict()
    lo_unit = lo_df.set_index("lo_id")["unit"].astype(str).to_dict()
    lo_chapter = lo_df.set_index("lo_id")["chapter"].astype(str).to_dict()
    lo_book = lo_df.set_index("lo_id")["book"].astype(str).to_dict()

    content_type_map = content_df.set_index("content_id")["content_type"].astype(str).to_dict()
    content_lo_parent = content_df.set_index("content_id")["lo_id_parent"].astype(str).to_dict() if "lo_id_parent" in content_df.columns else {}
    content_text_map = content_df.set_index("content_id")["text"].astype(str).to_dict()
    content_unit = content_df.set_index("content_id")["unit"].astype(str).to_dict()
    content_chapter = content_df.set_index("content_id")["chapter"].astype(str).to_dict()

    # Optional limit on nodes
    keep_nodes: Set[str] = set()
    if args.max_nodes is not None and args.max_nodes > 0:
        # Seed with first N LOs and their content
        seed_los = [str(x) for x in lo_df["lo_id"].astype(str).head(args.max_nodes).tolist()]
        keep_nodes.update(seed_los)
    else:
        keep_nodes.update(str(x) for x in lo_df["lo_id"].astype(str).tolist())

    # Collect edges while respecting keep_nodes if max_nodes set
    def _edge_ok(src: str, tgt: str) -> bool:
        if args.max_nodes is None or args.max_nodes <= 0:
            return True
        return (src in keep_nodes) or (tgt in keep_nodes)

    # Create PyVis network
    try:
        from pyvis.network import Network  # type: ignore
    except Exception as e:
        raise SystemExit(f"PyVis not installed: pip install pyvis ({e})")

    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    def _truncate(text: str, length: int = 80) -> str:
        t = (text or "").replace("\n", " ").strip()
        return t if len(t) <= length else t[: length - 1] + "…"

    # Add LO nodes
    for lo_id in keep_nodes:
        label = f"LO {lo_id}\n" + _truncate(lo_label.get(lo_id, f"LO {lo_id}"), 40)
        title = (
            f"LO {lo_id}<br>Objective: {lo_label.get(lo_id, '')}<br>"
            f"Unit: {lo_unit.get(lo_id, '')} | Chapter: {lo_chapter.get(lo_id, '')}<br>"
            f"Book: {lo_book.get(lo_id, '')}"
        )
        net.add_node(f"lo:{lo_id}", label=label, title=title, color="#4F86F7", shape="dot")

    # Add content nodes (filtered by scope)
    content_ids = set(content_df["content_id"].astype(str).tolist())
    for cid in content_ids:
        ctype = (content_type_map.get(cid, "content") or "content").lower()
        shape = "box" if ctype == "concept" else ("ellipse" if ctype == "example" else "diamond")
        title = (
            f"Content {cid}<br>Type: {ctype}<br>"
            f"Parent LO: {content_lo_parent.get(cid, '')}<br>"
            f"Unit: {content_unit.get(cid, '')} | Chapter: {content_chapter.get(cid, '')}<br>"
            f"Text: {_truncate(content_text_map.get(cid, ''), 240)}"
        )
        net.add_node(
            f"content:{cid}",
            label=f"{ctype}\n{cid}",
            title=title,
            color="#7CB342",
            shape=shape,
        )

    # Add content edges
    if not edges_content.empty:
        for _, r in edges_content.iterrows():
            src = f"lo:{str(r['source_lo_id'])}"
            tgt = f"content:{str(r['target_content_id'])}"
            if not _edge_ok(str(r['source_lo_id']), str(r['target_content_id'])):
                continue
            score = float(r.get("score", 0.0))
            rel = str(r.get("relation", "explained_by"))
            net.add_edge(src, tgt, title=f"{rel} | {score:.2f}", label=rel, color="#78909C")

    # Add prereq edges
    if not edges_prereqs.empty:
        for _, r in edges_prereqs.iterrows():
            src = f"lo:{str(r['source_lo_id'])}"
            tgt = f"lo:{str(r['target_lo_id'])}"
            if not _edge_ok(str(r['source_lo_id']), str(r['target_lo_id'])):
                continue
            score = float(r.get("score", 0.0))
            rel = str(r.get("relation", "prerequisite"))
            net.add_edge(src, tgt, title=f"{rel} | {score:.2f}", label=rel, color="#FF7043")

    # Add a simple legend in the corner (fixed nodes and labeled edges)
    # Nodes
    net.add_node("legend_lo", label="LO (blue dot)", color="#4F86F7", shape="dot", x=-1200, y=-1000, fixed=True, physics=False)
    net.add_node("legend_content", label="Content (green)", color="#7CB342", shape="box", x=-1200, y=-950, fixed=True, physics=False)
    # Edges legend
    net.add_node("legend_a1", label="", color="#CCCCCC", shape="dot", x=-1200, y=-900, fixed=True, physics=False)
    net.add_node("legend_a2", label="", color="#CCCCCC", shape="dot", x=-1100, y=-900, fixed=True, physics=False)
    net.add_edge("legend_a1", "legend_a2", label="explained_by", color="#78909C")
    net.add_node("legend_b1", label="", color="#CCCCCC", shape="dot", x=-1200, y=-860, fixed=True, physics=False)
    net.add_node("legend_b2", label="", color="#CCCCCC", shape="dot", x=-1100, y=-860, fixed=True, physics=False)
    net.add_edge("legend_b1", "legend_b2", label="prerequisite", color="#FF7043")

    # Ensure output directory
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Write HTML without trying to open/notebook context to avoid template issues
    net.write_html(out_path, notebook=False)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
