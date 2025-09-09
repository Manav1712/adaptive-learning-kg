#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os


def main(argv: list[str] | None = None) -> int:
    try:
        from pyvis.network import Network  # type: ignore
    except Exception as e:  # pragma: no cover - runtime check
        raise SystemExit(f"PyVis not installed. Run: pip install pyvis ({e})")

    parser = argparse.ArgumentParser(
        description="Agent Architecture Mind Map → Interactive HTML",
    )
    parser.add_argument("--out", default="data/processed/agent_mindmap.html")
    args = parser.parse_args(argv)

    net = Network(
        height="1000px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#ffffff",
    )
    # We'll use fixed coordinates for a clean, ByteByteGo-style layout

    # Palette
    COLORS = {
        # Pastel tones
        "center": "#C7BFE6",   # pastel violet
        "core": "#A7C7E7",     # pastel blue
        "support": "#B8E0D2",  # pastel green
        "storage": "#FFD8B1",  # pastel orange
        "scenario": "#D1C4E9", # pastel purple
        "link": "#C0C0C0",     # light gray links
    }

    # Center node
    net.add_node(
        "center",
        label="Adaptive Learning Platform",
        color=COLORS["center"],
        shape="star",
        size=40,
        font={"size": 22, "face": "Inter, -apple-system, system-ui, Segoe UI"},
        x=0,
        y=0,
        physics=False,
        fixed=True,
        title="Local-first adaptive learning system over a static math KG",
    )

    # ByteByteGo-style quadrant layout positions
    categories = [
        ("core", "Core Agents", (0, -400), COLORS["core"]),                # above (reduced whitespace)
        ("support", "Supporting Components", (640, 0), COLORS["support"]),  # right
        ("storage", "Data Storage", (0, 520), COLORS["storage"]),          # below
        ("scenario", "Scenarios", (-640, 0), COLORS["scenario"]),          # left
    ]

    # Child nodes per category (label, tooltip)
    core_agents = [
        (
            "Coach",
            "Session orchestrator: identifies LOs, manages context, topic switches",
        ),
        ("Retrieval Agent", "Finds target LO, prereqs, examples, exercises"),
        ("Tutor", "Delivers explanations and exercises"),
        ("Grader", "Checks answers (exact/rubric later)"),
        (
            "Mastery Estimator",
            "Updates mastery/struggles edges for personalization",
        ),
        ("Overlay Writer", "Persists personalization overlay to local JSON"),
    ]
    supporting = [
        ("Pack Constructor", "Builds teaching pack with citations"),
        ("Lesson Planner", "Selects LOs for review from KG + overlay"),
        ("Answer Composer", "GraphRAG-based unit overviews"),
    ]
    storage = [
        ("Static KG", "kg_nodes.json + kg_edges.json"),
        ("Embedding Index", "embeddings.json"),
        ("Session Logs", "One JSON per session"),
        ("Personalization Overlay", "overlay_{student_id}.json"),
        ("GraphRAG Summaries", "summaries.json"),
    ]
    scenarios = [
        ("First-time Session", "Coach → Retrieval → Pack → Tutor"),
        ("Continue Session", "Resume context; Retrieval refresh; Tutor"),
        ("Session Switch", "Close current; start new LO"),
        (
            "Grading & Feedback",
            "Grader → Mastery Estimator → Overlay Writer",
        ),
        ("Personalized Review", "Lesson Planner → Retrieval → Tutor"),
        ("Unit Overview", "Answer Composer → Retrieval → Tutor"),
    ]

    by_cat = {
        # Rounded shapes for a minimalist look
        "core": (core_agents, "ellipse"),
        "support": (supporting, "ellipse"),
        "storage": (storage, "ellipse"),
        "scenario": (scenarios, "ellipse"),
    }

    # Place category hubs (rounded, fixed positions)
    cat_positions: dict[str, tuple[int, int]] = {}
    for key, label, (x, y), color in categories:
        hub_id = f"cat:{key}"
        cat_positions[key] = (x, y)
        net.add_node(
            hub_id,
            label=label,
            color=color,
            shape="ellipse",
            size=30,
            font={"size": 18, "bold": True, "face": "Inter, -apple-system, system-ui, Segoe UI"},
            x=x,
            y=y,
            physics=False,
            fixed=True,
            title=(
                "Core Agents: Orchestrate sessions and student-facing flows"
                if key == "core" else
                "Supporting Components: Build packs, plan lessons, compose overviews"
                if key == "support" else
                "Data Storage: Files backing the local-first architecture"
                if key == "storage" else
                "Scenarios: Typical user flows through the system"
            ),
        )
        net.add_edge("center", hub_id, color=color, width=3, arrows="to")

    # Place children in tidy vertical lists around each category
    label_to_id: dict[str, str] = {}
    for key, (children, shape) in by_cat.items():
        hub_x, hub_y = cat_positions[key]
        n = len(children)
        spacing = 112  # reduced whitespace ~20%
        y_start = int(hub_y - (n - 1) * spacing / 2)
        x_offset = 0
        # Slightly push children away from the hub for each quadrant
        if key == "core":
            x_offset = 0
        elif key == "support":
            x_offset = 250
        elif key == "storage":
            x_offset = 0
        elif key == "scenario":
            x_offset = -250
        for i, (label, tip) in enumerate(children):
            cx = int(hub_x + x_offset)
            cy = int(y_start + i * spacing)
            node_id = f"{key}:{label}"
            label_to_id[label] = node_id
            # Build contextual tooltip
            context_why = (
                "Drives session flow and coaching"
                if key == "core" else
                "Enables content building and planning"
                if key == "support" else
                "Persists data for local-first operation"
                if key == "storage" else
                "Represents a common user interaction pattern"
            )
            net.add_node(
                node_id,
                label=label,
                title=f"{label}<br>{tip}<br><b>Why here:</b> {context_why}",
                shape=shape,
                color=COLORS[key],
                size=18,
                font={"size": 16, "face": "Inter, -apple-system, system-ui, Segoe UI"},
                x=cx,
                y=cy,
                physics=False,
                fixed=True,
            )
            net.add_edge(f"cat:{key}", node_id, color=COLORS[key], width=2, arrows="to")

    # Cross-links showing data/control flow (dashed)
    def link(a_label: str, b_label: str) -> None:
        if a_label in label_to_id and b_label in label_to_id:
            net.add_edge(
                label_to_id[a_label],
                label_to_id[b_label],
                color=COLORS["link"],
                width=2,
                arrows="to",
            )

    # Core learning flow
    link("Coach", "Retrieval Agent")
    link("Retrieval Agent", "Pack Constructor")
    link("Pack Constructor", "Tutor")

    # Assessment → personalization
    link("Grader", "Mastery Estimator")
    link("Mastery Estimator", "Overlay Writer")

    # Review planning and overviews
    link("Lesson Planner", "Retrieval Agent")
    link("Answer Composer", "Retrieval Agent")
    link("Answer Composer", "Tutor")

    # Storage interactions
    link("Coach", "Session Logs")
    link("Lesson Planner", "Personalization Overlay")
    link("Lesson Planner", "Static KG")
    link("Answer Composer", "GraphRAG Summaries")
    link("Retrieval Agent", "Embedding Index")
    link("Retrieval Agent", "Static KG")
    link("Tutor", "Session Logs")

    # Minimalist ByteByteGo-style theme
    net.set_options("""
    {
      "nodes": {
        "borderWidth": 1,
        "shape": "ellipse",
        "font": { "size": 16, "face": "Inter, -apple-system, system-ui, Segoe UI" }
      },
      "edges": {
        "smooth": { "type": "cubicBezier", "forceDirection": "horizontal", "roundness": 0.45 },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.9 } },
        "color": { "inherit": false }
      },
      "layout": { "improvedLayout": true },
      "physics": { "enabled": false },
      "interaction": { "hover": true, "tooltipDelay": 120, "multiselect": false, "dragView": true, "zoomView": true }
    }
    """)

    # Legend
    net.add_node(
        "legend",
        label="Legend",
        shape="box",
        color="#F5F5F5",
        x=720,
        y=-620,
        physics=False,
        fixed=True,
        font={"size": 16, "face": "Inter, -apple-system, system-ui, Segoe UI"},
        title="Pastel swatches indicate groups. Arrows show data/control direction.",
    )
    legend_items = [
        ("Core Agents", COLORS["core"]),
        ("Supporting", COLORS["support"]),
        ("Data Storage", COLORS["storage"]),
        ("Scenarios", COLORS["scenario"]),
    ]
    y0 = -590
    for i, (lab, col) in enumerate(legend_items):
        nid = f"legend:{lab}"
        net.add_node(
            nid,
            label=lab,
            shape="dot",
            color=col,
            x=720,
            y=y0 + i * 40,
            physics=False,
            fixed=True,
            font={"size": 14, "face": "Inter, -apple-system, system-ui, Segoe UI"},
            title=(
                "Orchestration & tutoring agents"
                if lab == "Core Agents" else
                "Pack building, lesson planning, summarization"
                if lab == "Supporting" else
                "Local-first files powering the platform"
                if lab == "Data Storage" else
                "Typical user flows across the platform"
            ),
        )
        net.add_edge("legend", nid, color="#BDBDBD", width=1)

    # Arrow direction sample in legend
    net.add_node("legend_dir_a", label="A", color="#DDDDDD", shape="dot", x=720, y=y0 + len(legend_items) * 40 + 20, physics=False, fixed=True,
                 font={"size": 12, "face": "Inter, -apple-system, system-ui, Segoe UI"})
    net.add_node("legend_dir_b", label="B", color="#DDDDDD", shape="dot", x=780, y=y0 + len(legend_items) * 40 + 20, physics=False, fixed=True,
                 font={"size": 12, "face": "Inter, -apple-system, system-ui, Segoe UI"})
    net.add_edge("legend_dir_a", "legend_dir_b", color="#BDBDBD", width=1, arrows="to", label="direction of flow")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    net.write_html(out_path, notebook=False)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())


