"""
BASELINE V4: Raw stats evaluation for pruned/tight-schema graph
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any
from zep_cloud.client import Zep


GRAPH_ID = "baseline_v4_pruned_schema"


def get_raw_stats(client: Zep) -> Dict[str, Any]:
    try:
        nodes = client.graph.node.get_by_graph_id(graph_id=GRAPH_ID)
        edges = client.graph.edge.get_by_graph_id(graph_id=GRAPH_ID)
        rel_types: Dict[str, int] = {}
        for e in edges:
            name = getattr(e, "name", "unknown")
            rel_types[name] = rel_types.get(name, 0) + 1
        return {
            "graph_id": GRAPH_ID,
            "nodes": len(nodes),
            "edges": len(edges),
            "relationship_types": len(rel_types),
            "top_types": sorted(rel_types.items(), key=lambda x: x[1], reverse=True)[:15],
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    load_dotenv()
    api_key = os.getenv("ZEP_API_KEY")
    if not api_key:
        print("Error: ZEP_API_KEY not set"); return
    client = Zep(api_key=api_key)
    stats = get_raw_stats(client)
    print(stats)


if __name__ == "__main__":
    main()



