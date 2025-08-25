"""
BASELINE V3 EVALUATION: Ontology-Enforced Knowledge Graph Analysis

What this does:
- Evaluates the baseline_v3_ontology_enforced graph with enforced ontology and fact ratings
- Measures relationship constraint effectiveness vs noise
- Tests retrieval (rate/search-limited), contextual retrieval, and full-graph structure

Goal: Verify ontology enforcement reduced edge-type noise and improved structure for production readiness.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from zep_cloud.client import Zep


class KnowledgeGraphEvaluatorV3:
    def __init__(self, api_key: str):
        self.client = Zep(api_key=api_key)
        self.graph_id = "baseline_v3_ontology_enforced"

    def get_graph_statistics(self) -> Dict[str, Any]:
        try:
            graph_info = self.client.graph.get(graph_id=self.graph_id)
            nodes = self.client.graph.node.get_by_graph_id(graph_id=self.graph_id)
            edges = self.client.graph.edge.get_by_graph_id(graph_id=self.graph_id)
            return {
                "graph_id": self.graph_id,
                "node_count": len(nodes) if nodes else 0,
                "edge_count": len(edges) if edges else 0,
                "graph_info": graph_info,
            }
        except Exception as e:
            return {"error": str(e)}

    def test_basic_retrieval(self, test_queries: List[str]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for query in test_queries:
            try:
                nodes = self.client.graph.search(graph_id=self.graph_id, query=query, scope="nodes", limit=5)
                edges = self.client.graph.search(graph_id=self.graph_id, query=query, scope="edges", limit=5)
                results[query] = {
                    "nodes_found": len(nodes.nodes) if nodes.nodes else 0,
                    "edges_found": len(edges.edges) if edges.edges else 0,
                }
            except Exception as e:
                results[query] = {"error": str(e)}
        results["note"] = "Retrieval is rate/search-limited; not a measure of coverage."
        return results

    def analyze_relationship_constraints(self) -> Dict[str, Any]:
        try:
            all_edges = self.client.graph.edge.get_by_graph_id(graph_id=self.graph_id)
            target = {"PREREQUISITE_OF", "PART_OF", "ASSESSED_BY"}
            dist: Dict[str, int] = {}
            target_count = 0
            for e in all_edges:
                t = getattr(e, "name", "unknown")
                dist[t] = dist.get(t, 0) + 1
                if t in target:
                    target_count += 1
            total = len(all_edges)
            return {
                "total_relationships": total,
                "target_relationship_count": target_count,
                "noise_relationship_count": total - target_count,
                "constraint_effectiveness": (target_count / total) if total else 0.0,
                "relationship_distribution": dist,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        try:
            nodes = self.client.graph.node.get_by_graph_id(graph_id=self.graph_id)
            edges = self.client.graph.edge.get_by_graph_id(graph_id=self.graph_id)

            node_analysis = {
                "total_count": len(nodes),
                "labels_distribution": {},
                "summary_lengths": [],
                "metadata_coverage": {},
                "content_type_analysis": {},
            }
            for n in nodes:
                if hasattr(n, "labels") and n.labels:
                    for lbl in n.labels:
                        node_analysis["labels_distribution"][lbl] = node_analysis["labels_distribution"].get(lbl, 0) + 1
                if hasattr(n, "summary") and n.summary:
                    node_analysis["summary_lengths"].append(len(n.summary))
                if hasattr(n, "metadata") and n.metadata:
                    for k in n.metadata.keys():
                        node_analysis["metadata_coverage"][k] = node_analysis["metadata_coverage"].get(k, 0) + 1
                if hasattr(n, "summary") and n.summary:
                    s = n.summary.lower()
                    if "concept" in s or "definition" in s:
                        node_analysis["content_type_analysis"]["concept"] = node_analysis["content_type_analysis"].get("concept", 0) + 1
                    elif "example" in s or "instance" in s:
                        node_analysis["content_type_analysis"]["example"] = node_analysis["content_type_analysis"].get("example", 0) + 1
                    elif any(x in s for x in ["try", "practice", "exercise"]):
                        node_analysis["content_type_analysis"]["try_it"] = node_analysis["content_type_analysis"].get("try_it", 0) + 1
                    elif any(x in s for x in ["problem", "solve"]):
                        node_analysis["content_type_analysis"]["problem"] = node_analysis["content_type_analysis"].get("problem", 0) + 1
                    else:
                        node_analysis["content_type_analysis"]["other"] = node_analysis["content_type_analysis"].get("other", 0) + 1

            edge_analysis = {
                "total_count": len(edges),
                "relationship_types": {},
                "fact_lengths": [],
                "connection_patterns": {},
            }
            for e in edges:
                t = getattr(e, "name", "unknown")
                edge_analysis["relationship_types"][t] = edge_analysis["relationship_types"].get(t, 0) + 1
                if hasattr(e, "fact") and e.fact:
                    edge_analysis["fact_lengths"].append(len(e.fact))
                if hasattr(e, "source_node_uuid") and hasattr(e, "target_node_uuid"):
                    s = e.source_node_uuid; tg = e.target_node_uuid
                    edge_analysis["connection_patterns"][s] = edge_analysis["connection_patterns"].get(s, 0) + 1
                    edge_analysis["connection_patterns"][tg] = edge_analysis["connection_patterns"].get(tg, 0) + 1

            node_analysis["avg_summary_length"] = (
                sum(node_analysis["summary_lengths"]) / len(node_analysis["summary_lengths"]) if node_analysis["summary_lengths"] else 0
            )
            edge_analysis["avg_fact_length"] = (
                sum(edge_analysis["fact_lengths"]) / len(edge_analysis["fact_lengths"]) if edge_analysis["fact_lengths"] else 0
            )

            return {
                "node_analysis": node_analysis,
                "edge_analysis": edge_analysis,
                "graph_connectivity": {
                    "avg_connections_per_node": (len(edges) / len(nodes)) if nodes else 0,
                },
            }
        except Exception as e:
            return {"error": str(e)}

    def run_full_evaluation(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        results["statistics"] = self.get_graph_statistics()
        results["retrieval_tests"] = self.test_basic_retrieval(["derivative", "limit", "integration", "calculus", "mathematics"])
        results["relationship_constraints"] = self.analyze_relationship_constraints()
        results["comprehensive_analysis"] = self.get_comprehensive_analysis()
        return results


def demo_evaluation():
    load_dotenv()
    api_key = os.getenv("ZEP_API_KEY")
    if not api_key:
        print("Error: ZEP_API_KEY not found in environment variables")
        return
    evaluator = KnowledgeGraphEvaluatorV3(api_key)
    results = evaluator.run_full_evaluation()

    print("\n" + "="*60)
    print("BASELINE V3 KNOWLEDGE GRAPH EVALUATION RESULTS")
    print("="*60)

    if "statistics" in results:
        stats = results["statistics"]
        if "error" not in stats:
            print("\n📊 GRAPH STATISTICS")
            print(f"   Graph ID: {stats.get('graph_id', 'N/A')}")
            print(f"   Total Nodes: {stats.get('node_count', 'N/A'):,}")
            print(f"   Total Edges: {stats.get('edge_count', 'N/A'):,}")
        else:
            print(f"Statistics Error: {stats['error']}")

    if "retrieval_tests" in results:
        print("\n🔎 RETRIEVAL PERFORMANCE (rate/search-limited)")
        for q, r in results["retrieval_tests"].items():
            if q == "note":
                continue
            if "error" not in r:
                print(f"   • {q:12} → {r['nodes_found']} nodes, {r['edges_found']} edges")
            else:
                print(f"   • {q:12} → Error: {r['error']}")

    if "relationship_constraints" in results:
        c = results["relationship_constraints"]
        if "error" not in c:
            print("\n🔗 RELATIONSHIP CONSTRAINT ANALYSIS")
            print(f"   Total Relationships: {c.get('total_relationships', 0):,}")
            print(f"   Target Relationships: {c.get('target_relationship_count', 0):,}")
            print(f"   Noise Relationships: {c.get('noise_relationship_count', 0):,}")
            print(f"   Constraint Effectiveness: {c.get('constraint_effectiveness', 0):.1%}")
        else:
            print(f"Relationship Constraints Error: {c['error']}")

    if "comprehensive_analysis" in results:
        a = results["comprehensive_analysis"]
        if "error" not in a:
            print("\n🔬 COMPREHENSIVE GRAPH ANALYSIS (FULL GRAPH)")
            na = a.get("node_analysis", {})
            print(f"   Nodes: {na.get('total_count', 0):,}")
            ea = a.get("edge_analysis", {})
            print(f"   Edges: {ea.get('total_count', 0):,}")
            print(f"   Relationship Types: {len(ea.get('relationship_types', {}))}")
        else:
            print(f"Comprehensive Analysis Error: {a['error']}")


if __name__ == "__main__":
    demo_evaluation()


