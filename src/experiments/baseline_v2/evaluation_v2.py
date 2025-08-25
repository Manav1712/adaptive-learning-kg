"""
BASELINE V2 EVALUATION: Knowledge Graph Analysis

What this does:
- Evaluates the baseline_v2_experiment graph with schema hints and relationship constraints
- Compares entity classification quality vs baseline_v1
- Analyzes relationship type distribution and constraint effectiveness
- Tests retrieval performance with improved graph structure

Key differences from baseline_v1 evaluation:
- Targets baseline_v2_experiment graph instead of calculus-learning-content
- Added relationship constraint analysis to verify schema hint effectiveness
- Enhanced content type classification analysis
- Focused evaluation on schema hint impact

Goal: Measure improvements in entity classification and relationship quality from schema hints.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from zep_cloud.client import Zep


class KnowledgeGraphEvaluatorV2:
    """
    Evaluation framework for baseline_v2 knowledge graph with schema constraints.
    """
    
    def __init__(self, api_key: str):
        self.client = Zep(api_key=api_key)
        self.graph_id = "baseline_v2_experiment"  # <-- experiment-specific graph
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the baseline_v2 knowledge graph."""
        try:
            graph_info = self.client.graph.get(graph_id=self.graph_id)
            nodes_response = self.client.graph.node.get_by_graph_id(graph_id=self.graph_id)
            node_count = len(nodes_response) if nodes_response else 0
            edges_response = self.client.graph.edge.get_by_graph_id(graph_id=self.graph_id)
            edge_count = len(edges_response) if edges_response else 0
            
            return {
                "graph_id": self.graph_id,
                "node_count": node_count,
                "edge_count": edge_count,
                "graph_info": graph_info
            }
        except Exception as e:
            print(f"Error getting graph statistics: {e}")
            return {}
    
    def test_basic_retrieval(self, test_queries: List[str]) -> Dict[str, Any]:
        """Test basic retrieval capabilities with sample queries."""
        results = {}
        for query in test_queries:
            try:
                node_results = self.client.graph.search(
                    graph_id=self.graph_id, query=query, scope="nodes", limit=5
                )
                edge_results = self.client.graph.search(
                    graph_id=self.graph_id, query=query, scope="edges", limit=5
                )
                results[query] = {
                    "nodes_found": len(node_results.nodes) if node_results.nodes else 0,
                    "edges_found": len(edge_results.edges) if edge_results.edges else 0,
                    "node_results": node_results.nodes[:3] if node_results.nodes else [],
                    "edge_results": edge_results.edges[:3] if edge_results.edges else []
                }
            except Exception as e:
                results[query] = {"error": str(e)}
        return results
    
    def test_learning_objective_coverage(self) -> Dict[str, Any]:
        """Test how well learning objectives are represented in the graph."""
        try:
            lo_results = self.client.graph.search(
                graph_id=self.graph_id, query="learning objective", scope="nodes", limit=50
            )
            content_types = {}; units = {}; chapters = {}
            if lo_results.nodes:
                for node in lo_results.nodes:
                    if hasattr(node, 'summary') and node.summary:
                        summary = node.summary.lower()
                        if 'concept' in summary: content_types['concept'] = content_types.get('concept', 0) + 1
                        elif 'example' in summary: content_types['example'] = content_types.get('example', 0) + 1
                        elif 'try' in summary: content_types['try_it'] = content_types.get('try_it', 0) + 1
                        elif 'exercise' in summary: content_types['exercise'] = content_types.get('exercise', 0) + 1
                    if hasattr(node, 'metadata') and node.metadata:
                        metadata = node.metadata
                        if 'unit' in metadata: units[metadata['unit']] = units.get(metadata['unit'], 0) + 1
                        if 'chapter' in metadata: chapters[metadata['chapter']] = chapters.get(metadata['chapter'], 0) + 1
            return {
                "total_nodes_found": len(lo_results.nodes) if lo_results.nodes else 0,
                "content_type_distribution": content_types,
                "unit_coverage": units,
                "chapter_coverage": chapters
            }
        except Exception as e: 
            return {"error": str(e)}
    
    def test_contextual_retrieval(self, lo_id: str) -> Dict[str, Any]:
        """Test contextual retrieval for a specific learning objective."""
        try:
            lo_query = f"LO {lo_id}"
            node_results = self.client.graph.search(
                graph_id=self.graph_id, query=lo_query, scope="nodes", limit=10
            )
            edge_results = self.client.graph.search(
                graph_id=self.graph_id, query=lo_query, scope="edges", limit=10
            )
            return {
                "lo_id": lo_id,
                "nodes_found": len(node_results.nodes) if node_results.nodes else 0,
                "edges_found": len(edge_results.edges) if edge_results.edges else 0,
                "related_content": node_results.nodes[:5] if node_results.nodes else [],
                "relationships": edge_results.edges[:5] if edge_results.edges else []
            }
        except Exception as e: 
            return {"error": str(e)}
    
    def analyze_relationship_constraints(self) -> Dict[str, Any]:
        """
        NEW: Analyze if schema hints effectively constrained relationships.
        Check for presence of target relationships vs noise.
        """
        try:
            print("   Analyzing relationship constraint effectiveness...")
            all_edges = self.client.graph.edge.get_by_graph_id(graph_id=self.graph_id)
            
            # Target relationships from schema hints
            target_relationships = {"PREREQUISITE_OF", "PART_OF", "ASSESSED_BY"}
            
            relationship_analysis = {
                "total_relationships": len(all_edges),
                "target_relationship_count": 0,
                "noise_relationship_count": 0,
                "relationship_distribution": {},
                "constraint_effectiveness": 0.0
            }
            
            for edge in all_edges:
                edge_type = getattr(edge, 'name', 'unknown')
                relationship_analysis["relationship_distribution"][edge_type] = \
                    relationship_analysis["relationship_distribution"].get(edge_type, 0) + 1
                
                if edge_type in target_relationships:
                    relationship_analysis["target_relationship_count"] += 1
                else:
                    relationship_analysis["noise_relationship_count"] += 1
            
            # Calculate constraint effectiveness (% of target relationships)
            if len(all_edges) > 0:
                relationship_analysis["constraint_effectiveness"] = \
                    relationship_analysis["target_relationship_count"] / len(all_edges)
            
            return relationship_analysis
        except Exception as e:
            return {"error": str(e)}
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis of the ENTIRE baseline_v2 graph."""
        try:
            print("   Fetching ALL nodes and edges...")
            all_nodes = self.client.graph.node.get_by_graph_id(graph_id=self.graph_id)
            all_edges = self.client.graph.edge.get_by_graph_id(graph_id=self.graph_id)
            print(f"   Analyzing {len(all_nodes)} nodes and {len(all_edges)} edges...")
            
            # Node analysis
            node_analysis = {
                "total_count": len(all_nodes),
                "labels_distribution": {},
                "summary_lengths": [],
                "metadata_coverage": {},
                "content_type_analysis": {}
            }
            
            for node in all_nodes:
                if hasattr(node, 'labels') and node.labels:
                    for label in node.labels:
                        node_analysis["labels_distribution"][label] = node_analysis["labels_distribution"].get(label, 0) + 1
                if hasattr(node, 'summary') and node.summary:
                    node_analysis["summary_lengths"].append(len(node.summary))
                if hasattr(node, 'metadata') and node.metadata:
                    for key in node.metadata.keys():
                        node_analysis["metadata_coverage"][key] = node_analysis["metadata_coverage"].get(key, 0) + 1
                
                # Enhanced content type detection for v2 comparison
                if hasattr(node, 'summary') and node.summary:
                    summary_lower = node.summary.lower()
                    if 'concept' in summary_lower or 'definition' in summary_lower:
                        node_analysis["content_type_analysis"]["concept"] = node_analysis["content_type_analysis"].get("concept", 0) + 1
                    elif 'example' in summary_lower or 'instance' in summary_lower:
                        node_analysis["content_type_analysis"]["example"] = node_analysis["content_type_analysis"].get("example", 0) + 1
                    elif 'try' in summary_lower or 'practice' in summary_lower or 'exercise' in summary_lower:
                        node_analysis["content_type_analysis"]["try_it"] = node_analysis["content_type_analysis"].get("try_it", 0) + 1
                    elif 'problem' in summary_lower or 'solve' in summary_lower:
                        node_analysis["content_type_analysis"]["problem"] = node_analysis["content_type_analysis"].get("problem", 0) + 1
                    else:
                        node_analysis["content_type_analysis"]["other"] = node_analysis["content_type_analysis"].get("other", 0) + 1
            
            # Edge analysis
            edge_analysis = {
                "total_count": len(all_edges),
                "relationship_types": {},
                "fact_lengths": [],
                "connection_patterns": {}
            }
            
            for edge in all_edges:
                edge_type = getattr(edge, 'name', 'unknown')
                edge_analysis["relationship_types"][edge_type] = edge_analysis["relationship_types"].get(edge_type, 0) + 1
                if hasattr(edge, 'fact') and edge.fact:
                    edge_analysis["fact_lengths"].append(len(edge.fact))
                if hasattr(edge, 'source_node_uuid') and hasattr(edge, 'target_node_uuid'):
                    source = edge.source_node_uuid; target = edge.target_node_uuid
                    if source not in edge_analysis["connection_patterns"]: edge_analysis["connection_patterns"][source] = 0
                    if target not in edge_analysis["connection_patterns"]: edge_analysis["connection_patterns"][target] = 0
                    edge_analysis["connection_patterns"][source] += 1; edge_analysis["connection_patterns"][target] += 1
            
            # Statistics
            node_analysis["avg_summary_length"] = sum(node_analysis["summary_lengths"]) / len(node_analysis["summary_lengths"]) if node_analysis["summary_lengths"] else 0
            edge_analysis["avg_fact_length"] = sum(edge_analysis["fact_lengths"]) / len(edge_analysis["fact_lengths"]) if edge_analysis["fact_lengths"] else 0
            sorted_connections = sorted(edge_analysis["connection_patterns"].items(), key=lambda x: x[1], reverse=True)
            edge_analysis["top_connected_nodes"] = sorted_connections[:10]
            
            return {
                "node_analysis": node_analysis,
                "edge_analysis": edge_analysis,
                "graph_connectivity": {
                    "avg_connections_per_node": len(all_edges) / len(all_nodes) if all_nodes else 0,
                    "isolated_nodes": len([n for n in all_nodes if getattr(n, 'uuid', '') not in edge_analysis["connection_patterns"]]),
                    "highly_connected_nodes": len([n for n in edge_analysis["connection_patterns"].values() if n > 5])
                }
            }
        except Exception as e: 
            return {"error": str(e)}
    
    def get_detailed_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of graph content and structure."""
        try:
            nodes_sample = self.client.graph.search(
                graph_id=self.graph_id, query="calculus", scope="nodes", limit=10
            )
            edges_sample = self.client.graph.search(
                graph_id=self.graph_id, query="calculus", scope="edges", limit=10
            )
            
            node_analysis = {}
            if nodes_sample.nodes:
                for i, node in enumerate(nodes_sample.nodes[:3]):
                    node_analysis[f"node_{i+1}"] = {
                        "uuid": getattr(node, 'uuid', 'N/A'),
                        "summary": node.summary[:100] + "..." if node.summary and len(node.summary) > 100 else node.summary,
                        "labels": getattr(node, 'labels', []),
                        "metadata_keys": list(node.metadata.keys()) if hasattr(node, 'metadata') and node.metadata else []
                    }
            
            edge_analysis = {}; edge_types = {}
            if edges_sample.edges:
                for edge in edges_sample.edges:
                    edge_type = getattr(edge, 'name', 'unknown')
                    edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
                for i, edge in enumerate(edges_sample.edges[:3]):
                    edge_analysis[f"edge_{i+1}"] = {
                        "uuid": getattr(edge, 'uuid', 'N/A'),
                        "fact": edge.fact[:100] + "..." if edge.fact and len(edge.fact) > 100 else edge.fact,
                        "name": getattr(edge, 'name', 'N/A'),
                        "source_node": getattr(edge, 'source_node_uuid', 'N/A'),
                        "target_node": getattr(edge, 'target_node_uuid', 'N/A')
                    }
            
            return {
                "sample_nodes": node_analysis,
                "sample_edges": edge_analysis,
                "edge_type_distribution": edge_types if edges_sample.edges else {}
            }
        except Exception as e: 
            return {"error": str(e)}
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run a complete evaluation of the baseline_v2 knowledge graph."""
        print("🔍 Starting Baseline V2 Knowledge Graph Evaluation...")
        results = {}
        
        print("📊 Getting graph statistics...")
        results["statistics"] = self.get_graph_statistics()
        
        print("🔎 Testing basic retrieval...")
        test_queries = ["derivative", "limit", "integration", "calculus", "mathematics"]
        results["retrieval_tests"] = self.test_basic_retrieval(test_queries)
        
        print("📚 Testing learning objective coverage...")
        results["lo_coverage"] = self.test_learning_objective_coverage()
        
        print("🎯 Testing contextual retrieval...")
        actual_lo_ids = ["1867", "1868", "1869", "1870", "1872"]
        results["contextual_retrieval"] = {}
        for lo_id in actual_lo_ids:
            results["contextual_retrieval"][lo_id] = self.test_contextual_retrieval(lo_id)
        
        print("🔗 Analyzing relationship constraints...")
        results["relationship_constraints"] = self.analyze_relationship_constraints()
        
        print("🔬 Getting comprehensive graph analysis...")
        results["comprehensive_analysis"] = self.get_comprehensive_analysis()
        
        print("📋 Getting sample analysis...")
        results["sample_analysis"] = self.get_detailed_analysis()
        
        print("✅ Baseline V2 Evaluation complete!")
        return results


def demo_evaluation():
    """Run baseline_v2 evaluation and display results."""
    load_dotenv()
    api_key = os.getenv("ZEP_API_KEY")
    if not api_key:
        print("Error: ZEP_API_KEY not found in environment variables")
        return
    
    evaluator = KnowledgeGraphEvaluatorV2(api_key)
    results = evaluator.run_full_evaluation()
    
    print("\n" + "="*60)
    print("BASELINE V2 KNOWLEDGE GRAPH EVALUATION RESULTS")
    print("="*60)
    
    # Statistics
    if "statistics" in results:
        stats = results["statistics"]
        print(f"\n📊 GRAPH STATISTICS")
        print(f"   Graph ID: {stats.get('graph_id', 'N/A')}")
        print(f"   Total Nodes: {stats.get('node_count', 'N/A'):,}")
        print(f"   Total Edges: {stats.get('edge_count', 'N/A'):,}")
        print(f"   Node-Edge Ratio: {stats.get('edge_count', 0) / stats.get('node_count', 1):.2f}")
    
    # Retrieval Tests
    if "retrieval_tests" in results:
        print(f"\n🔎 RETRIEVAL PERFORMANCE")
        print(f"   Test Queries:")
        for query, result in results["retrieval_tests"].items():
            if "error" not in result:
                print(f"     • {query:12} → {result['nodes_found']} nodes, {result['edges_found']} edges")
            else:
                print(f"     • {query:12} → Error: {result['error']}")
    
    # Relationship Constraints (NEW)
    if "relationship_constraints" in results:
        constraints = results["relationship_constraints"]
        if "error" not in constraints:
            print(f"\n🔗 RELATIONSHIP CONSTRAINT ANALYSIS")
            print(f"   Total Relationships: {constraints.get('total_relationships', 0):,}")
            print(f"   Target Relationships: {constraints.get('target_relationship_count', 0):,}")
            print(f"   Noise Relationships: {constraints.get('noise_relationship_count', 0):,}")
            print(f"   Constraint Effectiveness: {constraints.get('constraint_effectiveness', 0):.1%}")
            print(f"   Top Relationship Types:")
            rel_dist = constraints.get('relationship_distribution', {})
            for rel_type, count in sorted(rel_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"     • {rel_type:20}: {count:3}")
        else:
            print(f"\n🔗 Relationship Constraints: Error - {constraints['error']}")
    
    # LO Coverage
    if "lo_coverage" in results:
        coverage = results["lo_coverage"]
        if "error" not in coverage:
            print(f"\n📚 CONTENT COVERAGE")
            print(f"   Total LO Nodes: {coverage.get('total_nodes_found', 'N/A')}")
            print(f"   Content Distribution:")
            content_types = coverage.get('content_type_distribution', {})
            for content_type, count in content_types.items():
                print(f"     • {content_type:10}: {count:2} nodes")
            print(f"   Units Covered: {len(coverage.get('unit_coverage', {}))}")
            print(f"   Chapters Covered: {len(coverage.get('chapter_coverage', {}))}")
        else:
            print(f"\n📚 Content Coverage: Error - {coverage['error']}")
    
    # Contextual Retrieval
    if "contextual_retrieval" in results:
        print(f"\n🎯 CONTEXTUAL RETRIEVAL")
        print(f"   Learning Objective Tests:")
        for lo_id, result in results["contextual_retrieval"].items():
            if "error" not in result:
                print(f"     • {lo_id:8} → {result['nodes_found']} nodes, {result['edges_found']} edges")
            else:
                print(f"     • {lo_id:8} → Error: {result['error']}")
    
    # Comprehensive Analysis
    if "comprehensive_analysis" in results:
        analysis = results["comprehensive_analysis"]
        if "error" not in analysis:
            print(f"\n🔬 COMPREHENSIVE GRAPH ANALYSIS (FULL GRAPH)")
            node_analysis = analysis.get("node_analysis", {})
            print(f"\n   📊 NODE ANALYSIS ({node_analysis.get('total_count', 0):,} total)")
            print(f"     Average Summary Length: {node_analysis.get('avg_summary_length', 0):.0f} characters")
            
            if node_analysis.get("labels_distribution"):
                print(f"     Labels Distribution:")
                for label, count in sorted(node_analysis["labels_distribution"].items(), key=lambda x: x[1], reverse=True):
                    print(f"       • {label:15}: {count:3}")
            
            if node_analysis.get("content_type_analysis"):
                print(f"     Content Type Distribution:")
                for content_type, count in sorted(node_analysis["content_type_analysis"].items(), key=lambda x: x[1], reverse=True):
                    print(f"       • {content_type:10}: {count:3} nodes")
            
            if node_analysis.get("metadata_coverage"):
                print(f"     Metadata Coverage:")
                for key, count in sorted(node_analysis["metadata_coverage"].items(), key=lambda x: x[1], reverse=True):
                    print(f"       • {key:20}: {count:3} nodes")
            
            edge_analysis = analysis.get("edge_analysis", {})
            print(f"\n   🔗 EDGE ANALYSIS ({edge_analysis.get('total_count', 0):,} total)")
            print(f"     Average Fact Length: {edge_analysis.get('avg_fact_length', 0):.0f} characters")
            
            if edge_analysis.get("relationship_types"):
                print(f"     Relationship Types ({len(edge_analysis['relationship_types'])} total):")
                for rel_type, count in sorted(edge_analysis["relationship_types"].items(), key=lambda x: x[1], reverse=True):
                    print(f"       • {rel_type:25}: {count:3}")
            
            connectivity = analysis.get("graph_connectivity", {})
            print(f"\n   🌐 GRAPH CONNECTIVITY")
            print(f"     Average Connections per Node: {connectivity.get('avg_connections_per_node', 0):.2f}")
            print(f"     Isolated Nodes: {connectivity.get('isolated_nodes', 0)}")
            print(f"     Highly Connected Nodes (>5 connections): {connectivity.get('highly_connected_nodes', 0)}")
            
            if edge_analysis.get("top_connected_nodes"):
                print(f"     Top Connected Nodes:")
                for i, (node_id, connections) in enumerate(edge_analysis["top_connected_nodes"][:5]):
                    print(f"       {i+1}. Node {node_id[:8]}...: {connections} connections")
        else:
            print(f"\n🔬 Comprehensive Analysis: Error - {analysis['error']}")
    
    # Sample Analysis
    if "sample_analysis" in results:
        analysis = results["sample_analysis"]
        if "error" not in analysis:
            print(f"\n📋 SAMPLE ANALYSIS (Limited Search Results)")
            if analysis.get("edge_type_distribution"):
                edge_types = analysis["edge_type_distribution"]
                print(f"   Sample Relationship Types ({len(edge_types)} found):")
                for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"     • {edge_type:20}: {count}")
            
            if analysis.get("sample_nodes"):
                print(f"\n   Sample Entities:")
                for node_id, node_info in analysis["sample_nodes"].items():
                    summary = node_info['summary'] or "No summary available"
                    print(f"     • {summary}")
            
            if analysis.get("sample_edges"):
                print(f"\n   Sample Facts:")
                for edge_id, edge_info in analysis["sample_edges"].items():
                    fact = edge_info['fact'] or "No fact available"
                    edge_type = edge_info['name'] or "Unknown"
                    print(f"     • [{edge_type}] {fact}")
        else:
            print(f"\n📋 Sample Analysis: Error - {analysis['error']}")
    
    print(f"\n" + "="*60)


if __name__ == "__main__":
    demo_evaluation()
