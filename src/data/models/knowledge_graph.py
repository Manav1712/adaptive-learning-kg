from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class KGNode:
    """Basic knowledge graph node"""
    id: str
    type: str  # "learning_objective", "concept", "problem", "section"
    title: str
    content: str
    metadata: Dict[str, Any]

@dataclass  
class KGEdge:
    """Basic knowledge graph edge"""
    source: str
    target: str
    relation: str  # "PREREQUISITE_OF", "ASSESSED_BY", "BELONGS_TO"
    weight: float = 1.0
    metadata: Dict[str, Any] = None

@dataclass
class KnowledgeGraph:
    """Simple knowledge graph structure"""
    nodes: List[KGNode]
    edges: List[KGEdge]
    
    def get_node(self, node_id: str) -> Optional[KGNode]:
        """Get node by ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_neighbors(self, node_id: str) -> List[KGNode]:
        """Get connected nodes"""
        neighbors = []
        for edge in self.edges:
            if edge.source == node_id:
                neighbor = self.get_node(edge.target)
                if neighbor:
                    neighbors.append(neighbor)
        return neighbors
