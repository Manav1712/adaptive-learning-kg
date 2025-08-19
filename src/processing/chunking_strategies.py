"""
Chunking strategies for OpenStax textbook content preparation for Zep ingestion.

This module implements the 3-tiered chunking approach:
- Tier 1: Simple subheading-based chunks
- Tier 2: Contextualized chunks with document context  
- Tier 3: Custom entity types with domain-specific schemas

Each strategy produces episodes ready for Zep's knowledge graph construction.
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ChunkingTier(Enum):
    """Enumeration of available chunking strategies"""
    TIER_1_SIMPLE = "tier_1_simple"
    TIER_2_CONTEXTUALIZED = "tier_2_contextualized" 
    TIER_3_CUSTOM_ENTITIES = "tier_3_custom_entities"


@dataclass
class TextChunk:
    """
    Represents a chunk of text ready for Zep episode creation.
    
    Attributes:
        content: The text content of the chunk
        chunk_id: Unique identifier for this chunk
        source_file: Original source file name
        chunk_type: Type of content (e.g., "section", "example", "definition")
        metadata: Additional metadata for the chunk
        context: Optional context information for contextualized chunking
    """
    content: str
    chunk_id: str
    source_file: str
    chunk_type: str
    metadata: Dict[str, Any]
    context: Optional[str] = None
    
    def to_zep_episode(self) -> Dict[str, Any]:
        """
        Convert chunk to Zep episode format (current API).
        
        Returns:
            Dict containing EpisodeData for Zep's add_batch API
        """
        episode_content = self.content
        if self.context:
            # Prepend context for better entity extraction (Anthropic's technique)
            episode_content = f"{self.context}\n\n{self.content}"
            
        # Add metadata as part of the content for Zep to extract
        metadata_text = f"[Source: {self.source_file}, Type: {self.chunk_type}, ID: {self.chunk_id}]\n\n"
        full_content = metadata_text + episode_content
        
        return {
            "data": full_content,
            "type": "text"
        }


class BaseChunkingStrategy:
    """Base class for all chunking strategies"""
    
    def __init__(self, max_chunk_size: int = 8000):
        """
        Initialize chunking strategy.
        
        Args:
            max_chunk_size: Maximum characters per chunk (Zep limit is 10,000)
        """
        self.max_chunk_size = max_chunk_size
        
    def chunk_text(self, text: str, source_file: str) -> List[TextChunk]:
        """
        Chunk text into episodes ready for Zep ingestion.
        
        Args:
            text: Input text to chunk
            source_file: Name of source file
            
        Returns:
            List of TextChunk objects
        """
        raise NotImplementedError("Subclasses must implement chunk_text")
        
    def _split_large_chunk(self, content: str, base_id: str, chunk_type: str) -> List[str]:
        """
        Split a chunk that exceeds max_chunk_size into smaller pieces.
        
        Args:
            content: Content to split
            base_id: Base identifier for sub-chunks
            chunk_type: Type of content being split
            
        Returns:
            List of content strings within size limits
        """
        if len(content) <= self.max_chunk_size:
            return [content]
            
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', content)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks


class Tier1SimpleChunking(BaseChunkingStrategy):
    """
    Tier 1: Simple subheading-based chunking strategy.
    
    Splits text based on obvious structural markers:
    - Chapter headings (e.g., "Chapter 1", "1.1", "1.2.1")
    - Section breaks
    - Example/Problem blocks
    
    This is the most straightforward approach for initial testing.
    """
    
    def __init__(self, max_chunk_size: int = 8000):
        super().__init__(max_chunk_size)
        
        # Regex patterns for detecting structural elements
        self.chapter_pattern = re.compile(r'^Chapter\s+\d+', re.IGNORECASE | re.MULTILINE)
        self.section_pattern = re.compile(r'^\d+\.\d+.*$', re.MULTILINE)
        self.subsection_pattern = re.compile(r'^\d+\.\d+\.\d+.*$', re.MULTILINE)
        self.example_pattern = re.compile(r'^Example\s+\d+', re.IGNORECASE | re.MULTILINE)
        self.problem_pattern = re.compile(r'^Problem\s+\d+', re.IGNORECASE | re.MULTILINE)
    
    def chunk_text(self, text: str, source_file: str) -> List[TextChunk]:
        """
        Chunk text using simple subheading detection.
        
        Args:
            text: Input text to chunk
            source_file: Name of source file
            
        Returns:
            List of TextChunk objects ready for Zep ingestion
        """
        chunks = []
        lines = text.split('\n')
        current_chunk = ""
        current_type = "content"
        chunk_counter = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Detect content type based on patterns
            detected_type = self._detect_content_type(line)
            
            # If we hit a new section or chunk is getting large, finalize current chunk
            if (detected_type != current_type and current_chunk) or \
               (len(current_chunk) + len(line) > self.max_chunk_size):
                
                if current_chunk.strip():
                    chunk_id = f"{source_file}_chunk_{chunk_counter:03d}"
                    
                    # Split if still too large
                    split_chunks = self._split_large_chunk(
                        current_chunk.strip(), chunk_id, current_type
                    )
                    
                    for j, split_content in enumerate(split_chunks):
                        final_chunk_id = f"{chunk_id}_{j}" if len(split_chunks) > 1 else chunk_id
                        
                        chunk = TextChunk(
                            content=split_content,
                            chunk_id=final_chunk_id,
                            source_file=source_file,
                            chunk_type=current_type,
                            metadata={
                                "line_start": i - current_chunk.count('\n'),
                                "line_end": i,
                                "character_count": len(split_content)
                            }
                        )
                        chunks.append(chunk)
                        chunk_counter += 1
                
                current_chunk = line + '\n'
                current_type = detected_type
            else:
                current_chunk += line + '\n'
        
        # Handle final chunk
        if current_chunk.strip():
            chunk_id = f"{source_file}_chunk_{chunk_counter:03d}"
            split_chunks = self._split_large_chunk(
                current_chunk.strip(), chunk_id, current_type
            )
            
            for j, split_content in enumerate(split_chunks):
                final_chunk_id = f"{chunk_id}_{j}" if len(split_chunks) > 1 else chunk_id
                
                chunk = TextChunk(
                    content=split_content,
                    chunk_id=final_chunk_id,
                    source_file=source_file,
                    chunk_type=current_type,
                    metadata={
                        "line_start": len(lines) - current_chunk.count('\n'),
                        "line_end": len(lines),
                        "character_count": len(split_content)
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def _detect_content_type(self, line: str) -> str:
        """
        Detect the type of content based on line patterns.
        
        Args:
            line: Text line to analyze
            
        Returns:
            String indicating content type
        """
        if self.chapter_pattern.match(line):
            return "chapter"
        elif self.section_pattern.match(line):
            return "section"
        elif self.subsection_pattern.match(line):
            return "subsection"
        elif self.example_pattern.match(line):
            return "example"
        elif self.problem_pattern.match(line):
            return "problem"
        else:
            return "content"


class Tier2ContextualizedChunking(BaseChunkingStrategy):
    """
    Tier 2: Contextualized chunking strategy.
    
    Uses Anthropic's contextual retrieval technique:
    - Prepends each chunk with document context
    - Maintains chapter/section hierarchy information
    - Better entity extraction through contextual awareness
    
    Placeholder for future implementation.
    """
    
    def chunk_text(self, text: str, source_file: str) -> List[TextChunk]:
        """Placeholder - to be implemented"""
        raise NotImplementedError("Tier 2 contextualized chunking not yet implemented")


class Tier3CustomEntitiesChunking(BaseChunkingStrategy):
    """
    Tier 3: Custom entity types chunking strategy.
    
    Defines domain-specific entity schemas for mathematics:
    - Mathematical concepts and definitions
    - Formulas and equations
    - Problem-solution pairs
    - Learning objectives and prerequisites
    
    Placeholder for future implementation.
    """
    
    def chunk_text(self, text: str, source_file: str) -> List[TextChunk]:
        """Placeholder - to be implemented"""
        raise NotImplementedError("Tier 3 custom entities chunking not yet implemented")


def get_chunking_strategy(tier: ChunkingTier, **kwargs) -> BaseChunkingStrategy:
    """
    Factory function to get the appropriate chunking strategy.
    
    Args:
        tier: The chunking tier to use
        **kwargs: Additional arguments for strategy initialization
        
    Returns:
        Initialized chunking strategy instance
    """
    if tier == ChunkingTier.TIER_1_SIMPLE:
        return Tier1SimpleChunking(**kwargs)
    elif tier == ChunkingTier.TIER_2_CONTEXTUALIZED:
        return Tier2ContextualizedChunking(**kwargs)
    elif tier == ChunkingTier.TIER_3_CUSTOM_ENTITIES:
        return Tier3CustomEntitiesChunking(**kwargs)
    else:
        raise ValueError(f"Unknown chunking tier: {tier}")


# Example usage and testing function
def demo_tier1_chunking():
    """
    Demonstration of Tier 1 chunking with sample OpenStax content.
    """
    sample_text = """Chapter 1 Functions and Graphs

1.1 Functions

A function is a rule that assigns each input exactly one output.

Example 1
Find the domain of f(x) = 1/x.
Solution: The domain is all real numbers except x = 0.

1.2 Linear Functions

A linear function has the form f(x) = mx + b.

Example 2
Graph y = 2x + 1.
Solution: This is a line with slope 2 and y-intercept 1.

Problem 1
Find the slope of the line through (1,2) and (3,8).
"""
    
    chunker = Tier1SimpleChunking(max_chunk_size=500)  # Small for demo
    chunks = chunker.chunk_text(sample_text, "calculus_v1_sample")
    
    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}: {chunk.chunk_id}")
        print(f"Type: {chunk.chunk_type}")
        print(f"Characters: {len(chunk.content)}")
        print(f"Content preview: {chunk.content[:100]}...")
        print(f"Zep episode format: {chunk.to_zep_episode()}")


if __name__ == "__main__":
    demo_tier1_chunking()
