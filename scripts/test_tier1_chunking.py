#!/usr/bin/env python3
"""
Test script for Tier 1 chunking strategy.

This script demonstrates and validates the basic functionality of the
simple subheading-based chunking approach.
"""
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from processing.chunking_strategies import (
    Tier1SimpleChunking, 
    ChunkingTier, 
    get_chunking_strategy
)


def test_basic_chunking():
    """Test basic chunking functionality with sample OpenStax content."""
    
    print("🧪 Testing Tier 1 Simple Chunking Strategy")
    print("=" * 50)
    
    # Sample OpenStax-style content
    sample_content = """Chapter 3 Derivatives

3.1 Defining the Derivative

The derivative of a function f at a point x is defined as the limit of the difference quotient.

Definition: The derivative of f(x) is f'(x) = lim(h→0) [f(x+h) - f(x)]/h

Example 1
Find the derivative of f(x) = x².
Solution: Using the definition, f'(x) = 2x.

3.2 Differentiation Rules

There are several rules that make finding derivatives easier.

Power Rule: If f(x) = x^n, then f'(x) = nx^(n-1).

Product Rule: If f(x) = g(x)h(x), then f'(x) = g'(x)h(x) + g(x)h'(x).

Example 2
Find the derivative of f(x) = x³ + 2x².
Solution: f'(x) = 3x² + 4x.

Problem 1
Use the power rule to find the derivative of f(x) = x⁴ - 3x² + 5x - 1.

3.3 Applications of Derivatives

Derivatives have many practical applications in physics and engineering.

Critical Points: Points where f'(x) = 0 or f'(x) is undefined.

Example 3
Find the critical points of f(x) = x³ - 3x².
Solution: f'(x) = 3x² - 6x = 3x(x - 2), so critical points are x = 0 and x = 2.
"""
    
    # Initialize chunker
    chunker = Tier1SimpleChunking(max_chunk_size=1000)
    
    # Perform chunking
    chunks = chunker.chunk_text(sample_content, "calculus_derivatives_test")
    
    # Display results
    print(f"📊 Generated {len(chunks)} chunks from {len(sample_content)} characters\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk.chunk_id}")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Size: {len(chunk.content)} characters")
        print(f"  Content preview: {chunk.content[:80].replace(chr(10), ' ')}...")
        
        # Show first chunk in full for validation
        if i == 1:
            print(f"\n  📄 Full content of first chunk:")
            print(f"  {chunk.content[:200]}...")
            print(f"\n  🔧 Zep episode format (current API):")
            episode = chunk.to_zep_episode()
            print(f"    Type: {episode['type']}")
            print(f"    Data length: {len(episode['data'])} characters")
            print(f"    Data preview: {episode['data'][:150]}...")
            print(f"\n  ✅ Ready for: client.graph.add_batch(episodes=[EpisodeData(data=..., type='text')])")
        
        print("-" * 40)
    
    print(f"\n✅ Tier 1 chunking test completed successfully!")
    return chunks


def test_factory_pattern():
    """Test the factory pattern for getting chunking strategies."""
    
    print("\n🏭 Testing Factory Pattern")
    print("=" * 30)
    
    # Test getting Tier 1 strategy via factory
    strategy = get_chunking_strategy(ChunkingTier.TIER_1_SIMPLE, max_chunk_size=500)
    
    print(f"✅ Factory created: {type(strategy).__name__}")
    print(f"   Max chunk size: {strategy.max_chunk_size}")
    
    # Test with small content
    test_content = "Chapter 1\n\n1.1 Introduction\n\nThis is a test section."
    chunks = strategy.chunk_text(test_content, "factory_test")
    
    print(f"✅ Factory strategy produced {len(chunks)} chunks")
    

def main():
    """Run all tests for Tier 1 chunking."""
    
    print("🚀 Starting Tier 1 Chunking Strategy Tests")
    print("=" * 60)
    
    try:
        # Test basic functionality
        chunks = test_basic_chunking()
        
        # Test factory pattern
        test_factory_pattern()
        
        print(f"\n🎉 All tests passed! Ready for Zep integration.")
        print(f"📈 Generated {len(chunks)} chunks ready for episode creation.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
