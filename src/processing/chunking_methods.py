
import csv
import json
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class CSVEpisode:
    """
    Represents a CSV row converted to a Zep episode.
    
    Attributes:
        lo_id: Learning objective ID
        content: The main content (problem + solution)
        episode_type: Type of session (concept, exercise, example, try-it)
        metadata: Additional metadata from CSV
        raw_content: Original JSON content for reference
    """
    lo_id: str
    content: str
    episode_type: str
    metadata: Dict[str, Any]
    raw_content: Dict[str, Any]
    
    def to_zep_episode(self) -> Dict[str, Any]:
        """
        Convert to Zep episode format.
        
        Returns:
            Dict containing EpisodeData for Zep's add_batch API
        """
        # Create a clean, readable content string
        content_parts = []
        
        # Add problem if it exists
        if "problem" in self.raw_content:
            content_parts.append(f"Problem: {self.raw_content['problem']}")
        
        # Add solution steps if they exist
        if "solution" in self.raw_content and "steps" in self.raw_content["solution"]:
            content_parts.append("\nSolution:")
            for i, step in enumerate(self.raw_content["solution"]["steps"], 1):
                content_parts.append(f"{i}. {step['step']}")
        
        # Combine all parts
        full_content = "\n\n".join(content_parts)
        
        # Add metadata as context
        metadata_text = f"[LO: {self.lo_id}, Type: {self.episode_type}, Unit: {self.metadata.get('unit', 'N/A')}, Chapter: {self.metadata.get('chapter', 'N/A')}]\n\n"
        final_content = metadata_text + full_content
        
        return {
            "data": final_content,
            "type": "text"
        }


class CSVToEpisodeConverter:
    """
    Converts CSV content to Zep episodes for knowledge graph ingestion.
    """
    
    def __init__(self, csv_file_path: str):
        """
        Initialize converter with CSV file path.
        
        Args:
            csv_file_path: Path to the CSV file
        """
        self.csv_file_path = csv_file_path
    
    def convert_csv_to_episodes(self) -> List[CSVEpisode]:
        """
        Convert all CSV rows to episodes.
        
        Returns:
            List of CSVEpisode objects
        """
        episodes = []
        
        with open(self.csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                episode = self._convert_row_to_episode(row)
                if episode:
                    episodes.append(episode)
        
        return episodes
    
    def _convert_row_to_episode(self, row: Dict[str, str]) -> CSVEpisode:
        """
        Convert a single CSV row to an episode.
        
        Args:
            row: Dictionary representing a CSV row
            
        Returns:
            CSVEpisode object
        """
        # Parse the raw_content JSON
        try:
            raw_content = json.loads(row['raw_content'])
        except json.JSONDecodeError:
            # If JSON parsing fails, use the raw string
            raw_content = {"problem": row['raw_content']}
        
        # Extract metadata
        metadata = {
            "book": row.get('book', ''),
            "learning_objective": row.get('learning_objective', ''),
            "unit": row.get('unit', ''),
            "chapter": row.get('chapter', '')
        }
        
        # Create episode
        episode = CSVEpisode(
            lo_id=row['lo_id'],
            content=row['raw_content'],  # Will be processed in to_zep_episode()
            episode_type=row['type'],
            metadata=metadata,
            raw_content=raw_content
        )
        
        return episode
    
    def preview_episodes(self, num_episodes: int = 3) -> None:
        """
        Preview the first few episodes for verification.
        
        Args:
            num_episodes: Number of episodes to preview
        """
        episodes = self.convert_csv_to_episodes()
        
        print(f"Total episodes found: {len(episodes)}")
        print(f"Previewing first {min(num_episodes, len(episodes))} episodes:\n")
        
        for i, episode in enumerate(episodes[:num_episodes]):
            print(f"Episode {i+1}:")
            print(f"  LO ID: {episode.lo_id}")
            print(f"  Type: {episode.episode_type}")
            print(f"  Unit: {episode.metadata['unit']}")
            print(f"  Chapter: {episode.metadata['chapter']}")
            print(f"  Content preview: {episode.content[:100]}...")
            print(f"  Zep format: {episode.to_zep_episode()}")
            print("-" * 50)


class ZepKnowledgeGraphBuilder:
    """
    Simple scaffold for adding episodes to Zep's knowledge graph.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Zep client.
        
        Args:
            api_key: Zep API key
        """
        try:
            from zep_cloud import Zep
            self.client = Zep(api_key=api_key)
        except ImportError:
            print("Error: zep-cloud package not installed. Run: pip install zep-cloud")
            self.client = None
    
    def add_single_episode(self, episode: CSVEpisode, user_id: str = None, graph_id: str = None) -> Dict[str, Any]:
        """
        Add a single episode to the knowledge graph.
        
        Args:
            episode: CSVEpisode to add
            user_id: Optional user ID for user-specific graph
            graph_id: Optional graph ID for group/shared graph
            
        Returns:
            Response from Zep API
        """
        if not self.client:
            print("Zep client not initialized")
            return {}
        
        episode_data = episode.to_zep_episode()
        
        try:
            if user_id:
                # Add to user-specific graph
                response = self.client.graph.add(
                    user_id=user_id,
                    type=episode_data["type"],
                    data=episode_data["data"]
                )
            elif graph_id:
                # Add to group/shared graph
                response = self.client.graph.add(
                    graph_id=graph_id,
                    type=episode_data["type"],
                    data=episode_data["data"]
                )
            else:
                print("Error: Must provide either user_id or graph_id")
                return {}
            
            print(f"Added episode {episode.lo_id} successfully")
            return response
            
        except Exception as e:
            print(f"Error adding episode {episode.lo_id}: {e}")
            return {}
    
    def add_batch_episodes(self, episodes: List[CSVEpisode], user_id: str = None, graph_id: str = None) -> Dict[str, Any]:
        """
        Add multiple episodes to the knowledge graph in batch.
        
        Args:
            episodes: List of CSVEpisodes to add
            user_id: Optional user ID for user-specific graph
            graph_id: Optional graph ID for group/shared graph
            
        Returns:
            Response from Zep API
        """
        if not self.client:
            print("Zep client not initialized")
            return {}
        
        # Convert episodes to Zep format
        episode_data_list = []
        for episode in episodes:
            episode_data = episode.to_zep_episode()
            episode_data_list.append({
                "data": episode_data["data"],
                "type": episode_data["type"]
            })
        
        try:
            if user_id:
                # Add to user-specific graph
                response = self.client.graph.add_batch(
                    user_id=user_id,
                    episodes=episode_data_list
                )
            elif graph_id:
                # Add to group/shared graph
                response = self.client.graph.add_batch(
                    graph_id=graph_id,
                    episodes=episode_data_list
                )
            else:
                print("Error: Must provide either user_id or graph_id")
                return {}
            
            print(f"Added {len(episodes)} episodes successfully")
            return response
            
        except Exception as e:
            print(f"Error adding batch episodes: {e}")
            return {}


def demo_csv_conversion():
    """
    Demonstration of CSV to episode conversion.
    """
    # Update this path to match your CSV file location
    csv_path = "data/raw/try_it_draft_contents.csv"
    
    try:
        converter = CSVToEpisodeConverter(csv_path)
        converter.preview_episodes(3)
    except FileNotFoundError:
        print(f"CSV file not found at: {csv_path}")
        print("Please update the csv_path in demo_csv_conversion()")


def demo_kg_building():
    """
    Demonstration of building knowledge graph from CSV episodes.
    Loads API key from environment variables for security.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("ZEP_API_KEY")
    
    if not api_key:
        print("Error: ZEP_API_KEY not found in environment variables")
        print("Please create a .env file with: ZEP_API_KEY=your_actual_key")
        return
    
    # Initialize KG builder
    kg_builder = ZepKnowledgeGraphBuilder(api_key)
    
    # Convert CSV to episodes
    csv_path = "data/raw/try_it_draft_contents.csv"
    converter = CSVToEpisodeConverter(csv_path)
    episodes = converter.convert_csv_to_episodes()
    
    print(f"Ready to add {len(episodes)} episodes to knowledge graph")
    print("API key loaded successfully from environment")
    
    # Test adding a few episodes to the knowledge graph
    print("\nTesting knowledge graph building...")
    
    try:
        # First, create a graph for our learning content
        print("Creating graph for calculus learning content...")
        graph = kg_builder.client.graph.create(
            graph_id="calculus-learning-content"
        )
        print(f"✅ Graph created successfully: {graph}")
        
        # Now add episodes to that graph
        print("\nAdding episodes to the knowledge graph...")
        result = kg_builder.add_batch_episodes(episodes[:3], graph_id="calculus-learning-content")
        print(f"Test result: {result}")
        print("✅ Successfully added episodes to knowledge graph!")
        
    except Exception as e:
        print(f"❌ Error testing knowledge graph: {e}")
        print("This might be expected if you don't have the right permissions or graph setup")


if __name__ == "__main__":
    demo_csv_conversion()
    print("\n" + "="*50 + "\n")
    demo_kg_building()
