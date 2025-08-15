"""
Simple Zep client for Phase 1
"""
from zep_python import ZepClient

class SimpleZepClient:
    """Basic Zep client for Phase 1"""
    
    def __init__(self, api_key: str):
        self.client = ZepClient(api_key=api_key)
    
    def add_content(self, content: str) -> None:
        """Add content to Zep for automatic KG construction"""
        # Basic implementation for Phase 1
        pass
