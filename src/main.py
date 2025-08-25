"""
Main application for Phase 1
"""
import os
from src.zep_client import SimpleZepClient
from src.processing.corpus_ingestion import OpenStaxIngestion

def main():
    """Phase 1 main function"""

    # Get Zep API key from environment
    api_key = os.getenv("ZEP_API_KEY")
    

    if not api_key:
        print("Please set ZEP_API_KEY environment variable")
        return
    

    # Initialize basic components for Phase 1
    zep_client = SimpleZepClient(api_key) 
    ingestion = OpenStaxIngestion()
    

    print("Phase 1: Basic setup complete")
    print("Ready to implement OpenStax ingestion and Zep integration")


if __name__ == "__main__":
    main()

