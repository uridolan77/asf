"""
Example script for searching medical literature using the Medical Research Synthesizer API client.

This script demonstrates how to use the Medical Research Synthesizer API client to search for medical literature.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

from asf.medical.client.api_client import MedicalResearchSynthesizerClient

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main function."""
    # Create client
    client = MedicalResearchSynthesizerClient(
        base_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
        api_version=os.getenv("API_VERSION", "v1")
    )
    
    try:
        # Login
        email = os.getenv("API_EMAIL", "user@example.com")
        password = os.getenv("API_PASSWORD", "password")
        
        logger.info(f"Logging in as {email}...")
        login_response = await client.login(email, password)
        
        if not login_response.success:
            logger.error(f"Login failed: {login_response.message}")
            return
        
        logger.info("Login successful!")
        
        # Search for medical literature
        query = "statin therapy cardiovascular"
        max_results = 10
        
        logger.info(f"Searching for '{query}'...")
        search_response = await client.search(query, max_results=max_results)
        
        if not search_response.success:
            logger.error(f"Search failed: {search_response.message}")
            return
        
        # Process search results
        results = search_response.data.get("results", [])
        logger.info(f"Search successful! Found {len(results)} results.")
        
        # Print results
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. {result.get('title')}")
            logger.info(f"   Authors: {result.get('authors')}")
            logger.info(f"   Journal: {result.get('journal')}")
            logger.info(f"   Date: {result.get('publication_date')}")
            logger.info(f"   PMID: {result.get('pmid')}")
            logger.info(f"   DOI: {result.get('doi')}")
            logger.info("")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        # Close client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
