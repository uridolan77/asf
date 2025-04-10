"""
Example script for using the Medical Research Synthesizer API client.

This script demonstrates how to use the Medical Research Synthesizer API client.
"""

import os
import asyncio
import logging
from typing import Dict, Any
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
        
        # Get current user
        logger.info("Getting current user...")
        user_response = await client.get_current_user()
        
        if not user_response.success:
            logger.error(f"Failed to get current user: {user_response.message}")
            return
        
        logger.info(f"Current user: {user_response.data}")
        
        # Search for medical literature
        query = "statin therapy cardiovascular"
        logger.info(f"Searching for '{query}'...")
        search_response = await client.search(query, max_results=5)
        
        if not search_response.success:
            logger.error(f"Search failed: {search_response.message}")
            return
        
        logger.info(f"Search successful! Found {len(search_response.data.get('results', []))} results.")
        
        # Analyze contradictions
        logger.info(f"Analyzing contradictions for '{query}'...")
        contradiction_response = await client.analyze_contradictions(query, max_results=5)
        
        if not contradiction_response.success:
            logger.error(f"Contradiction analysis failed: {contradiction_response.message}")
            return
        
        logger.info(f"Contradiction analysis successful! Found {contradiction_response.data.get('contradictions_found', 0)} contradictions.")
        
        # Create a knowledge base
        kb_name = "cardiovascular_statins"
        logger.info(f"Creating knowledge base '{kb_name}'...")
        kb_response = await client.create_knowledge_base(kb_name, query)
        
        if not kb_response.success:
            logger.error(f"Knowledge base creation failed: {kb_response.message}")
            return
        
        logger.info(f"Knowledge base created! ID: {kb_response.data.get('kb_id')}")
        
        # List knowledge bases
        logger.info("Listing knowledge bases...")
        kb_list_response = await client.list_knowledge_bases()
        
        if not kb_list_response.success:
            logger.error(f"Failed to list knowledge bases: {kb_list_response.message}")
            return
        
        logger.info(f"Found {len(kb_list_response.data)} knowledge bases.")
        
        # Export search results
        logger.info("Exporting search results to JSON...")
        export_response = await client.export_results("json", query=query, max_results=5)
        
        if not export_response.success:
            logger.error(f"Export failed: {export_response.message}")
            return
        
        logger.info(f"Export successful! File URL: {export_response.data.get('file_url')}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        # Close client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
