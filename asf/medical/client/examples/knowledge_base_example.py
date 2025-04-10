"""
Example script for managing knowledge bases using the Medical Research Synthesizer API client.

This script demonstrates how to use the Medical Research Synthesizer API client to create, list, update, and delete knowledge bases.
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
        
        # Create a knowledge base
        kb_name = "cardiovascular_statins"
        kb_query = "statin therapy cardiovascular"
        kb_update_schedule = "weekly"
        
        logger.info(f"Creating knowledge base '{kb_name}'...")
        kb_response = await client.create_knowledge_base(
            name=kb_name,
            query=kb_query,
            update_schedule=kb_update_schedule
        )
        
        if not kb_response.success:
            logger.error(f"Knowledge base creation failed: {kb_response.message}")
            return
        
        kb_id = kb_response.data.get("kb_id")
        logger.info(f"Knowledge base created! ID: {kb_id}")
        logger.info("")
        
        # List knowledge bases
        logger.info("Listing knowledge bases...")
        kb_list_response = await client.list_knowledge_bases()
        
        if not kb_list_response.success:
            logger.error(f"Failed to list knowledge bases: {kb_list_response.message}")
            return
        
        kbs = kb_list_response.data
        logger.info(f"Found {len(kbs)} knowledge bases:")
        for i, kb in enumerate(kbs, 1):
            logger.info(f"{i}. {kb.get('name')}")
            logger.info(f"   ID: {kb.get('kb_id')}")
            logger.info(f"   Query: {kb.get('query')}")
            logger.info(f"   Update schedule: {kb.get('update_schedule')}")
            logger.info(f"   Last updated: {kb.get('last_updated')}")
            logger.info(f"   Next update: {kb.get('next_update')}")
            logger.info("")
        
        # Get a knowledge base
        logger.info(f"Getting knowledge base '{kb_id}'...")
        kb_get_response = await client.get_knowledge_base(kb_id)
        
        if not kb_get_response.success:
            logger.error(f"Failed to get knowledge base: {kb_get_response.message}")
            return
        
        kb = kb_get_response.data
        logger.info(f"Knowledge base details:")
        logger.info(f"Name: {kb.get('name')}")
        logger.info(f"ID: {kb.get('kb_id')}")
        logger.info(f"Query: {kb.get('query')}")
        logger.info(f"Update schedule: {kb.get('update_schedule')}")
        logger.info(f"Last updated: {kb.get('last_updated')}")
        logger.info(f"Next update: {kb.get('next_update')}")
        logger.info("")
        
        # Update a knowledge base
        logger.info(f"Updating knowledge base '{kb_id}'...")
        kb_update_response = await client.update_knowledge_base(kb_id)
        
        if not kb_update_response.success:
            logger.error(f"Failed to update knowledge base: {kb_update_response.message}")
            return
        
        logger.info(f"Knowledge base update started!")
        logger.info(f"Status: {kb_update_response.data.get('status')}")
        logger.info("")
        
        # Delete a knowledge base
        logger.info(f"Deleting knowledge base '{kb_id}'...")
        kb_delete_response = await client.delete_knowledge_base(kb_id)
        
        if not kb_delete_response.success:
            logger.error(f"Failed to delete knowledge base: {kb_delete_response.message}")
            return
        
        logger.info(f"Knowledge base deleted!")
        logger.info(f"Status: {kb_delete_response.data.get('status')}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        # Close client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
