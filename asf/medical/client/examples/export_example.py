"""
Example script for exporting results using the Medical Research Synthesizer API client.

This script demonstrates how to use the Medical Research Synthesizer API client to export search results in various formats.
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
        
        # Get result ID
        result_id = search_response.data.get("result_id")
        logger.info(f"Search successful! Result ID: {result_id}")
        
        # Export to JSON
        logger.info("Exporting to JSON...")
        json_response = await client.export_results(
            format="json",
            result_id=result_id
        )
        
        if not json_response.success:
            logger.error(f"JSON export failed: {json_response.message}")
        else:
            logger.info(f"JSON export successful! File URL: {json_response.data.get('file_url')}")
        
        # Export to CSV
        logger.info("Exporting to CSV...")
        csv_response = await client.export_results(
            format="csv",
            result_id=result_id
        )
        
        if not csv_response.success:
            logger.error(f"CSV export failed: {csv_response.message}")
        else:
            logger.info(f"CSV export successful! File URL: {csv_response.data.get('file_url')}")
        
        # Export to Excel
        logger.info("Exporting to Excel...")
        excel_response = await client.export_results(
            format="excel",
            result_id=result_id
        )
        
        if not excel_response.success:
            logger.error(f"Excel export failed: {excel_response.message}")
        else:
            logger.info(f"Excel export successful! File URL: {excel_response.data.get('file_url')}")
        
        # Export to PDF
        logger.info("Exporting to PDF...")
        pdf_response = await client.export_results(
            format="pdf",
            result_id=result_id
        )
        
        if not pdf_response.success:
            logger.error(f"PDF export failed: {pdf_response.message}")
        else:
            logger.info(f"PDF export successful! File URL: {pdf_response.data.get('file_url')}")
        
        # Export with query instead of result ID
        logger.info("Exporting with query...")
        query_export_response = await client.export_results(
            format="json",
            query=query,
            max_results=max_results
        )
        
        if not query_export_response.success:
            logger.error(f"Query export failed: {query_export_response.message}")
        else:
            logger.info(f"Query export successful! File URL: {query_export_response.data.get('file_url')}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        # Close client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
