"""
Example script for analyzing contradictions using the Medical Research Synthesizer API client.

This script demonstrates how to use the Medical Research Synthesizer API client to analyze contradictions in medical literature.
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
        
        # Analyze contradictions
        query = "statin therapy cardiovascular"
        max_results = 20
        threshold = 0.7
        use_biomedlm = True
        use_tsmixer = False
        use_lorentz = False
        
        logger.info(f"Analyzing contradictions for '{query}'...")
        contradiction_response = await client.analyze_contradictions(
            query=query,
            max_results=max_results,
            threshold=threshold,
            use_biomedlm=use_biomedlm,
            use_tsmixer=use_tsmixer,
            use_lorentz=use_lorentz
        )
        
        if not contradiction_response.success:
            logger.error(f"Contradiction analysis failed: {contradiction_response.message}")
            return
        
        # Process contradiction analysis results
        data = contradiction_response.data
        contradictions = data.get("contradictions", [])
        
        logger.info(f"Contradiction analysis successful!")
        logger.info(f"Query: {data.get('query')}")
        logger.info(f"Total articles: {data.get('total_articles')}")
        logger.info(f"Contradictions found: {data.get('contradictions_found')}")
        logger.info(f"Analysis ID: {data.get('analysis_id')}")
        logger.info("")
        
        # Print contradiction types
        contradiction_types = data.get("contradiction_types", {})
        logger.info("Contradiction types:")
        for contradiction_type, count in contradiction_types.items():
            logger.info(f"  {contradiction_type}: {count}")
        logger.info("")
        
        # Print contradictions
        logger.info("Contradictions:")
        for i, contradiction in enumerate(contradictions, 1):
            logger.info(f"{i}. Type: {contradiction.get('contradiction_type')}")
            logger.info(f"   Score: {contradiction.get('contradiction_score')}")
            logger.info(f"   Confidence: {contradiction.get('confidence')}")
            logger.info(f"   Explanation: {contradiction.get('explanation')}")
            logger.info(f"   Publication 1: {contradiction.get('publication1', {}).get('title')}")
            logger.info(f"   Publication 2: {contradiction.get('publication2', {}).get('title')}")
            logger.info("")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        # Close client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
