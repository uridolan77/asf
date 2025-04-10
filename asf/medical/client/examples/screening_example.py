"""
Example script for screening articles using the Medical Research Synthesizer API client.

This script demonstrates how to use the Medical Research Synthesizer API client to screen articles according to PRISMA guidelines and assess risk of bias.
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
        
        # Screen articles according to PRISMA guidelines
        query = "statin therapy cardiovascular"
        max_results = 20
        stage = "screening"
        criteria = {
            "include": ["randomized controlled trial", "cardiovascular outcomes"],
            "exclude": ["animal study", "in vitro"]
        }
        
        logger.info(f"Screening articles for '{query}'...")
        screening_response = await client.screen_articles(
            query=query,
            max_results=max_results,
            stage=stage,
            criteria=criteria
        )
        
        if not screening_response.success:
            logger.error(f"Screening failed: {screening_response.message}")
            return
        
        # Process screening results
        data = screening_response.data
        results = data.get("results", [])
        
        logger.info(f"Screening successful!")
        logger.info(f"Query: {data.get('query')}")
        logger.info(f"Stage: {data.get('stage')}")
        logger.info(f"Total articles: {data.get('total_articles')}")
        logger.info(f"Included: {data.get('included')}")
        logger.info(f"Excluded: {data.get('excluded')}")
        logger.info(f"Uncertain: {data.get('uncertain')}")
        logger.info("")
        
        # Print screening results
        logger.info("Screening results:")
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. {result.get('title')}")
            logger.info(f"   Decision: {result.get('decision')}")
            logger.info(f"   Reason: {result.get('reason')}")
            logger.info(f"   Confidence: {result.get('confidence')}")
            logger.info("")
        
        # Assess risk of bias
        domains = [
            "randomization",
            "blinding",
            "allocation_concealment",
            "sample_size",
            "attrition"
        ]
        
        logger.info(f"Assessing risk of bias for '{query}'...")
        bias_response = await client.assess_bias(
            query=query,
            max_results=max_results,
            domains=domains
        )
        
        if not bias_response.success:
            logger.error(f"Bias assessment failed: {bias_response.message}")
            return
        
        # Process bias assessment results
        data = bias_response.data
        results = data.get("results", [])
        
        logger.info(f"Bias assessment successful!")
        logger.info(f"Query: {data.get('query')}")
        logger.info(f"Total articles: {data.get('total_articles')}")
        logger.info(f"Low risk: {data.get('low_risk')}")
        logger.info(f"Moderate risk: {data.get('moderate_risk')}")
        logger.info(f"High risk: {data.get('high_risk')}")
        logger.info(f"Unclear risk: {data.get('unclear_risk')}")
        logger.info("")
        
        # Print bias assessment results
        logger.info("Bias assessment results:")
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. {result.get('title')}")
            logger.info(f"   Overall risk: {result.get('assessment', {}).get('overall', {}).get('risk')}")
            logger.info(f"   Randomization: {result.get('assessment', {}).get('randomization', {}).get('risk')}")
            logger.info(f"   Blinding: {result.get('assessment', {}).get('blinding', {}).get('risk')}")
            logger.info(f"   Allocation concealment: {result.get('assessment', {}).get('allocation_concealment', {}).get('risk')}")
            logger.info(f"   Sample size: {result.get('assessment', {}).get('sample_size', {}).get('risk')}")
            logger.info(f"   Attrition: {result.get('assessment', {}).get('attrition', {}).get('risk')}")
            logger.info("")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        # Close client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
