#!/usr/bin/env python
"""
Example usage of the Cochrane Library client.

This script demonstrates how to use the Cochrane Library client
to search for reviews, extract PICO elements, and access evidence ratings.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any, Optional, List

# Add the parent directory to the path to allow importing the client
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from .cochrane_client import CochraneClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def check_api_status(client: CochraneClient) -> None:
    """Check the status of the Cochrane Library API."""
    logger.info("Checking API status...")
    status = await client.check_api_status()
    logger.info(f"API Status: {status['status']}")
    if status['status'] == 'error':
        logger.error(f"API Error: {status['message']}")

async def search_example(client: CochraneClient) -> None:
    """Demonstrate searching for Cochrane reviews."""
    logger.info("Searching for reviews about diabetes...")
    results = await client.search("diabetes mellitus type 2", max_results=5)
    
    logger.info(f"Found {results['total_results']} results. Showing first {len(results['results'])}:")
    for i, result in enumerate(results['results'], 1):
        logger.info(f"{i}. {result.get('title', 'No title')} (ID: {result.get('id', 'Unknown')})")

async def pico_search_example(client: CochraneClient) -> None:
    """Demonstrate searching using PICO framework."""
    logger.info("Searching by PICO elements...")
    results = await client.search_by_pico(
        population="adults",
        intervention="statins",
        outcome="cardiovascular disease",
        max_results=5
    )
    
    logger.info(f"Found {results['total_results']} results with PICO search. Showing first {len(results['results'])}:")
    for i, result in enumerate(results['results'], 1):
        logger.info(f"{i}. {result.get('title', 'No title')}")

async def get_review_example(client: CochraneClient) -> None:
    """Demonstrate retrieving a specific review."""
    # Using a sample review ID - replace with a valid one in real usage
    review_id = "CD012345"  # Example ID
    
    try:
        logger.info(f"Getting review {review_id}...")
        review = await client.get_review(review_id)
        
        logger.info(f"Review Title: {review.get('title', 'No title')}")
        logger.info(f"Authors: {review.get('authors', 'No authors')}")
        logger.info(f"Publication Date: {review.get('publicationDate', 'Unknown')}")
    except Exception as e:
        logger.error(f"Failed to retrieve review: {str(e)}")

async def extract_pico_example(client: CochraneClient) -> None:
    """Demonstrate extracting PICO elements from a review."""
    # Using a sample review ID - replace with a valid one in real usage
    review_id = "CD012345"  # Example ID
    
    try:
        logger.info(f"Extracting PICO from review {review_id}...")
        pico = await client.extract_pico(review_id)
        
        logger.info("PICO Elements:")
        for element_type, elements in pico['pico'].items():
            logger.info(f"  {element_type.capitalize()}:")
            for element in elements:
                logger.info(f"    - {element.get('description', 'No description')} (confidence: {element.get('confidence', 0)})")
    except Exception as e:
        logger.error(f"Failed to extract PICO: {str(e)}")

async def extract_evidence_example(client: CochraneClient) -> None:
    """Demonstrate extracting evidence ratings from a review."""
    # Using a sample review ID - replace with a valid one in real usage
    review_id = "CD012345"  # Example ID
    
    try:
        logger.info(f"Extracting evidence ratings from review {review_id}...")
        evidence = await client.extract_evidence(review_id)
        
        logger.info("Evidence Certainty Ratings:")
        for rating in evidence['evidence']['certainty_ratings']:
            logger.info(f"  Grade: {rating.get('grade', 'Unknown')} - {rating.get('explanation', 'No explanation')}")
    except Exception as e:
        logger.error(f"Failed to extract evidence: {str(e)}")

async def get_topics_example(client: CochraneClient) -> None:
    """Demonstrate retrieving Cochrane Library topics."""
    logger.info("Getting Cochrane Library topics...")
    
    topics = await client.get_topics()
    logger.info(f"Found {len(topics)} topics:")
    
    for i, topic in enumerate(topics[:5], 1):  # Show only first 5 topics
        logger.info(f"{i}. {topic.get('name', 'No name')} ({topic.get('id', 'Unknown')})")

async def get_recent_reviews_example(client: CochraneClient) -> None:
    """Demonstrate retrieving recent reviews."""
    logger.info("Getting recent reviews...")
    
    results = await client.get_recent_reviews(count=5)
    logger.info(f"Found {len(results['results'])} recent reviews:")
    
    for i, result in enumerate(results['results'], 1):
        logger.info(f"{i}. {result.get('title', 'No title')} (Date: {result.get('publicationDate', 'Unknown')})")

async def get_full_review_example(client: CochraneClient) -> None:
    """Demonstrate retrieving a review with full content."""
    # Using a sample review ID - replace with a valid one in real usage
    review_id = "CD012345"  # Example ID
    
    try:
        logger.info(f"Getting full content for review {review_id}...")
        full_review = await client.get_review_with_full_content(review_id)
        
        logger.info(f"Review Title: {full_review['metadata'].get('title', 'No title')}")
        
        if 'abstract' in full_review['content']:
            abstract = full_review['content']['abstract']
            logger.info(f"Abstract: {abstract[:150]}..." if len(abstract) > 150 else abstract)
    except Exception as e:
        logger.error(f"Failed to retrieve full review: {str(e)}")

async def main() -> None:
    """Main function to run the examples."""
    # Initialize the client with conservative rate limiting
    client = CochraneClient(
        requests_per_second=1.0,  # Be conservative with rate limits
        burst_size=2,
        timeout=60.0,
        max_retries=3,
        use_cache=True,
        cache_ttl=3600  # 1 hour
    )
    
    try:
        logger.info("Starting Cochrane client examples...")
        
        # Check API status
        await check_api_status(client)
        
        # Run search examples
        await search_example(client)
        await pico_search_example(client)
        
        # Get topics
        await get_topics_example(client)
        
        # Get recent reviews
        await get_recent_reviews_example(client)
        
        # Note: The following examples use sample IDs that need to be replaced
        # with valid IDs for actual use
        # await get_review_example(client)
        # await extract_pico_example(client)
        # await extract_evidence_example(client)
        # await get_full_review_example(client)
        
        logger.info("Examples completed successfully.")
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
    finally:
        # Always close the client to clean up resources
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())