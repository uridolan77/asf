"""Example usage of the ChronographMiddleware.

This module demonstrates how to use the ChronographMiddleware for various operations,
including data ingestion, embedding generation, trend analysis, and neighbor retrieval.
It serves as a practical guide for developers integrating with the ChronoGraph system.
"""

import asyncio
import logging
from datetime import datetime, timezone

from asf.medical.layer1_knowledge_substrate.chronograph import (
    ChronographMiddleware,
    Config,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("chronograph-example")


async def main():
    """Main function to demonstrate the ChronographMiddleware.

    This function shows a complete workflow of using the ChronographMiddleware:
    1. Initialize and start the middleware
    2. Ingest sample data
    3. Generate embeddings for the ingested entities
    4. Analyze temporal trends for the entities
    5. Retrieve neighbors for an entity
    6. Properly shut down the middleware

    Returns:
        None
    """

    try:
        # Initialize config
        config = Config()

        # Create middleware
        middleware = ChronographMiddleware(config)

        # Start middleware
        await middleware.startup()

        # Example 1: Ingest some data
        # This demonstrates how to ingest different types of medical data into the system
        data_points = [
            {
                "type": "clinical_trial",
                "title": "Example Trial 1",
                "status": "active",
                "phase": "Phase 3",
                "condition": "Type 2 Diabetes",
                "intervention": "Drug XYZ",
                "start_date": "2023-01-15"
            },
            {
                "type": "publication",
                "title": "Example Publication on Diabetes Treatment",
                "journal": "Nature Medicine",
                "authors": ["Smith, J.", "Johnson, M."],
                "publication_date": "2023-03-20",
                "doi": "10.1038/example12345"
            }
        ]
        entity_ids = await middleware.ingest_data(data_points)
        logger.info(f"Ingested entities: {entity_ids}")

        # Example 2: Generate embeddings
        # This demonstrates how to generate vector embeddings for entities
        # These embeddings can be used for similarity search, clustering, etc.
        embeddings = await middleware.generate_embeddings(entity_ids)
        logger.info(f"Generated embeddings for {len(embeddings)} entities")

        # You can also provide metadata to influence the embedding generation
        metadata = {
            entity_ids[0]: {
                "impact_factor": 4.5,
                "citation_count": 120,
                "design_score": 0.8
            },
            entity_ids[1]: {
                "impact_factor": 29.8,
                "citation_count": 45,
                "design_score": 0.9
            }
        }
        enhanced_embeddings = await middleware.generate_embeddings(entity_ids, metadata)
        logger.info(f"Generated enhanced embeddings with metadata")

        # Example 3: Analyze temporal trends
        # This demonstrates how to analyze temporal trends for entities
        # The analysis identifies emerging topics, changes in importance, etc.
        trends = await middleware.analyze_temporal_trends(entity_ids)
        logger.info(f"Analyzed trends with default time window (30 days)")

        # You can specify a custom time window for trend analysis
        long_term_trends = await middleware.analyze_temporal_trends(entity_ids, time_window=90)
        logger.info(f"Analyzed long-term trends (90 days)")

        # Example 4: Get neighbors
        # This demonstrates how to retrieve neighboring entities in the knowledge graph
        # Neighbors are entities connected by relationships in the graph
        neighbors = await middleware.get_neighbors(entity_ids[0])
        logger.info(f"Found {len(neighbors)} direct neighbors for entity {entity_ids[0]}")

        # You can specify a maximum distance to retrieve more distant neighbors
        extended_neighbors = await middleware.get_neighbors(entity_ids[0], max_distance=2)
        logger.info(f"Found {len(extended_neighbors)} neighbors within 2 steps for entity {entity_ids[0]}")

        # Shutdown middleware
        await middleware.shutdown()

    except Exception as e:
        logger.exception(f"Error in main: {e}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
