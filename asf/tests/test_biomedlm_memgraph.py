"""
Test script for BioMedLM and Memgraph integration.

This script tests the integration of BioMedLM for contradiction scoring
and Memgraph as an alternative to Neo4j in the ChronoGnosisLayer.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_biomedlm_memgraph')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asf.medical.models.biomedlm_wrapper import BioMedLMScorer
from asf.medical.layer1_knowledge_substrate.memgraph_manager import MemgraphManager, MemgraphConfig
from asf.medical.layer1_knowledge_substrate.chronograph_gnosis_layer import GnosisConfig, ChronoGnosisLayer

async def test_biomedlm():
    logger.info("Testing Memgraph integration...")

    try:
        config = MemgraphConfig(
            host="localhost",
            port=7687,
            username="",
            password=""
        )

        manager = MemgraphManager(config)

        await manager.connect()
        logger.info("Connected to Memgraph")

        entity_id = await manager.create_entity({
            "name": "Test Entity",
            "type": "test",
            "created_at": "2023-01-01"
        })
        logger.info(f"Created entity with ID: {entity_id}")

        entities = await manager.get_all_entity_ids()
        logger.info(f"Entities: {entities}")

        await manager.close()
        logger.info("Closed Memgraph connection")

        return True
    except Exception as e:
        logger.error(f"Error testing Memgraph: {e}")
        return False

async def test_chronognosis_with_memgraph():
    logger.info("Starting tests...")

    biomedlm_success = await test_biomedlm()
    logger.info(f"BioMedLM test {'succeeded' if biomedlm_success else 'failed'}")

    memgraph_success = await test_memgraph()
    logger.info(f"Memgraph test {'succeeded' if memgraph_success else 'failed'}")

    chronognosis_success = await test_chronognosis_with_memgraph()
    logger.info(f"ChronoGnosisLayer with Memgraph test {'succeeded' if chronognosis_success else 'failed'}")

    overall_success = biomedlm_success and memgraph_success and chronognosis_success
    logger.info(f"Overall test {'succeeded' if overall_success else 'failed'}")

    return overall_success

if __name__ == "__main__":
    asyncio.run(main())
