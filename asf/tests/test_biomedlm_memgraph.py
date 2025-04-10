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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_biomedlm_memgraph')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from asf.medical.models.biomedlm_wrapper import BioMedLMScorer
from asf.layer1_knowledge_substrate.memgraph_manager import MemgraphManager, MemgraphConfig
from asf.layer1_knowledge_substrate.chronograph_gnosis_layer import GnosisConfig, ChronoGnosisLayer

async def test_biomedlm():
    """Test BioMedLM contradiction scoring."""
    logger.info("Testing BioMedLM contradiction scoring...")
    
    try:
        # Initialize BioMedLM scorer
        scorer = BioMedLMScorer()
        
        # Test contradiction scoring
        claim1 = "Aspirin is effective for treating headaches."
        claim2 = "Aspirin has no effect on headache symptoms."
        
        # Get contradiction score
        score = scorer.get_score(claim1, claim2)
        logger.info(f"Contradiction score: {score}")
        
        # Get detailed scores
        detailed_scores = scorer.get_detailed_scores(claim1, claim2)
        logger.info(f"Detailed scores: {detailed_scores}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing BioMedLM: {e}")
        return False

async def test_memgraph():
    """Test Memgraph integration."""
    logger.info("Testing Memgraph integration...")
    
    try:
        # Initialize Memgraph config
        config = MemgraphConfig(
            host="localhost",
            port=7687,
            username="",
            password=""
        )
        
        # Initialize Memgraph manager
        manager = MemgraphManager(config)
        
        # Connect to Memgraph
        await manager.connect()
        logger.info("Connected to Memgraph")
        
        # Create test entity
        entity_id = await manager.create_entity({
            "name": "Test Entity",
            "type": "test",
            "created_at": "2023-01-01"
        })
        logger.info(f"Created entity with ID: {entity_id}")
        
        # Fetch entity
        entities = await manager.get_all_entity_ids()
        logger.info(f"Entities: {entities}")
        
        # Close connection
        await manager.close()
        logger.info("Closed Memgraph connection")
        
        return True
    except Exception as e:
        logger.error(f"Error testing Memgraph: {e}")
        return False

async def test_chronognosis_with_memgraph():
    """Test ChronoGnosisLayer with Memgraph."""
    logger.info("Testing ChronoGnosisLayer with Memgraph...")
    
    try:
        # Initialize GnosisConfig with Memgraph
        config = GnosisConfig(
            use_memgraph=True,
            memgraph=MemgraphConfig(
                host="localhost",
                port=7687,
                username="",
                password=""
            )
        )
        
        # Initialize ChronoGnosisLayer
        gnosis = ChronoGnosisLayer(config)
        
        # Start up
        await gnosis.startup()
        logger.info("Started ChronoGnosisLayer with Memgraph")
        
        # Generate embeddings for test entity
        entity_ids = ["test_entity_1", "test_entity_2"]
        embeddings = await gnosis.generate_embeddings(entity_ids)
        logger.info(f"Generated embeddings for {len(embeddings)} entities")
        
        # Shut down
        await gnosis.shutdown()
        logger.info("Shut down ChronoGnosisLayer")
        
        return True
    except Exception as e:
        logger.error(f"Error testing ChronoGnosisLayer with Memgraph: {e}")
        return False

async def main():
    """Run all tests."""
    logger.info("Starting tests...")
    
    # Test BioMedLM
    biomedlm_success = await test_biomedlm()
    logger.info(f"BioMedLM test {'succeeded' if biomedlm_success else 'failed'}")
    
    # Test Memgraph
    memgraph_success = await test_memgraph()
    logger.info(f"Memgraph test {'succeeded' if memgraph_success else 'failed'}")
    
    # Test ChronoGnosisLayer with Memgraph
    chronognosis_success = await test_chronognosis_with_memgraph()
    logger.info(f"ChronoGnosisLayer with Memgraph test {'succeeded' if chronognosis_success else 'failed'}")
    
    # Overall result
    overall_success = biomedlm_success and memgraph_success and chronognosis_success
    logger.info(f"Overall test {'succeeded' if overall_success else 'failed'}")
    
    return overall_success

if __name__ == "__main__":
    asyncio.run(main())
