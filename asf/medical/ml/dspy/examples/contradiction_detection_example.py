Contradiction Detection Example

This example demonstrates how to use the Contradiction Detection modules to identify
contradictions between medical statements.

import asyncio
import logging
from typing import List, Dict, Any, Tuple

import dspy

from asf.medical.ml.dspy.dspy_client import get_dspy_client
from asf.medical.ml.dspy.modules.contradiction_detection import (
    ContradictionDetectionModule,
    TemporalContradictionModule,
    BioMedLMContradictionModule
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_contradiction_detection_example():
    """Run the Contradiction Detection example."""
    # Initialize DSPy client
    logger.info("Initializing DSPy client...")
    dspy_client = await get_dspy_client()
    
    # Create a Contradiction Detection module
    logger.info("Creating Contradiction Detection module...")
    contradiction_module = ContradictionDetectionModule()
    
    # Register the module
    logger.info("Registering module...")
    await dspy_client.register_module(
        name="contradiction_detection",
        module=contradiction_module,
        description="Module for detecting contradictions between medical statements"
    )
    
    # Sample statement pairs (statement1, statement2, expected_contradiction)
    statement_pairs = [
        (
            "Aspirin is recommended for primary prevention of cardiovascular disease in adults aged 50-59 with high cardiovascular risk.",
            "Aspirin should not be used for primary prevention of cardiovascular disease due to bleeding risks.",
            True
        ),
        (
            "Regular exercise can help control blood glucose levels in patients with type 2 diabetes.",
            "Physical activity is beneficial for glycemic control in type 2 diabetes patients.",
            False
        ),
        (
            "Statins are the first-line treatment for hypercholesterolemia.",
            "PCSK9 inhibitors should be used as initial therapy for patients with high cholesterol.",
            True
        )
    ]
    
    # Check each statement pair
    for i, (statement1, statement2, expected) in enumerate(statement_pairs):
        logger.info(f"Statement pair {i+1}:")
        logger.info(f"Statement 1: {statement1}")
        logger.info(f"Statement 2: {statement2}")
        logger.info(f"Expected contradiction: {expected}")
        
        # Call the module directly
        logger.info("Calling module directly...")
        direct_result = contradiction_module(
            statement1=statement1,
            statement2=statement2
        )
        logger.info(f"Detected contradiction: {direct_result['contradiction']}")
        logger.info(f"Explanation: {direct_result['explanation']}")
        logger.info(f"Confidence: {direct_result['confidence']}")
        
        # Call the module through the client
        logger.info("Calling module through client...")
        client_result = await dspy_client.call_module(
            module_name="contradiction_detection",
            statement1=statement1,
            statement2=statement2
        )
        logger.info(f"Detected contradiction: {client_result['contradiction']}")
        
        logger.info("-" * 50)
    
    # Create a Temporal Contradiction module
    logger.info("Creating Temporal Contradiction module...")
    temporal_module = TemporalContradictionModule()
    
    # Register the temporal module
    logger.info("Registering temporal module...")
    await dspy_client.register_module(
        name="temporal_contradiction",
        module=temporal_module,
        description="Module for detecting temporal contradictions between medical statements"
    )
    
    # Sample temporal statement pairs (statement1, timestamp1, statement2, timestamp2)
    temporal_pairs = [
        (
            "Hydroxychloroquine is a promising treatment for COVID-19 based on preliminary studies.",
            "March 2020",
            "Hydroxychloroquine is not recommended for COVID-19 treatment based on randomized controlled trials.",
            "June 2020"
        ),
        (
            "The patient's blood pressure is 140/90 mmHg.",
            "January 15, 2023",
            "The patient's blood pressure is 120/80 mmHg after starting antihypertensive medication.",
            "April 10, 2023"
        )
    ]
    
    # Check each temporal statement pair
    for i, (statement1, timestamp1, statement2, timestamp2) in enumerate(temporal_pairs):
        logger.info(f"Temporal statement pair {i+1}:")
        logger.info(f"Statement 1 ({timestamp1}): {statement1}")
        logger.info(f"Statement 2 ({timestamp2}): {statement2}")
        
        # Call the temporal module
        logger.info("Calling temporal module...")
        temporal_result = temporal_module(
            statement1=statement1,
            timestamp1=timestamp1,
            statement2=statement2,
            timestamp2=timestamp2
        )
        logger.info(f"Detected contradiction: {temporal_result['contradiction']}")
        logger.info(f"Temporal relation: {temporal_result['temporal_relation']}")
        logger.info(f"Explanation: {temporal_result['explanation']}")
        
        logger.info("-" * 50)
    
    # Create a BioMedLM Contradiction module
    logger.info("Creating BioMedLM Contradiction module...")
    biomedlm_module = BioMedLMContradictionModule()
    
    # Register the BioMedLM module
    logger.info("Registering BioMedLM module...")
    await dspy_client.register_module(
        name="biomedlm_contradiction",
        module=biomedlm_module,
        description="Module for detecting contradictions using BioMedLM"
    )
    
    # Check a complex medical contradiction with BioMedLM
    complex_statement1 = "Beta-blockers are contraindicated in patients with asthma due to the risk of bronchospasm."
    complex_statement2 = "Cardioselective beta-blockers can be used cautiously in patients with mild to moderate asthma when necessary for cardiovascular indications."
    
    logger.info("Complex medical contradiction:")
    logger.info(f"Statement 1: {complex_statement1}")
    logger.info(f"Statement 2: {complex_statement2}")
    
    # Call the BioMedLM module
    logger.info("Calling BioMedLM module...")
    biomedlm_result = biomedlm_module(
        statement1=complex_statement1,
        statement2=complex_statement2
    )
    logger.info(f"Detected contradiction: {biomedlm_result['contradiction']}")
    logger.info(f"Explanation: {biomedlm_result['explanation']}")
    logger.info(f"Confidence: {biomedlm_result['confidence']}")
    logger.info(f"Medical context: {biomedlm_result['medical_context']}")
    
    # Shut down the client
    logger.info("Shutting down DSPy client...")
    await dspy_client.shutdown()


if __name__ == "__main__":
    # Run the example
    asyncio.run(run_contradiction_detection_example())
