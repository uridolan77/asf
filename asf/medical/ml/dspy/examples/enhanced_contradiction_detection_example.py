Enhanced Contradiction Detection Example

This script demonstrates how to use the enhanced DSPy integration for medical contradiction detection.

import os
import asyncio
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import enhanced DSPy components
from asf.medical.ml.dspy import (
    EnhancedDSPyClient,
    get_enhanced_client,
    MedicalContradictionDetector,
    TemporalContradictionDetector,
    AuditLogger,
    configure_audit_logger
)


async def main():
    """Run the enhanced contradiction detection example."""
    # Configure audit logging
    configure_audit_logger(
        log_dir="audit_logs",
        enable_console_logging=True,
        log_level=logging.INFO
    )
    
    # Get enhanced client
    client = await get_enhanced_client()
    
    # Create contradiction detector modules
    contradiction_detector = MedicalContradictionDetector()
    temporal_contradiction_detector = TemporalContradictionDetector()
    
    # Register the modules with the client
    await client.register_module(
        name="contradiction_detector",
        module=contradiction_detector,
        description="Medical contradiction detector"
    )
    
    await client.register_module(
        name="temporal_contradiction_detector",
        module=temporal_contradiction_detector,
        description="Temporal medical contradiction detector"
    )
    
    # Sample statement pairs for contradiction detection
    statement_pairs = [
        (
            "Aspirin is safe for children with viral infections.",
            "Aspirin should not be given to children with viral infections due to the risk of Reye's syndrome."
        ),
        (
            "Statins are effective for lowering LDL cholesterol levels.",
            "Statins work by inhibiting HMG-CoA reductase, which reduces cholesterol production in the liver."
        ),
        (
            "Antibiotics are effective for treating viral infections.",
            "Antibiotics are only effective against bacterial infections, not viral infections."
        )
    ]
    
    # Detect contradictions
    logger.info("Testing basic contradiction detection:")
    for i, (statement1, statement2) in enumerate(statement_pairs):
        logger.info(f"Pair {i+1}:")
        logger.info(f"Statement 1: {statement1}")
        logger.info(f"Statement 2: {statement2}")
        
        try:
            # Call the module
            result = await client.call_module(
                "contradiction_detector",
                statement1=statement1,
                statement2=statement2
            )
            
            # Print the result
            logger.info(f"Contradiction: {result['contradiction']}")
            logger.info(f"Explanation: {result['explanation']}")
            logger.info(f"Confidence: {result['confidence']}")
            logger.info(f"Medical context: {result['medical_context']}")
            logger.info("-" * 80)
        except Exception as e:
            logger.error(f"Error detecting contradiction: {str(e)}")
    
    # Sample statement pairs for temporal contradiction detection
    temporal_statement_pairs = [
        (
            "The patient has no history of cardiovascular disease.",
            "2020-01-15",
            "The patient was diagnosed with atrial fibrillation.",
            "2022-03-10"
        ),
        (
            "The patient's HbA1c level is 7.2%, indicating good glycemic control.",
            "2021-06-20",
            "The patient's HbA1c level is 7.0%, showing improved glycemic control.",
            "2022-01-05"
        ),
        (
            "The patient is allergic to penicillin.",
            "2019-05-12",
            "The patient has no known drug allergies.",
            "2023-02-28"
        )
    ]
    
    # Detect temporal contradictions
    logger.info("Testing temporal contradiction detection:")
    for i, (statement1, timestamp1, statement2, timestamp2) in enumerate(temporal_statement_pairs):
        logger.info(f"Pair {i+1}:")
        logger.info(f"Statement 1 ({timestamp1}): {statement1}")
        logger.info(f"Statement 2 ({timestamp2}): {statement2}")
        
        try:
            # Call the module
            result = await client.call_module(
                "temporal_contradiction_detector",
                statement1=statement1,
                timestamp1=timestamp1,
                statement2=statement2,
                timestamp2=timestamp2
            )
            
            # Print the result
            logger.info(f"Contradiction: {result['contradiction']}")
            logger.info(f"Temporal relation: {result['temporal_relation']}")
            logger.info(f"Explanation: {result['explanation']}")
            logger.info(f"Confidence: {result['confidence']}")
            logger.info("-" * 80)
        except Exception as e:
            logger.error(f"Error detecting temporal contradiction: {str(e)}")
    
    # Clean shutdown
    await client.shutdown()


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
