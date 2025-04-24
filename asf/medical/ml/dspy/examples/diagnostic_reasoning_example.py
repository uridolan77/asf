"""Diagnostic Reasoning Example

This example demonstrates how to use the DiagnosticReasoningModule for medical case analysis.
"""

import asyncio
import logging
from typing import Dict, Any

from asf.medical.ml.dspy.client import get_enhanced_client
from asf.medical.ml.dspy.modules.diagnostic_reasoning import DiagnosticReasoningModule, SpecialistConsultModule

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_diagnostic_reasoning():
    """Run a diagnostic reasoning example."""
    # Initialize DSPy client
    client = await get_enhanced_client()
    
    # Create a diagnostic reasoning module
    diagnostic_module = DiagnosticReasoningModule(
        max_diagnoses=5,
        include_rare_conditions=True
    )
    
    # Register the module
    await client.register_module(
        name="diagnostic_reasoning",
        module=diagnostic_module,
        description="Medical diagnostic reasoning module"
    )
    
    # Example case description
    case_description = """
    A 45-year-old male presents with sudden onset of severe chest pain radiating to the left arm and jaw. 
    The pain started about 2 hours ago while he was resting. He describes it as a heavy pressure sensation. 
    He has a history of hypertension and hyperlipidemia, and his father had a myocardial infarction at age 50. 
    He is a current smoker with a 20 pack-year history. On examination, he appears diaphoretic and anxious. 
    Vital signs show BP 160/95, HR 110, RR 22, and oxygen saturation 96% on room air. 
    His ECG shows ST-segment elevation in leads II, III, and aVF.
    """
    
    # Call the module
    logger.info("Running diagnostic reasoning...")
    result = await client.call_module(
        module_name="diagnostic_reasoning",
        case_description=case_description
    )
    
    # Print the results
    logger.info("Diagnostic Reasoning Results:")
    logger.info(f"Case Summary: {result['case_summary']}")
    logger.info(f"Differential Diagnosis: {result['differential_diagnosis']}")
    logger.info(f"Recommended Tests: {result['recommended_tests']}")
    logger.info(f"Reasoning: {result['reasoning'][:200]}...")
    logger.info(f"Confidence: {result['confidence']}")
    
    # Create a specialist consultation module (cardiology)
    cardiology_module = SpecialistConsultModule(
        specialty="cardiology",
        base_reasoning_module=diagnostic_module
    )
    
    # Register the specialist module
    await client.register_module(
        name="cardiology_consult",
        module=cardiology_module,
        description="Cardiology specialist consultation module"
    )
    
    # Call the specialist module
    logger.info("\nRunning cardiology consultation...")
    specialist_result = await client.call_module(
        module_name="cardiology_consult",
        case_description=case_description
    )
    
    # Print the specialist results
    logger.info("Cardiology Consultation Results:")
    logger.info(f"Specialist Diagnosis: {specialist_result['specialist_diagnosis']}")
    logger.info(f"Specialist Recommendations: {specialist_result['specialist_recommendations']}")
    logger.info(f"Specialist Assessment: {specialist_result['specialist_assessment'][:200]}...")
    logger.info(f"Confidence: {specialist_result['confidence']}")
    
    # Shut down the client
    await client.shutdown()


if __name__ == "__main__":
    # Run the example
    asyncio.run(run_diagnostic_reasoning())
