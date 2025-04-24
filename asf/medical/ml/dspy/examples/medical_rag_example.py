Medical RAG Example

This example demonstrates how to use the Medical RAG module to answer medical questions.

import asyncio
import logging
from typing import List, Dict, Any

import dspy

from asf.medical.ml.dspy.dspy_client import get_dspy_client
from asf.medical.ml.dspy.modules.medical_rag import MedicalRAGModule, EnhancedMedicalRAGModule

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_medical_rag_example():
    """Run the Medical RAG example."""
    # Initialize DSPy client
    logger.info("Initializing DSPy client...")
    dspy_client = await get_dspy_client()
    
    # Create a simple Medical RAG module
    logger.info("Creating Medical RAG module...")
    medical_rag = MedicalRAGModule(k=3)
    
    # Register the module
    logger.info("Registering module...")
    await dspy_client.register_module(
        name="medical_rag",
        module=medical_rag,
        description="Medical RAG module for answering medical questions"
    )
    
    # Sample medical questions
    questions = [
        "What are the common side effects of statins?",
        "How does COVID-19 affect the cardiovascular system?",
        "What is the recommended treatment for type 2 diabetes?"
    ]
    
    # Answer each question
    for i, question in enumerate(questions):
        logger.info(f"Question {i+1}: {question}")
        
        # Call the module directly
        logger.info("Calling module directly...")
        direct_result = medical_rag(question=question)
        logger.info(f"Answer: {direct_result['answer']}")
        
        # Call the module through the client
        logger.info("Calling module through client...")
        client_result = await dspy_client.call_module(
            module_name="medical_rag",
            question=question
        )
        logger.info(f"Answer: {client_result['answer']}")
        
        logger.info("-" * 50)
    
    # Create an Enhanced Medical RAG module
    logger.info("Creating Enhanced Medical RAG module...")
    enhanced_rag = EnhancedMedicalRAGModule(
        initial_k=2,
        secondary_k=3
    )
    
    # Register the enhanced module
    logger.info("Registering enhanced module...")
    await dspy_client.register_module(
        name="enhanced_medical_rag",
        module=enhanced_rag,
        description="Enhanced Medical RAG module with query expansion and fact checking"
    )
    
    # Answer a complex medical question with the enhanced module
    complex_question = "What is the current consensus on using aspirin for primary prevention of cardiovascular disease?"
    
    logger.info(f"Complex question: {complex_question}")
    
    # Call the enhanced module
    logger.info("Calling enhanced module...")
    enhanced_result = enhanced_rag(question=complex_question)
    
    logger.info(f"Expanded query: {enhanced_result['expanded_query']}")
    logger.info(f"Answer: {enhanced_result['answer']}")
    logger.info(f"Confidence: {enhanced_result['confidence']}")
    logger.info(f"Verification notes: {enhanced_result['verification_notes']}")
    
    # Shut down the client
    logger.info("Shutting down DSPy client...")
    await dspy_client.shutdown()


if __name__ == "__main__":
    # Run the example
    asyncio.run(run_medical_rag_example())
