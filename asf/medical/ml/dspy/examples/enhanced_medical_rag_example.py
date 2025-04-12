Enhanced Medical RAG Example

This script demonstrates how to use the enhanced DSPy integration for medical RAG.

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
    EnhancedDSPySettings,
    get_enhanced_settings,
    MedicalRAG,
    EnhancedMedicalRAG,
    AuditLogger,
    configure_audit_logger
)


class SimpleRetriever:
    Simple retriever that returns predefined passages.
        Initialize the simple retriever.
        
        Args:
            passages: List of passages to return
        """
        self.passages = passages
    
    def __call__(self, question: str, k: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Retrieve passages for a question.
        
        Args:
            question: Question to retrieve passages for
            k: Number of passages to retrieve
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: Retrieved passages
        """
        # Return a subset of passages
        return {"passages": self.passages[:k]}


async def main():
    """Run the enhanced medical RAG example."""
    # Configure audit logging
    configure_audit_logger(
        log_dir="audit_logs",
        enable_console_logging=True,
        log_level=logging.INFO
    )
    
    # Get enhanced client
    client = await get_enhanced_client()
    
    # Sample medical passages
    passages = [
        "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce pain, fever, and inflammation. "
        "It works by inhibiting the production of prostaglandins, which are hormone-like substances that cause pain and inflammation.",
        
        "Common side effects of aspirin include stomach pain, heartburn, nausea, and easy bruising. "
        "More serious side effects include stomach bleeding, allergic reactions, and Reye's syndrome in children.",
        
        "Aspirin is contraindicated in patients with a history of gastrointestinal bleeding, bleeding disorders, "
        "aspirin allergy, and children under 12 years of age due to the risk of Reye's syndrome.",
        
        "Low-dose aspirin (75-100 mg daily) is often prescribed for primary prevention of cardiovascular events "
        "in high-risk patients and for secondary prevention in patients with a history of cardiovascular disease.",
        
        "Recent studies have shown that the risks of routine aspirin use for primary prevention of cardiovascular disease "
        "may outweigh the benefits in many patients, particularly those at low risk for cardiovascular events."
    ]
    
    # Create a simple retriever
    retriever = SimpleRetriever(passages)
    
    # Create a medical RAG module
    medical_rag = MedicalRAG(retriever=retriever)
    
    # Register the module with the client
    await client.register_module(
        name="medical_rag",
        module=medical_rag,
        description="Medical RAG module for answering medical questions with citations"
    )
    
    # Sample questions
    questions = [
        "What are the main side effects of aspirin?",
        "Is aspirin safe for children?",
        "How does aspirin work to reduce pain and inflammation?"
    ]
    
    # Answer each question
    for question in questions:
        logger.info(f"Question: {question}")
        
        try:
            # Call the module
            result = await client.call_module("medical_rag", question=question)
            
            # Print the result
            logger.info(f"Answer: {result['answer']}")
            logger.info(f"Confidence: {result['confidence']}")
            logger.info(f"Citations: {result['citations']}")
            logger.info(f"Medical context: {result['medical_context']}")
            logger.info(f"Limitations: {result['limitations']}")
            logger.info("-" * 80)
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
    
    # Demonstrate circuit breaker pattern
    logger.info("Demonstrating circuit breaker pattern...")
    
    # Create a failing retriever to trigger circuit breaker
    class FailingRetriever:
        """
        FailingRetriever class.
        
        This class provides functionality for...
        """
        def __call__(self, question: str, k: int = 3, **kwargs):
            raise ConnectionError("Simulated connection error")
    
    # Create a medical RAG module with failing retriever
    failing_rag = MedicalRAG(retriever=FailingRetriever())
    
    # Register the module with the client
    await client.register_module(
        name="failing_rag",
        module=failing_rag,
        description="Medical RAG module with failing retriever"
    )
    
    # Try to call the failing module multiple times to trigger circuit breaker
    for i in range(6):  # Default threshold is 5
        try:
            logger.info(f"Attempt {i+1} to call failing module...")
            await client.call_module("failing_rag", question="Will this fail?")
        except Exception as e:
            logger.error(f"Error: {type(e).__name__}: {str(e)}")
    
    # Clean shutdown
    await client.shutdown()


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
