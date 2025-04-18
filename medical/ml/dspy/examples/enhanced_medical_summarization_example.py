Enhanced Medical Summarization Example

This script demonstrates how to use the enhanced DSPy integration for medical text summarization.

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
    MedicalSummarizer,
    AuditLogger,
    configure_audit_logger
)


async def main():
    """Run the enhanced medical summarization example."""
    # Configure audit logging
    configure_audit_logger(
        log_dir="audit_logs",
        enable_console_logging=True,
        log_level=logging.INFO
    )
    
    # Get enhanced client
    client = await get_enhanced_client()
    
    # Create medical summarizer modules
    basic_summarizer = MedicalSummarizer(structured=False)
    structured_summarizer = MedicalSummarizer(structured=True)
    
    # Register the modules with the client
    await client.register_module(
        name="basic_summarizer",
        module=basic_summarizer,
        description="Basic medical summarizer"
    )
    
    await client.register_module(
        name="structured_summarizer",
        module=structured_summarizer,
        description="Structured medical summarizer"
    )
    
    # Sample medical text
    medical_text = """
    BACKGROUND: Hypertension is a major risk factor for cardiovascular disease and affects approximately 1 billion people worldwide. Despite advances in treatment, many patients have uncontrolled hypertension.
    
    METHODS: We conducted a randomized, double-blind, placebo-controlled trial involving 10,705 patients with hypertension who were at high risk for cardiovascular events. Patients were randomly assigned to receive either a combination pill containing low doses of three antihypertensive drugs (the "polypill") or placebo. The primary outcome was a composite of cardiovascular death, myocardial infarction, stroke, or hospitalization for heart failure.
    
    RESULTS: After a median follow-up of 4.6 years, the primary outcome occurred in 4.7% of patients in the polypill group and 5.9% in the placebo group (hazard ratio, 0.79; 95% confidence interval [CI], 0.68 to 0.92; P=0.003). The incidence of the individual components of the primary outcome was also lower in the polypill group. The mean reduction in systolic blood pressure was 7.5 mm Hg greater in the polypill group than in the placebo group. Adverse events were similar in the two groups, although dizziness and hypotension were more common in the polypill group.
    
    CONCLUSIONS: In patients with hypertension who were at high risk for cardiovascular events, a polypill containing low doses of three antihypertensive drugs led to a significantly lower incidence of cardiovascular events than placebo. (Funded by the National Heart, Lung, and Blood Institute; ClinicalTrials.gov number, NCT01646437.)
    """
    
    # Summarize for different audiences
    audiences = ["clinician", "researcher", "patient"]
    
    # Basic summarization
    logger.info("Testing basic medical summarization:")
    for audience in audiences:
        logger.info(f"Audience: {audience}")
        
        try:
            # Call the module
            result = await client.call_module(
                "basic_summarizer",
                text=medical_text,
                audience=audience
            )
            
            # Print the result
            logger.info(f"Summary: {result['summary']}")
            logger.info(f"Key findings: {result['key_findings']}")
            logger.info(f"Implications: {result['implications']}")
            logger.info(f"Limitations: {result['limitations']}")
            logger.info("-" * 80)
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
    
    # Structured summarization
    logger.info("Testing structured medical summarization:")
    content_types = ["research_paper", "clinical_note", "guideline"]
    
    for content_type in content_types:
        logger.info(f"Content type: {content_type}")
        
        try:
            # Call the module
            result = await client.call_module(
                "structured_summarizer",
                text=medical_text,
                audience="clinician",
                content_type=content_type
            )
            
            # Print the result
            logger.info(f"Background: {result['background']}")
            logger.info(f"Methods: {result['methods']}")
            logger.info(f"Results: {result['results']}")
            logger.info(f"Conclusions: {result['conclusions']}")
            logger.info(f"Limitations: {result['limitations']}")
            logger.info("-" * 80)
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
    
    # Clean shutdown
    await client.shutdown()


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
