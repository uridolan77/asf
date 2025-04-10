"""
Test script for PRISMA-guided screening and bias assessment.

This script tests the PRISMA-guided screening and bias assessment services.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, List, Any

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from asf.medical.ml.services.prisma_screening_service import (
    PRISMAScreeningService, ScreeningStage, ScreeningDecision
)
from asf.medical.ml.services.bias_assessment_service import (
    BiasAssessmentService, BiasRisk, BiasDomain
)
from asf.medical.ml.models.biomedlm import BioMedLMService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Test data
TEST_ARTICLES = [
    {
        "pmid": "12345",
        "title": "Randomized controlled trial of drug X for condition Y",
        "abstract": "Background: Condition Y affects many patients. Methods: We conducted a double-blind, randomized controlled trial with 500 patients. Results: Drug X showed significant improvement compared to placebo (p<0.001). Conclusion: Drug X is effective for condition Y."
    },
    {
        "pmid": "67890",
        "title": "Observational study of drug X for condition Y",
        "abstract": "Background: Drug X is used for condition Y, but evidence is limited. Methods: We conducted an observational study with 200 patients. Results: Drug X showed some improvement, but results were not statistically significant (p=0.08). Conclusion: More research is needed to establish the efficacy of drug X for condition Y."
    },
    {
        "pmid": "11111",
        "title": "Meta-analysis of interventions for condition Y",
        "abstract": "Background: Multiple interventions exist for condition Y. Methods: We conducted a meta-analysis of 15 studies. Results: Drug Z showed the most consistent benefits. Drug X showed mixed results. Conclusion: Drug Z should be considered first-line therapy for condition Y."
    }
]

# Test PRISMA screening criteria
TEST_SCREENING_CRITERIA = {
    ScreeningStage.IDENTIFICATION: {
        "include": [
            "randomized",
            "controlled trial",
            "meta-analysis"
        ],
        "exclude": [
            "animal study",
            "in vitro"
        ]
    },
    ScreeningStage.SCREENING: {
        "include": [
            "condition Y",
            "drug X"
        ],
        "exclude": [
            "pediatric",
            "pregnant women"
        ]
    },
    ScreeningStage.ELIGIBILITY: {
        "include": [
            "efficacy",
            "safety"
        ],
        "exclude": [
            "case report",
            "sample size < 100"
        ]
    }
}

async def test_prisma_screening():
    """Test PRISMA-guided screening."""
    logger.info("Testing PRISMA-guided screening...")
    
    # Initialize services
    try:
        biomedlm_service = BioMedLMService()
        logger.info("Initialized BioMedLM service")
    except Exception as e:
        logger.warning(f"Could not initialize BioMedLM service: {str(e)}. Using basic screening.")
        biomedlm_service = None
    
    screening_service = PRISMAScreeningService(biomedlm_service=biomedlm_service)
    
    # Set screening criteria
    for stage, criteria in TEST_SCREENING_CRITERIA.items():
        screening_service.set_criteria(
            stage=stage,
            include_criteria=criteria["include"],
            exclude_criteria=criteria["exclude"]
        )
    
    # Test screening at different stages
    for stage in [ScreeningStage.IDENTIFICATION, ScreeningStage.SCREENING, ScreeningStage.ELIGIBILITY]:
        logger.info(f"Testing screening at {stage} stage...")
        
        # Screen articles
        results = await screening_service.screen_articles(
            articles=TEST_ARTICLES,
            stage=stage
        )
        
        # Print results
        included = sum(1 for r in results if r["decision"] == ScreeningDecision.INCLUDE)
        excluded = sum(1 for r in results if r["decision"] == ScreeningDecision.EXCLUDE)
        uncertain = sum(1 for r in results if r["decision"] == ScreeningDecision.UNCERTAIN)
        
        logger.info(f"Screening results for {stage} stage:")
        logger.info(f"  Included: {included}")
        logger.info(f"  Excluded: {excluded}")
        logger.info(f"  Uncertain: {uncertain}")
        
        # Print details for each article
        for result in results:
            logger.info(f"  Article {result['article_id']}: {result['decision']} (confidence: {result['confidence']})")
            if result["matched_include_criteria"]:
                logger.info(f"    Matched include criteria: {result['matched_include_criteria']}")
            if result["matched_exclude_criteria"]:
                logger.info(f"    Matched exclude criteria: {result['matched_exclude_criteria']}")
    
    # Get PRISMA flow data
    flow_data = screening_service.get_flow_data()
    logger.info(f"PRISMA flow data: {flow_data}")
    
    # Generate flow diagram
    diagram_data = screening_service.generate_flow_diagram()
    logger.info(f"PRISMA flow diagram data: {diagram_data}")

async def test_bias_assessment():
    """Test bias assessment."""
    logger.info("Testing bias assessment...")
    
    # Initialize service
    try:
        import spacy
        nlp = spacy.load("en_core_sci_md")
        logger.info("Loaded spaCy model for bias assessment")
    except Exception as e:
        logger.warning(f"Could not load spaCy model: {str(e)}. Falling back to basic pattern matching.")
        nlp = None
    
    bias_service = BiasAssessmentService(nlp_model=nlp)
    
    # Assess bias in articles
    results = await bias_service.assess_studies(TEST_ARTICLES)
    
    # Print results
    low_risk = sum(1 for r in results 
                  if r["assessment"][BiasDomain.OVERALL]["risk"] == BiasRisk.LOW)
    moderate_risk = sum(1 for r in results 
                       if r["assessment"][BiasDomain.OVERALL]["risk"] == BiasRisk.MODERATE)
    high_risk = sum(1 for r in results 
                   if r["assessment"][BiasDomain.OVERALL]["risk"] == BiasRisk.HIGH)
    unclear_risk = sum(1 for r in results 
                      if r["assessment"][BiasDomain.OVERALL]["risk"] == BiasRisk.UNCLEAR)
    
    logger.info("Bias assessment results:")
    logger.info(f"  Low risk: {low_risk}")
    logger.info(f"  Moderate risk: {moderate_risk}")
    logger.info(f"  High risk: {high_risk}")
    logger.info(f"  Unclear risk: {unclear_risk}")
    
    # Print details for each article
    for result in results:
        logger.info(f"  Article {result['study_id']}: Overall risk - {result['assessment'][BiasDomain.OVERALL]['risk']}")
        logger.info(f"    Summary: {result['assessment'][BiasDomain.OVERALL]['summary']}")
        
        # Print domain-specific results
        for domain in [d for d in BiasDomain if d != BiasDomain.OVERALL]:
            domain_result = result["assessment"][domain]
            logger.info(f"    {domain.value.replace('_', ' ').title()}: {domain_result['risk']}")
            
            if domain_result["evidence"]:
                for evidence in domain_result["evidence"][:2]:  # Show only first 2 pieces of evidence
                    logger.info(f"      - {evidence['text']} ({evidence['type']})")
    
    # Generate summary for first article
    if results:
        summary = bias_service.generate_summary(results[0]["assessment"])
        logger.info(f"Bias assessment summary for article {results[0]['study_id']}:")
        logger.info(summary)

async def main():
    """Main function."""
    logger.info("Starting screening and bias assessment test...")
    
    # Test PRISMA screening
    await test_prisma_screening()
    
    # Test bias assessment
    await test_bias_assessment()
    
    logger.info("Screening and bias assessment test completed")

if __name__ == "__main__":
    asyncio.run(main())
