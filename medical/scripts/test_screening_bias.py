"""
Test script for PRISMA-guided screening and bias assessment.
This script tests the PRISMA-guided screening and bias assessment services.
"""
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from asf.medical.ml.services.prisma_screening_service import (
    PRISMAScreeningService, ScreeningStage, ScreeningDecision
)
from asf.medical.ml.services.bias_assessment_service import (
    BiasAssessmentService, BiasRisk, BiasDomain
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
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
    logger.info("Testing bias assessment...")
    try:
        import spacy
        nlp = spacy.load("en_core_sci_md")
        logger.info("Loaded spaCy model for bias assessment")
    except Exception as e:
        logger.warning(f"Could not load spaCy model: {str(e)}. Falling back to basic pattern matching.")
        nlp = None
    bias_service = BiasAssessmentService(nlp_model=nlp)
    results = await bias_service.assess_studies(TEST_ARTICLES)
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
    for result in results:
        logger.info(f"  Article {result['study_id']}: Overall risk - {result['assessment'][BiasDomain.OVERALL]['risk']}")
        logger.info(f"    Summary: {result['assessment'][BiasDomain.OVERALL]['summary']}")
        for domain in [d for d in BiasDomain if d != BiasDomain.OVERALL]:
            domain_result = result["assessment"][domain]
            logger.info(f"    {domain.value.replace('_', ' ').title()}: {domain_result['risk']}")
            if domain_result["evidence"]:
                for evidence in domain_result["evidence"][:2]:  # Show only first 2 pieces of evidence
                    logger.info(f"      - {evidence['text']} ({evidence['type']})")
    if results:
        summary = bias_service.generate_summary(results[0]["assessment"])
        logger.info(f"Bias assessment summary for article {results[0]['study_id']}:")
        logger.info(summary)
async def main():