#!/usr/bin/env python
"""
Test script for domain-specific temporal contradiction detection.

This script tests the domain-specific temporal contradiction detection and explanation generation.
"""

import sys
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from asf.medical.ml.services.contradiction_service import ContradictionService
    from asf.medical.ml.services.temporal_service import TemporalService
    from asf.medical.ml.models.biomedlm import BioMedLMService
    USING_REAL_SERVICES = True
except ImportError:
    logger.warning("Failed to import required modules. Using mock implementations.")
    USING_REAL_SERVICES = False
    
    # Define enums for testing
    class ContradictionType:
        NONE = "none"
        DIRECT = "direct"
        NEGATION = "negation"
        STATISTICAL = "statistical"
        METHODOLOGICAL = "methodological"
        TEMPORAL = "temporal"
        UNKNOWN = "unknown"

    class ContradictionConfidence:
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        UNKNOWN = "unknown"
    
    class MockBioMedLMService:
        """Mock BioMedLM service for testing."""
        
        def __init__(self):
            """Initialize the mock BioMedLM service."""
            logger.info("Mock BioMedLM service initialized")
        
        def calculate_similarity(self, text1, text2):
            """Calculate similarity between two texts."""
            # Simple similarity calculation
            if text1 == text2:
                return 1.0
            
            # Calculate Jaccard similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
        
        def encode(self, text):
            """Encode text into a vector."""
            # Return a dummy vector
            import numpy as np
            np.random.seed(hash(text) % 2**32)
            return np.random.rand(768)
    
    class MockTemporalService:
        """Mock temporal service for testing."""
        
        def __init__(self):
            """Initialize the mock temporal service."""
            logger.info("Mock temporal service initialized")
            
            # Domain-specific decay rates (half-life in days) and characteristics
            self.domain_characteristics = {
                # Rapidly evolving fields with frequent new treatments and research
                "oncology": {
                    "half_life": 365 * 2,  # 2 years
                    "evolution_rate": "rapid",
                    "evidence_stability": "moderate",
                    "technology_dependence": "high",
                    "description": "Cancer research evolves rapidly with new treatments and targeted therapies"
                },
                "infectious_disease": {
                    "half_life": 365 * 1,  # 1 year
                    "evolution_rate": "very_rapid",
                    "evidence_stability": "low",
                    "technology_dependence": "high",
                    "description": "Infectious disease knowledge changes quickly with emerging pathogens and resistance patterns"
                },
                # Slowly evolving fields with established principles
                "neurology": {
                    "half_life": 365 * 4,  # 4 years
                    "evolution_rate": "slow",
                    "evidence_stability": "high",
                    "technology_dependence": "moderate",
                    "description": "Neurological principles change slowly despite technological advances in imaging"
                },
                "psychiatry": {
                    "half_life": 365 * 5,  # 5 years
                    "evolution_rate": "slow",
                    "evidence_stability": "moderate",
                    "technology_dependence": "low",
                    "description": "Psychiatric knowledge evolves gradually with long-term studies and observations"
                },
                # Default for unknown domains
                "default": {
                    "half_life": 365 * 2.5,  # 2.5 years
                    "evolution_rate": "moderate",
                    "evidence_stability": "moderate",
                    "technology_dependence": "moderate",
                    "description": "General medical knowledge with moderate evolution rate"
                }
            }
        
        def calculate_temporal_confidence(
            self,
            publication_date: str,
            domain: str = "default",
            reference_date: str = None,
            include_details: bool = False
        ):
            """Calculate temporal confidence for a publication."""
            # Parse dates
            try:
                pub_date = datetime.datetime.strptime(publication_date, "%Y-%m-%d")
            except ValueError:
                try:
                    # Try with just year
                    pub_date = datetime.datetime.strptime(publication_date, "%Y")
                except ValueError:
                    logger.warning(f"Invalid publication date: {publication_date}")
                    if include_details:
                        return {
                            "confidence": 0.5,
                            "reason": "Invalid publication date format",
                            "domain": domain,
                            "domain_characteristics": self.domain_characteristics.get(domain.lower(), self.domain_characteristics["default"])
                        }
                    return 0.5  # Default confidence
            
            if reference_date:
                try:
                    ref_date = datetime.datetime.strptime(reference_date, "%Y-%m-%d")
                except ValueError:
                    logger.warning(f"Invalid reference date: {reference_date}")
                    ref_date = datetime.datetime.now()
            else:
                ref_date = datetime.datetime.now()
            
            # Get domain characteristics
            domain_key = domain.lower()
            domain_info = self.domain_characteristics.get(domain_key, self.domain_characteristics["default"])
            half_life = domain_info["half_life"]
            
            # Calculate time difference in days
            time_diff = (ref_date - pub_date).days
            time_diff_years = time_diff / 365.0
            
            # Calculate confidence using exponential decay
            if time_diff < 0:
                # Future publication (shouldn't happen)
                if include_details:
                    return {
                        "confidence": 0.5,
                        "reason": "Publication date is in the future",
                        "domain": domain,
                        "domain_characteristics": domain_info
                    }
                return 0.5
            
            # Base confidence using exponential decay
            import math
            base_confidence = math.exp(-math.log(2) * time_diff / half_life)
            
            # Apply domain-specific adjustments
            adjusted_confidence = base_confidence
            
            # Adjust based on evidence stability
            evidence_stability = domain_info["evidence_stability"]
            if evidence_stability == "very_high":
                # Very stable evidence decays more slowly
                stability_factor = 1.2
            elif evidence_stability == "high":
                stability_factor = 1.1
            elif evidence_stability == "moderate":
                stability_factor = 1.0
            elif evidence_stability == "low":
                stability_factor = 0.9
            else:  # very_low
                stability_factor = 0.8
            
            adjusted_confidence *= stability_factor
            
            # Ensure confidence is between 0 and 1
            final_confidence = max(0.0, min(1.0, adjusted_confidence))
            
            if include_details:
                return {
                    "confidence": float(final_confidence),
                    "base_confidence": float(base_confidence),
                    "stability_factor": stability_factor,
                    "time_diff_days": time_diff,
                    "time_diff_years": time_diff_years,
                    "half_life_days": half_life,
                    "half_life_years": half_life / 365.0,
                    "domain": domain,
                    "domain_characteristics": domain_info
                }
            
            return float(final_confidence)
    
    class MockContradictionService:
        """Mock contradiction service for testing."""
        
        def __init__(self, biomedlm_service=None, temporal_service=None):
            """Initialize the mock contradiction service."""
            logger.info("Mock contradiction service initialized")
            self.biomedlm_service = biomedlm_service or MockBioMedLMService()
            self.temporal_service = temporal_service or MockTemporalService()
            
            # Thresholds for contradiction detection
            self.thresholds = {
                ContradictionType.DIRECT: 0.7,
                ContradictionType.NEGATION: 0.8,
                ContradictionType.STATISTICAL: 0.7,
                ContradictionType.METHODOLOGICAL: 0.7,
                ContradictionType.TEMPORAL: 0.7,
            }
        
        async def detect_contradiction(
            self,
            claim1: str,
            claim2: str,
            metadata1: Dict[str, Any] = None,
            metadata2: Dict[str, Any] = None,
            threshold: float = 0.7,
            use_biomedlm: bool = True,
            use_temporal: bool = True,
            use_tsmixer: bool = False
        ) -> Dict[str, Any]:
            """Detect contradiction between two claims."""
            # Initialize result
            result = {
                "is_contradiction": False,
                "contradiction_score": 0.0,
                "contradiction_type": ContradictionType.NONE,
                "confidence": ContradictionConfidence.UNKNOWN,
                "explanation": None,
                "methods_used": [],
                "details": {}
            }
            
            # Check for temporal contradiction
            if use_temporal and metadata1 and metadata2:
                # Extract publication dates
                pub_date1 = metadata1.get("publication_date", "")
                pub_date2 = metadata2.get("publication_date", "")
                
                if pub_date1 and pub_date2:
                    try:
                        # Parse dates
                        date1 = datetime.datetime.strptime(pub_date1, "%Y-%m-%d")
                        date2 = datetime.datetime.strptime(pub_date2, "%Y-%m-%d")
                        
                        # Calculate time difference in days
                        time_diff = abs((date2 - date1).days)
                        
                        # Extract domains
                        domain1 = metadata1.get("domain", "default")
                        domain2 = metadata2.get("domain", "default")
                        
                        # Calculate similarity
                        similarity = self.biomedlm_service.calculate_similarity(claim1, claim2)
                        
                        # Check for numerical differences in claims
                        import re
                        numbers1 = re.findall(r'\d+(?:\.\d+)?%?', claim1)
                        numbers2 = re.findall(r'\d+(?:\.\d+)?%?', claim2)
                        has_different_numbers = False
                        
                        if numbers1 and numbers2 and len(numbers1) == len(numbers2):
                            for n1, n2 in zip(numbers1, numbers2):
                                if n1 != n2:
                                    has_different_numbers = True
                                    break
                        
                        # Check if one claim is a subset of the other
                        is_subset = claim1 in claim2 or claim2 in claim1
                        
                        # If time difference is significant and claims are similar or have different numbers, it's a temporal contradiction
                        if time_diff > 365 * 5 and (similarity > 0.7 or has_different_numbers or is_subset):
                            # Calculate temporal confidence for each claim with detailed information
                            confidence_details1 = self.temporal_service.calculate_temporal_confidence(
                                pub_date1, domain1, include_details=True
                            )
                            confidence_details2 = self.temporal_service.calculate_temporal_confidence(
                                pub_date2, domain2, include_details=True
                            )
                            
                            # Extract confidence values
                            confidence1 = confidence_details1["confidence"] if isinstance(confidence_details1, dict) else confidence_details1
                            confidence2 = confidence_details2["confidence"] if isinstance(confidence_details2, dict) else confidence_details2
                            
                            # Calculate confidence difference
                            confidence_diff = abs(confidence1 - confidence2)
                            
                            # Store domain-specific information for explanation generation
                            domain_info = {
                                "claim1": {
                                    "domain": domain1,
                                    "confidence": confidence1,
                                    "publication_date": pub_date1,
                                    "details": confidence_details1 if isinstance(confidence_details1, dict) else None
                                },
                                "claim2": {
                                    "domain": domain2,
                                    "confidence": confidence2,
                                    "publication_date": pub_date2,
                                    "details": confidence_details2 if isinstance(confidence_details2, dict) else None
                                },
                                "time_diff_days": time_diff,
                                "time_diff_years": time_diff / 365.0
                            }
                            
                            # Set result
                            result["is_contradiction"] = True
                            result["contradiction_score"] = min(1.0, time_diff / (365 * 10))  # Cap at 10 years
                            result["contradiction_type"] = ContradictionType.TEMPORAL
                            
                            # Set confidence based on time difference
                            if time_diff > 365 * 10:  # More than 10 years
                                result["confidence"] = ContradictionConfidence.HIGH
                            elif time_diff > 365 * 5:  # More than 5 years
                                result["confidence"] = ContradictionConfidence.MEDIUM
                            else:
                                result["confidence"] = ContradictionConfidence.LOW
                            
                            # Generate domain-specific explanation
                            result["explanation"] = self._generate_temporal_contradiction_explanation(
                                claim1, claim2, domain_info, has_different_numbers, is_subset
                            )
                            
                            # Add methods used
                            result["methods_used"] = ["temporal"]
                            
                            # Add domain-specific details to the result
                            result["details"]["domain_info"] = domain_info
                    except Exception as e:
                        logger.error(f"Error detecting temporal contradiction: {str(e)}")
            
            return result
        
        def _generate_temporal_contradiction_explanation(
            self,
            claim1: str,
            claim2: str,
            domain_info: Dict[str, Any],
            has_different_numbers: bool = False,
            is_subset: bool = False
        ) -> str:
            """Generate a detailed explanation for temporal contradiction with domain-specific insights."""
            # Extract information
            domain1 = domain_info["claim1"]["domain"]
            domain2 = domain_info["claim2"]["domain"]
            pub_date1 = domain_info["claim1"]["publication_date"]
            pub_date2 = domain_info["claim2"]["publication_date"]
            time_diff_years = domain_info["time_diff_years"]
            
            # Get detailed explanations if available
            details1 = domain_info["claim1"].get("details", {})
            details2 = domain_info["claim2"].get("details", {})
            
            # Base explanation
            explanation = f"Temporal contradiction detected: Claims are similar but published {time_diff_years:.1f} years apart ({pub_date1} vs {pub_date2}). "
            
            # Add domain-specific insights if available
            if details1 and details2 and isinstance(details1, dict) and isinstance(details2, dict):
                # Get domain characteristics
                domain1_chars = details1.get("domain_characteristics", {})
                domain2_chars = details2.get("domain_characteristics", {})
                
                # Add domain-specific explanation
                if domain1 == domain2:
                    # Same domain
                    domain_desc = domain1_chars.get("description", "")
                    evolution_rate = domain1_chars.get("evolution_rate", "moderate")
                    half_life_years = domain1_chars.get("half_life", 365 * 2.5) / 365.0
                    
                    # Evolution rate description
                    if evolution_rate == "very_rapid":
                        evolution_desc = "very rapidly evolving"
                        rate_explanation = "knowledge can change significantly even within a few years"
                    elif evolution_rate == "rapid":
                        evolution_desc = "rapidly evolving"
                        rate_explanation = "significant advances occur frequently"
                    elif evolution_rate == "moderate":
                        evolution_desc = "moderately evolving"
                        rate_explanation = "knowledge evolves steadily over time"
                    elif evolution_rate == "slow":
                        evolution_desc = "slowly evolving"
                        rate_explanation = "fundamental principles remain stable for many years"
                    else:  # very_slow
                        evolution_desc = "very slowly evolving"
                        rate_explanation = "core knowledge remains stable for decades"
                    
                    # Format domain name for better readability
                    formatted_domain = domain1.replace("_", " ").title()
                    
                    explanation += f"In {evolution_desc} {formatted_domain} medicine, {rate_explanation}. "
                    explanation += f"Evidence typically has a half-life of {half_life_years:.1f} years. "
                    
                    if domain_desc:
                        explanation += f"{domain_desc}. "
                else:
                    # Different domains
                    # Format domain names for better readability
                    formatted_domain1 = domain1.replace("_", " ").title()
                    formatted_domain2 = domain2.replace("_", " ").title()
                    
                    explanation += f"The claims come from different medical domains ({formatted_domain1} vs {formatted_domain2}), "
                    explanation += "which may have different rates of knowledge evolution. "
            
            # Add explanation based on claim characteristics
            if has_different_numbers:
                explanation += "The claims contain different numerical values, which likely reflect updated statistics or findings. "
            elif is_subset:
                explanation += "One claim appears to be an extension of the other, suggesting additional information has been discovered over time. "
            elif claim1 == claim2:
                explanation += "Despite identical wording, the significant time difference suggests the validity of this claim has been reconfirmed over time. "
            
            # Add recommendation
            explanation += "The more recent publication likely reflects updated evidence or changing medical knowledge."
            
            return explanation

# Test data with domain-specific temporal contradictions
TEST_CASES = [
    {
        "claim1": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
        "claim2": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
        "metadata1": {
            "publication_date": "2010-01-01",
            "domain": "cardiology",
            "study_design": "randomized controlled trial",
            "sample_size": 1000
        },
        "metadata2": {
            "publication_date": "2020-06-15",
            "domain": "cardiology",
            "study_design": "randomized controlled trial",
            "sample_size": 2000
        },
        "description": "Same claims in cardiology with 10-year difference"
    },
    {
        "claim1": "Antibiotics are effective for treating bacterial pneumonia with a success rate of 95%.",
        "claim2": "Antibiotics are effective for treating bacterial pneumonia with a success rate of 85% due to increasing resistance.",
        "metadata1": {
            "publication_date": "2000-05-10",
            "domain": "infectious_disease",
            "study_design": "meta-analysis",
            "sample_size": 5000
        },
        "metadata2": {
            "publication_date": "2022-03-20",
            "domain": "infectious_disease",
            "study_design": "meta-analysis",
            "sample_size": 8000
        },
        "description": "Different numerical values in infectious disease with 22-year difference"
    },
    {
        "claim1": "Cognitive behavioral therapy is effective for treating depression.",
        "claim2": "Cognitive behavioral therapy is effective for treating depression.",
        "metadata1": {
            "publication_date": "2005-01-15",
            "domain": "psychiatry",
            "study_design": "randomized controlled trial",
            "sample_size": 300
        },
        "metadata2": {
            "publication_date": "2022-01-15",
            "domain": "psychiatry",
            "study_design": "randomized controlled trial",
            "sample_size": 500
        },
        "description": "Same claims in psychiatry with 17-year difference"
    },
    {
        "claim1": "MRI is the preferred imaging modality for diagnosing multiple sclerosis.",
        "claim2": "MRI is the preferred imaging modality for diagnosing multiple sclerosis.",
        "metadata1": {
            "publication_date": "2000-01-01",
            "domain": "neurology",
            "study_design": "clinical guideline",
            "sample_size": None
        },
        "metadata2": {
            "publication_date": "2020-01-01",
            "domain": "neurology",
            "study_design": "clinical guideline",
            "sample_size": None
        },
        "description": "Same claims in neurology with 20-year difference"
    },
    {
        "claim1": "Targeted therapy improves survival in HER2-positive breast cancer.",
        "claim2": "Targeted therapy with trastuzumab and pertuzumab improves survival in HER2-positive breast cancer by 56%.",
        "metadata1": {
            "publication_date": "2005-01-01",
            "domain": "oncology",
            "study_design": "randomized controlled trial",
            "sample_size": 500
        },
        "metadata2": {
            "publication_date": "2020-01-01",
            "domain": "oncology",
            "study_design": "randomized controlled trial",
            "sample_size": 1000
        },
        "description": "Subset claim in oncology with 15-year difference"
    },
    {
        "claim1": "Regular physical activity reduces the risk of cardiovascular disease.",
        "claim2": "Regular physical activity reduces the risk of cardiovascular disease.",
        "metadata1": {
            "publication_date": "2010-01-01",
            "domain": "cardiology",
            "study_design": "meta-analysis",
            "sample_size": 10000
        },
        "metadata2": {
            "publication_date": "2012-01-01",
            "domain": "cardiology",
            "study_design": "meta-analysis",
            "sample_size": 12000
        },
        "description": "Same claims in cardiology with only 2-year difference (should not be a temporal contradiction)"
    },
    {
        "claim1": "Aspirin reduces the risk of heart attack in patients with a history of cardiovascular disease.",
        "claim2": "Aspirin reduces the risk of heart attack in patients with a history of cardiovascular disease.",
        "metadata1": {
            "publication_date": "2000-01-01",
            "domain": "cardiology",
            "study_design": "randomized controlled trial",
            "sample_size": 5000
        },
        "metadata2": {
            "publication_date": "2020-01-01",
            "domain": "public_health",
            "study_design": "meta-analysis",
            "sample_size": 50000
        },
        "description": "Same claims in different domains (cardiology vs public health) with 20-year difference"
    }
]

async def test_domain_specific_temporal_contradiction():
    """Test domain-specific temporal contradiction detection."""
    logger.info("Testing domain-specific temporal contradiction detection...")
    
    # Initialize services
    if USING_REAL_SERVICES:
        biomedlm_service = BioMedLMService()
        temporal_service = TemporalService()
        contradiction_service = ContradictionService(
            biomedlm_service=biomedlm_service,
            temporal_service=temporal_service
        )
    else:
        biomedlm_service = MockBioMedLMService()
        temporal_service = MockTemporalService()
        contradiction_service = MockContradictionService(
            biomedlm_service=biomedlm_service,
            temporal_service=temporal_service
        )
    
    # Test each case
    for i, test_case in enumerate(TEST_CASES):
        logger.info(f"Test case {i+1}: {test_case['description']}")
        
        # Extract test data
        claim1 = test_case["claim1"]
        claim2 = test_case["claim2"]
        metadata1 = test_case["metadata1"]
        metadata2 = test_case["metadata2"]
        
        # Detect contradiction
        result = await contradiction_service.detect_contradiction(
            claim1=claim1,
            claim2=claim2,
            metadata1=metadata1,
            metadata2=metadata2,
            use_biomedlm=True,
            use_temporal=True,
            use_tsmixer=False
        )
        
        # Print results
        logger.info(f"Result: {'Contradiction' if result['is_contradiction'] else 'No contradiction'}")
        logger.info(f"Type: {result['contradiction_type']}")
        logger.info(f"Confidence: {result['confidence']}")
        logger.info(f"Methods used: {', '.join(result['methods_used'])}")
        
        if result["explanation"]:
            logger.info(f"Explanation: {result['explanation']}")
        
        # Print domain-specific details if available
        domain_info = result.get("details", {}).get("domain_info")
        if domain_info:
            logger.info("Domain-specific information:")
            logger.info(f"  Claim 1 domain: {domain_info['claim1']['domain']}")
            logger.info(f"  Claim 2 domain: {domain_info['claim2']['domain']}")
            logger.info(f"  Time difference: {domain_info['time_diff_years']:.1f} years")
        
        logger.info("---")
    
    logger.info("Domain-specific temporal contradiction tests completed")

async def main():
    """Main function."""
    logger.info("Starting domain-specific temporal contradiction tests...")
    
    try:
        # Test domain-specific temporal contradiction detection
        await test_domain_specific_temporal_contradiction()
        
        logger.info("All tests completed successfully")
    except Exception as e:
        logger.error(f"Error during tests: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
