"""
Test Contradiction Detection

This script tests the contradiction detection capabilities of the BioMedLM wrapper,
including negation detection and multimodal fusion.
"""

import argparse
import logging
import json
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test-contradiction")

def test_basic_contradiction(biomedlm_scorer):
    """
    Test basic contradiction detection.
    
    Args:
        biomedlm_scorer: BioMedLMScorer instance
    """
    logger.info("Testing basic contradiction detection...")
    
    # Test cases
    test_cases = [
        {
            "claim1": "Aspirin is effective for treating headaches.",
            "claim2": "Aspirin has no effect on headache symptoms.",
            "expected_contradiction": True
        },
        {
            "claim1": "Regular exercise reduces the risk of heart disease.",
            "claim2": "Physical activity is beneficial for cardiovascular health.",
            "expected_contradiction": False
        },
        {
            "claim1": "Vitamin C prevents the common cold.",
            "claim2": "Vitamin C supplementation has no significant effect on cold prevention.",
            "expected_contradiction": True
        }
    ]
    
    # Run tests
    results = []
    for i, test_case in enumerate(test_cases):
        logger.info(f"Test case {i+1}:")
        logger.info(f"Claim 1: {test_case['claim1']}")
        logger.info(f"Claim 2: {test_case['claim2']}")
        
        # Get contradiction score
        scores = biomedlm_scorer.get_detailed_scores(test_case["claim1"], test_case["claim2"])
        
        # Check if contradiction was detected
        contradiction_detected = scores["contradiction_score"] > 0.7
        
        logger.info(f"Contradiction score: {scores['contradiction_score']:.4f}")
        logger.info(f"Contradiction detected: {contradiction_detected}")
        logger.info(f"Expected contradiction: {test_case['expected_contradiction']}")
        logger.info(f"Result: {'✓' if contradiction_detected == test_case['expected_contradiction'] else '✗'}")
        logger.info("")
        
        # Store result
        result = {
            "claim1": test_case["claim1"],
            "claim2": test_case["claim2"],
            "expected_contradiction": test_case["expected_contradiction"],
            "contradiction_detected": contradiction_detected,
            "contradiction_score": scores["contradiction_score"],
            "agreement_score": scores["agreement_score"],
            "success": contradiction_detected == test_case["expected_contradiction"]
        }
        results.append(result)
    
    # Calculate success rate
    success_count = sum(1 for result in results if result["success"])
    success_rate = success_count / len(results)
    
    logger.info(f"Basic contradiction detection success rate: {success_rate:.2%}")
    
    return results

def test_negation_detection(biomedlm_scorer):
    """
    Test negation-aware contradiction detection.
    
    Args:
        biomedlm_scorer: BioMedLMScorer instance
    """
    logger.info("Testing negation-aware contradiction detection...")
    
    # Test cases with negation
    test_cases = [
        {
            "claim1": "Patients with hypertension should take ACE inhibitors.",
            "claim2": "Patients with hypertension should not take ACE inhibitors.",
            "expected_contradiction": True
        },
        {
            "claim1": "There is no evidence that vitamin E prevents cancer.",
            "claim2": "Vitamin E has been shown to prevent cancer in clinical trials.",
            "expected_contradiction": True
        },
        {
            "claim1": "Smoking cessation does not increase the risk of depression.",
            "claim2": "Quitting smoking increases the risk of depression.",
            "expected_contradiction": True
        },
        {
            "claim1": "The study found no significant difference between the treatment and control groups.",
            "claim2": "The study found a significant difference between the treatment and control groups.",
            "expected_contradiction": True
        }
    ]
    
    # Run tests
    results = []
    for i, test_case in enumerate(test_cases):
        logger.info(f"Test case {i+1}:")
        logger.info(f"Claim 1: {test_case['claim1']}")
        logger.info(f"Claim 2: {test_case['claim2']}")
        
        # Get contradiction result using negation-aware detection
        result = biomedlm_scorer.detect_contradiction_with_negation(test_case["claim1"], test_case["claim2"])
        
        # Check if contradiction was detected
        contradiction_detected = result.get("has_contradiction", False)
        
        logger.info(f"Contradiction score: {result.get('contradiction_score', 0):.4f}")
        logger.info(f"Contradiction detected: {contradiction_detected}")
        logger.info(f"Expected contradiction: {test_case['expected_contradiction']}")
        logger.info(f"Method: {result.get('method', 'unknown')}")
        logger.info(f"Contradiction type: {result.get('contradiction_type', 'unknown')}")
        logger.info(f"Result: {'✓' if contradiction_detected == test_case['expected_contradiction'] else '✗'}")
        logger.info("")
        
        # Store result
        test_result = {
            "claim1": test_case["claim1"],
            "claim2": test_case["claim2"],
            "expected_contradiction": test_case["expected_contradiction"],
            "contradiction_detected": contradiction_detected,
            "contradiction_score": result.get("contradiction_score", 0),
            "method": result.get("method", "unknown"),
            "contradiction_type": result.get("contradiction_type", "unknown"),
            "success": contradiction_detected == test_case["expected_contradiction"]
        }
        results.append(test_result)
    
    # Calculate success rate
    success_count = sum(1 for result in results if result["success"])
    success_rate = success_count / len(results)
    
    logger.info(f"Negation-aware contradiction detection success rate: {success_rate:.2%}")
    
    return results

def test_multimodal_fusion(biomedlm_scorer):
    """
    Test multimodal fusion for contradiction detection.
    
    Args:
        biomedlm_scorer: BioMedLMScorer instance
    """
    logger.info("Testing multimodal fusion for contradiction detection...")
    
    # Test cases with study design and metadata
    test_cases = [
        {
            "claim1": "In a randomized controlled trial with 1000 patients, statin therapy was found to significantly reduce cardiovascular events.",
            "claim2": "A small observational study with 50 patients found no benefit of statin therapy on cardiovascular outcomes.",
            "expected_contradiction": True
        },
        {
            "claim1": "A meta-analysis of 15 randomized controlled trials (n=10,000) showed that aspirin reduces the risk of heart attack.",
            "claim2": "A single-center retrospective study (n=200) found that aspirin had no effect on heart attack risk.",
            "expected_contradiction": True
        },
        {
            "claim1": "In a double-blind placebo-controlled trial (n=500), the drug showed a 30% reduction in symptoms.",
            "claim2": "A larger multi-center randomized controlled trial (n=2000) found that the drug reduced symptoms by 25%.",
            "expected_contradiction": False
        }
    ]
    
    # Run tests
    results = []
    for i, test_case in enumerate(test_cases):
        logger.info(f"Test case {i+1}:")
        logger.info(f"Claim 1: {test_case['claim1']}")
        logger.info(f"Claim 2: {test_case['claim2']}")
        
        # Get contradiction result using multimodal fusion
        result = biomedlm_scorer.detect_contradiction_multimodal(test_case["claim1"], test_case["claim2"])
        
        # Check if contradiction was detected
        contradiction_detected = result.get("has_contradiction", False)
        
        logger.info(f"Contradiction score: {result.get('contradiction_score', 0):.4f}")
        logger.info(f"Multimodal score: {result.get('multimodal_contradiction_score', 0):.4f}")
        logger.info(f"Contradiction detected: {contradiction_detected}")
        logger.info(f"Expected contradiction: {test_case['expected_contradiction']}")
        logger.info(f"Method: {result.get('method', 'unknown')}")
        
        # Print metadata if available
        if "metadata1" in result:
            logger.info("Metadata for claim 1:")
            study_design1 = result["metadata1"].get("study_design", {}).get("study_design", "unknown")
            sample_size1 = result["metadata1"].get("sample_size", {}).get("sample_size", 0)
            logger.info(f"  Study design: {study_design1}")
            logger.info(f"  Sample size: {sample_size1}")
        
        if "metadata2" in result:
            logger.info("Metadata for claim 2:")
            study_design2 = result["metadata2"].get("study_design", {}).get("study_design", "unknown")
            sample_size2 = result["metadata2"].get("sample_size", {}).get("sample_size", 0)
            logger.info(f"  Study design: {study_design2}")
            logger.info(f"  Sample size: {sample_size2}")
        
        logger.info(f"Result: {'✓' if contradiction_detected == test_case['expected_contradiction'] else '✗'}")
        logger.info("")
        
        # Store result
        test_result = {
            "claim1": test_case["claim1"],
            "claim2": test_case["claim2"],
            "expected_contradiction": test_case["expected_contradiction"],
            "contradiction_detected": contradiction_detected,
            "contradiction_score": result.get("contradiction_score", 0),
            "multimodal_score": result.get("multimodal_contradiction_score", 0),
            "method": result.get("method", "unknown"),
            "metadata1": result.get("metadata1", {}),
            "metadata2": result.get("metadata2", {}),
            "success": contradiction_detected == test_case["expected_contradiction"]
        }
        results.append(test_result)
    
    # Calculate success rate
    success_count = sum(1 for result in results if result["success"])
    success_rate = success_count / len(results)
    
    logger.info(f"Multimodal fusion contradiction detection success rate: {success_rate:.2%}")
    
    return results

def test_combined_approach(biomedlm_scorer):
    """
    Test the combined approach for contradiction detection.
    
    Args:
        biomedlm_scorer: BioMedLMScorer instance
    """
    logger.info("Testing combined approach for contradiction detection...")
    
    # Test cases with various contradiction types
    test_cases = [
        {
            "claim1": "Aspirin is effective for treating headaches.",
            "claim2": "Aspirin has no effect on headache symptoms.",
            "expected_contradiction": True,
            "expected_type": "negation"
        },
        {
            "claim1": "A randomized controlled trial with 1000 patients found that the drug reduced mortality by 30%.",
            "claim2": "A small case series with 10 patients suggested that the drug might increase mortality.",
            "expected_contradiction": True,
            "expected_type": "multimodal"
        },
        {
            "claim1": "The study found no significant difference between the treatment and control groups.",
            "claim2": "The study found a significant difference between the treatment and control groups.",
            "expected_contradiction": True,
            "expected_type": "negation"
        },
        {
            "claim1": "Regular exercise improves cardiovascular health.",
            "claim2": "Physical activity has beneficial effects on heart health.",
            "expected_contradiction": False,
            "expected_type": "none"
        }
    ]
    
    # Run tests
    results = []
    for i, test_case in enumerate(test_cases):
        logger.info(f"Test case {i+1}:")
        logger.info(f"Claim 1: {test_case['claim1']}")
        logger.info(f"Claim 2: {test_case['claim2']}")
        logger.info(f"Expected contradiction: {test_case['expected_contradiction']}")
        logger.info(f"Expected type: {test_case['expected_type']}")
        
        # Get contradiction result using the combined approach
        result = biomedlm_scorer.detect_contradiction(test_case["claim1"], test_case["claim2"])
        
        # Check if contradiction was detected
        contradiction_detected = result.get("has_contradiction", False)
        contradiction_type = result.get("contradiction_type", "unknown")
        
        logger.info(f"Contradiction score: {result.get('contradiction_score', 0):.4f}")
        logger.info(f"Contradiction detected: {contradiction_detected}")
        logger.info(f"Contradiction type: {contradiction_type}")
        logger.info(f"Method: {result.get('method', 'unknown')}")
        
        # Check if the result matches the expected outcome
        contradiction_success = contradiction_detected == test_case["expected_contradiction"]
        type_success = test_case["expected_type"] == "none" or contradiction_type.lower().startswith(test_case["expected_type"].lower())
        overall_success = contradiction_success and (not contradiction_detected or type_success)
        
        logger.info(f"Result: {'✓' if overall_success else '✗'}")
        logger.info("")
        
        # Store result
        test_result = {
            "claim1": test_case["claim1"],
            "claim2": test_case["claim2"],
            "expected_contradiction": test_case["expected_contradiction"],
            "expected_type": test_case["expected_type"],
            "contradiction_detected": contradiction_detected,
            "contradiction_type": contradiction_type,
            "contradiction_score": result.get("contradiction_score", 0),
            "method": result.get("method", "unknown"),
            "contradiction_success": contradiction_success,
            "type_success": type_success,
            "overall_success": overall_success
        }
        results.append(test_result)
    
    # Calculate success rates
    contradiction_success_count = sum(1 for result in results if result["contradiction_success"])
    contradiction_success_rate = contradiction_success_count / len(results)
    
    type_success_count = sum(1 for result in results if result["contradiction_success"] and (not result["contradiction_detected"] or result["type_success"]))
    type_success_rate = type_success_count / contradiction_success_count if contradiction_success_count > 0 else 0
    
    overall_success_count = sum(1 for result in results if result["overall_success"])
    overall_success_rate = overall_success_count / len(results)
    
    logger.info(f"Contradiction detection success rate: {contradiction_success_rate:.2%}")
    logger.info(f"Contradiction type success rate: {type_success_rate:.2%}")
    logger.info(f"Overall success rate: {overall_success_rate:.2%}")
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test contradiction detection")
    parser.add_argument("--model", type=str, default="microsoft/BioMedLM", help="Model name")
    parser.add_argument("--output", type=str, help="Output file for test results")
    parser.add_argument("--test-basic", action="store_true", help="Test basic contradiction detection")
    parser.add_argument("--test-negation", action="store_true", help="Test negation-aware contradiction detection")
    parser.add_argument("--test-multimodal", action="store_true", help="Test multimodal fusion")
    parser.add_argument("--test-combined", action="store_true", help="Test combined approach")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # If no specific test is selected, run all tests
    if not (args.test_basic or args.test_negation or args.test_multimodal or args.test_combined):
        args.test_all = True
    
    try:
        # Import BioMedLMScorer
        from asf.medical.models.biomedlm_wrapper import BioMedLMScorer
        
        # Initialize BioMedLMScorer
        logger.info(f"Initializing BioMedLMScorer with model: {args.model}")
        biomedlm_scorer = BioMedLMScorer(
            model_name=args.model,
            use_negation_detection=True,
            use_multimodal_fusion=True
        )
        
        # Run tests
        results = {}
        
        if args.test_all or args.test_basic:
            results["basic"] = test_basic_contradiction(biomedlm_scorer)
        
        if args.test_all or args.test_negation:
            results["negation"] = test_negation_detection(biomedlm_scorer)
        
        if args.test_all or args.test_multimodal:
            results["multimodal"] = test_multimodal_fusion(biomedlm_scorer)
        
        if args.test_all or args.test_combined:
            results["combined"] = test_combined_approach(biomedlm_scorer)
        
        # Save results if output file is specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Test results saved to {args.output}")
        
        logger.info("All tests completed.")
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
