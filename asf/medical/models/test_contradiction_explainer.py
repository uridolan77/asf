"""
Test Contradiction Explainer

This script tests the SHAP-based explainability for contradiction analysis results.
"""

import argparse
import logging
import json
import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test-explainer")

def test_contradiction_explainer(biomedlm_scorer, output_dir: str = None):
    """
    Test the SHAP-based contradiction explainer.
    
    Args:
        biomedlm_scorer: BioMedLMScorer instance
        output_dir: Directory to save visualizations
    """
    logger.info("Testing SHAP-based contradiction explainer...")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Test cases
    test_cases = [
        {
            "name": "negation_contradiction",
            "claim1": "Aspirin is effective for treating headaches.",
            "claim2": "Aspirin has no effect on headache symptoms.",
            "expected_contradiction": True,
            "expected_type": "negation"
        },
        {
            "name": "study_design_contradiction",
            "claim1": "A randomized controlled trial with 1000 patients found that the drug reduced mortality by 30%.",
            "claim2": "A small case series with 10 patients suggested that the drug might increase mortality.",
            "expected_contradiction": True,
            "expected_type": "multimodal"
        },
        {
            "name": "semantic_contradiction",
            "claim1": "The treatment significantly improved patient outcomes compared to placebo.",
            "claim2": "Patients receiving the treatment showed worse outcomes than those on placebo.",
            "expected_contradiction": True,
            "expected_type": "semantic"
        },
        {
            "name": "no_contradiction",
            "claim1": "Regular exercise improves cardiovascular health.",
            "claim2": "Physical activity has beneficial effects on heart health.",
            "expected_contradiction": False,
            "expected_type": "none"
        }
    ]
    
    # Run tests
    results = []
    for i, test_case in enumerate(test_cases):
        logger.info(f"Test case {i+1}: {test_case['name']}")
        logger.info(f"Claim 1: {test_case['claim1']}")
        logger.info(f"Claim 2: {test_case['claim2']}")
        logger.info(f"Expected contradiction: {test_case['expected_contradiction']}")
        logger.info(f"Expected type: {test_case['expected_type']}")
        
        # Generate explanation
        output_path = None
        if output_dir:
            output_path = os.path.join(output_dir, f"{test_case['name']}_explanation.png")
        
        explanation = biomedlm_scorer.explain_contradiction(
            test_case["claim1"], 
            test_case["claim2"],
            generate_visualization=True,
            output_path=output_path
        )
        
        # Check if contradiction was detected
        contradiction_detected = explanation.get("contradiction_detected", False)
        
        logger.info(f"Contradiction detected: {contradiction_detected}")
        logger.info(f"Explanation type: {explanation.get('type', 'unknown')}")
        
        # Print explanation summary
        if "summary" in explanation:
            logger.info("Explanation summary:")
            logger.info(explanation["summary"])
        elif "explanation" in explanation:
            logger.info("Explanation:")
            logger.info(explanation["explanation"])
        
        # Check if visualization was generated
        if "visualization_path" in explanation:
            logger.info(f"Visualization saved to: {explanation['visualization_path']}")
        
        logger.info("")
        
        # Store result
        test_result = {
            "name": test_case["name"],
            "claim1": test_case["claim1"],
            "claim2": test_case["claim2"],
            "expected_contradiction": test_case["expected_contradiction"],
            "expected_type": test_case["expected_type"],
            "contradiction_detected": contradiction_detected,
            "explanation_type": explanation.get("type", "unknown"),
            "has_visualization": "visualization_path" in explanation,
            "success": contradiction_detected == test_case["expected_contradiction"]
        }
        results.append(test_result)
    
    # Calculate success rate
    success_count = sum(1 for result in results if result["success"])
    success_rate = success_count / len(results)
    
    logger.info(f"Contradiction explanation success rate: {success_rate:.2%}")
    
    return results

def test_explanation_components(biomedlm_scorer):
    """
    Test individual components of the explanation system.
    
    Args:
        biomedlm_scorer: BioMedLMScorer instance
    """
    logger.info("Testing explanation components...")
    
    # Test negation explanation
    logger.info("Testing negation explanation...")
    negation_test = {
        "claim1": "The study found a significant effect of the treatment.",
        "claim2": "The study found no significant effect of the treatment."
    }
    
    # Get contradiction result with negation detection
    contradiction_result = biomedlm_scorer.detect_contradiction_with_negation(
        negation_test["claim1"], 
        negation_test["claim2"]
    )
    
    # Check if negation analysis is available
    if "negation_analysis" in contradiction_result:
        logger.info("Negation analysis available:")
        negation_analysis = contradiction_result["negation_analysis"]
        
        if "contradictions" in negation_analysis:
            logger.info(f"Found {len(negation_analysis['contradictions'])} negation contradictions:")
            for i, contradiction in enumerate(negation_analysis["contradictions"]):
                logger.info(f"  {i+1}. {contradiction}")
    else:
        logger.info("Negation analysis not available")
    
    logger.info("")
    
    # Test multimodal explanation
    logger.info("Testing multimodal explanation...")
    multimodal_test = {
        "claim1": "A randomized controlled trial with 500 patients showed a 20% reduction in symptoms.",
        "claim2": "A case report of 3 patients suggested the treatment might reduce symptoms."
    }
    
    # Get contradiction result with multimodal fusion
    contradiction_result = biomedlm_scorer.detect_contradiction_multimodal(
        multimodal_test["claim1"], 
        multimodal_test["claim2"]
    )
    
    # Check if metadata is available
    if "metadata1" in contradiction_result and "metadata2" in contradiction_result:
        logger.info("Metadata available:")
        
        # Print metadata for claim 1
        metadata1 = contradiction_result["metadata1"]
        study_design1 = metadata1.get("study_design", {}).get("study_design", "unknown")
        design_score1 = metadata1.get("study_design", {}).get("design_score", 0.0)
        sample_size1 = metadata1.get("sample_size", {}).get("sample_size", 0)
        
        logger.info(f"Claim 1 metadata:")
        logger.info(f"  Study design: {study_design1} (score: {design_score1:.1f})")
        logger.info(f"  Sample size: {sample_size1}")
        
        # Print metadata for claim 2
        metadata2 = contradiction_result["metadata2"]
        study_design2 = metadata2.get("study_design", {}).get("study_design", "unknown")
        design_score2 = metadata2.get("study_design", {}).get("design_score", 0.0)
        sample_size2 = metadata2.get("sample_size", {}).get("sample_size", 0)
        
        logger.info(f"Claim 2 metadata:")
        logger.info(f"  Study design: {study_design2} (score: {design_score2:.1f})")
        logger.info(f"  Sample size: {sample_size2}")
    else:
        logger.info("Metadata not available")
    
    logger.info("")
    
    # Test SHAP explanation
    logger.info("Testing SHAP explanation...")
    shap_test = {
        "claim1": "The drug significantly reduced blood pressure in hypertensive patients.",
        "claim2": "The medication had no effect on blood pressure in patients with hypertension."
    }
    
    # Generate explanation with SHAP
    explanation = biomedlm_scorer.explain_contradiction(
        shap_test["claim1"], 
        shap_test["claim2"]
    )
    
    # Check if SHAP values are available
    if "type" in explanation and explanation["type"] == "shap" and "shap_values" in explanation:
        logger.info("SHAP explanation available:")
        
        if "top_words" in explanation:
            logger.info("Top influential words:")
            for i, (word, importance) in enumerate(explanation["top_words"]):
                logger.info(f"  {i+1}. '{word}' (importance: {importance:.4f})")
    else:
        logger.info("SHAP explanation not available")
    
    logger.info("")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test contradiction explainer")
    parser.add_argument("--model", type=str, default="microsoft/BioMedLM", help="Model name")
    parser.add_argument("--output-dir", type=str, default="./explanations", help="Output directory for visualizations")
    parser.add_argument("--test-components", action="store_true", help="Test individual explanation components")
    parser.add_argument("--save-results", type=str, help="Path to save test results")
    
    args = parser.parse_args()
    
    try:
        # Import BioMedLMScorer
        from asf.medical.models.biomedlm_wrapper import BioMedLMScorer
        
        # Initialize BioMedLMScorer
        logger.info(f"Initializing BioMedLMScorer with model: {args.model}")
        biomedlm_scorer = BioMedLMScorer(
            model_name=args.model,
            use_negation_detection=True,
            use_multimodal_fusion=True,
            use_shap_explainer=True
        )
        
        # Run tests
        results = test_contradiction_explainer(biomedlm_scorer, args.output_dir)
        
        # Test individual components if requested
        if args.test_components:
            test_explanation_components(biomedlm_scorer)
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Test results saved to {args.save_results}")
        
        logger.info("All tests completed.")
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
