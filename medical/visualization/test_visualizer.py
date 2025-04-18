"""
Test Contradiction Visualizer for the Medical Research Synthesizer.

This script tests the visualization capabilities for contradiction explanations.
It provides functions for testing the ContradictionVisualizer with both generated
explanations from the BioMedLMScorer and pre-defined explanations from a JSON file.

The script includes test cases for different types of contradictions:
- Negation-based contradictions (e.g., "X is effective" vs "X has no effect")
- Study design contradictions (e.g., RCT with large sample vs case series with small sample)
- Semantic contradictions (e.g., "improved outcomes" vs "worse outcomes")

The test results are saved as image files in the specified output directory.
"""

import argparse
import logging
import json
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test-visualizer")

def test_visualizer(biomedlm_scorer, output_dir: str = "./visualizations"):
    """
    Test the contradiction visualizer with generated explanations.

    This function tests the ContradictionVisualizer with explanations generated
    by the BioMedLMScorer for a set of predefined test cases. It creates
    visualizations for different types of contradictions and saves them to
    the specified output directory.

    Args:
        biomedlm_scorer: BioMedLMScorer instance used to generate explanations
        output_dir: Directory to save the generated visualization images

    Returns:
        List of paths to the generated visualization images
    """
    logger.info("Testing contradiction visualizer...")

    from ..visualization.contradiction_visualizer import ContradictionVisualizer

    visualizer = ContradictionVisualizer(output_dir=output_dir)

    test_cases = [
        {
            "name": "negation_contradiction",
            "claim1": "Aspirin is effective for treating headaches.",
            "claim2": "Aspirin has no effect on headache symptoms.",
            "expected_type": "negation"
        },
        {
            "name": "study_design_contradiction",
            "claim1": "A randomized controlled trial with 1000 patients found that the drug reduced mortality by 30%.",
            "claim2": "A small case series with 10 patients suggested that the drug might increase mortality.",
            "expected_type": "multimodal"
        },
        {
            "name": "semantic_contradiction",
            "claim1": "The treatment significantly improved patient outcomes compared to placebo.",
            "claim2": "Patients receiving the treatment showed worse outcomes than those on placebo.",
            "expected_type": "shap"
        }
    ]

    visualization_paths = []

    for i, test_case in enumerate(test_cases):
        logger.info(f"Test case {i+1}: {test_case['name']}")
        logger.info(f"Claim 1: {test_case['claim1']}")
        logger.info(f"Claim 2: {test_case['claim2']}")
        logger.info(f"Expected type: {test_case['expected_type']}")

        explanation = biomedlm_scorer.explain_contradiction(
            test_case["claim1"],
            test_case["claim2"]
        )

        if not explanation:
            logger.warning("No explanation generated")
            continue

        output_path = os.path.join(output_dir, f"{test_case['name']}.png")

        vis_path = visualizer.visualize_explanation(explanation, output_path)

        if vis_path:
            logger.info(f"Visualization saved to: {vis_path}")
            visualization_paths.append(vis_path)
        else:
            logger.warning("Visualization failed")

        logger.info("")

    return visualization_paths

def test_from_json(json_file: str, output_dir: str = "./visualizations"):
    """
    Test the visualizer using explanations from a JSON file.

    This function tests the ContradictionVisualizer with explanations loaded
    from a JSON file. This is useful for testing with pre-generated or manually
    created explanation data without requiring the BioMedLMScorer.

    Args:
        json_file: Path to JSON file containing contradiction explanations
        output_dir: Directory to save the generated visualization images

    Returns:
        List of paths to the generated visualization images
    """
    logger.info(f"Testing visualizer with explanations from {json_file}...")

    from ..visualization.contradiction_visualizer import ContradictionVisualizer

    visualizer = ContradictionVisualizer(output_dir=output_dir)

    with open(json_file, "r") as f:
        explanations = json.load(f)

    visualization_paths = []

    for i, explanation in enumerate(explanations):
        logger.info(f"Explanation {i+1}")

        output_path = os.path.join(output_dir, f"explanation_{i+1}.png")

        vis_path = visualizer.visualize_explanation(explanation, output_path)

        if vis_path:
            logger.info(f"Visualization saved to: {vis_path}")
            visualization_paths.append(vis_path)
        else:
            logger.warning("Visualization failed")

        logger.info("")

    return visualization_paths

def main():
    """
    Main function to run the visualization tests.

    This function parses command-line arguments, sets up the test environment,
    and runs either the test_visualizer or test_from_json function based on
    the provided arguments. It handles exceptions and reports the results.

    Command-line arguments:
        --model: Model name for BioMedLMScorer (default: microsoft/BioMedLM)
        --output-dir: Output directory for visualizations (default: ./visualizations)
        --json-file: Path to JSON file with explanations (optional)

    Returns:
        0 if tests completed successfully, 1 if an error occurred
    """
    parser = argparse.ArgumentParser(description="Test contradiction visualizer")
    parser.add_argument("--model", type=str, default="microsoft/BioMedLM", help="Model name")
    parser.add_argument("--output-dir", type=str, default="./visualizations", help="Output directory for visualizations")
    parser.add_argument("--json-file", type=str, help="Path to JSON file with explanations")

    args = parser.parse_args()

    try:
        os.makedirs(args.output_dir, exist_ok=True)

        if args.json_file:
            visualization_paths = test_from_json(args.json_file, args.output_dir)
        else:
            from medical.models.biomedlm_wrapper import BioMedLMScorer

            logger.info(f"Initializing BioMedLMScorer with model: {args.model}")
            biomedlm_scorer = BioMedLMScorer(
                model_name=args.model,
                use_negation_detection=True,
                use_multimodal_fusion=True,
                use_shap_explainer=True
            )

            visualization_paths = test_visualizer(biomedlm_scorer, args.output_dir)

        logger.info(f"Generated {len(visualization_paths)} visualizations")
        for path in visualization_paths:
            logger.info(f"  - {path}")

        logger.info("All tests completed.")
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
