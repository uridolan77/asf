"""
Contradiction Visualizer for the Medical Research Synthesizer.

This module provides utilities for visualizing contradiction analysis results,
including SHAP-based explanations, negation-based explanations, and multimodal
explanations. It generates visual representations of why two medical claims
were determined to be contradictory, helping users understand the reasoning
behind the contradiction detection.

The module includes:
- Visualization of SHAP values showing word importance for contradiction detection
- Visualization of negated elements in contradictory claims
- Visualization of study design and sample size differences in contradictory claims
- Utilities for saving visualizations to disk and managing output directories

These visualizations are designed to be informative, accessible, and suitable
for inclusion in reports and presentations.
"""

import logging
import os
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("contradiction-visualizer")

class ContradictionVisualizer:
    """
    Visualizer for contradiction analysis results.

    This class provides methods for visualizing contradiction analysis results,
    including SHAP-based explanations, negation-based explanations, and multimodal
    explanations. It helps users understand why two medical claims were determined
    to be contradictory by generating visual representations of the factors that
    contributed to the contradiction detection.

    The class supports multiple types of visualizations:
    - SHAP-based: Shows word importance for contradiction detection
    - Negation-based: Shows negated elements in contradictory claims
    - Multimodal: Shows study design and sample size differences
    """

    def __init__(self, output_dir: str = "./visualizations"):
        """
        Initialize the contradiction visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    def visualize_shap_explanation(self, explanation: Dict[str, Any], output_path: Optional[str] = None) -> Optional[str]:
        """
        Visualize SHAP-based explanation.

        Args:
            explanation: Explanation dictionary
            output_path: Path to save the visualization

        Returns:
            Path to the saved visualization or None if visualization failed
        """
        try:
            if "type" not in explanation or explanation["type"] != "shap" or "shap_values" not in explanation:
                logger.warning("SHAP values not available for visualization")
                return None

            import matplotlib.pyplot as plt

            shap_data = explanation["shap_values"]
            values = np.array(shap_data["values"])
            data = shap_data["data"]

            claim1 = explanation.get("claim1", "")
            claim2 = explanation.get("claim2", "")

            plt.figure(figsize=(12, 8))

            plt.suptitle("SHAP-based Contradiction Explanation", fontsize=16)

            plt.figtext(0.5, 0.92, f"Claim 1: {claim1}", ha="center", fontsize=10, wrap=True)
            plt.figtext(0.5, 0.88, f"Claim 2: {claim2}", ha="center", fontsize=10, wrap=True)

            cmap = plt.cm.coolwarm

            norm = plt.Normalize(vmin=-np.max(np.abs(values)), vmax=np.max(np.abs(values)))

            ax = plt.subplot(111)
            ax.set_title("Word Importance for Contradiction Detection", fontsize=14)

            combined_text = data[0] + " [SEP] " + data[1]

            words = combined_text.split()

            max_words = 50
            if len(words) > max_words:
                words = words[:max_words]
                values = values[:max_words]

            y_pos = np.arange(len(words))
            ax.barh(y_pos, values, color=[cmap(norm(v)) for v in values])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel("SHAP Value (Impact on Contradiction Score)")

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label("Impact on Contradiction Detection")

            if "summary" in explanation:
                summary = explanation["summary"]
                if len(summary) > 500:
                    summary = summary[:500] + "..."
                plt.figtext(0.5, 0.05, summary, ha="center", fontsize=10, wrap=True)

            plt.tight_layout(rect=[0, 0.1, 1, 0.85])

            if output_path is None:
                output_path = os.path.join(self.output_dir, "shap_explanation.png")

            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()

            logger.info(f"SHAP visualization saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error visualizing SHAP explanation: {e}")
            return None

    def visualize_negation_explanation(self, explanation: Dict[str, Any], output_path: Optional[str] = None) -> Optional[str]:
        """
        Visualize negation-based explanation.

        Args:
            explanation: Explanation dictionary
            output_path: Path to save the visualization

        Returns:
            Path to the saved visualization or None if visualization failed
        """
        try:
            if "type" not in explanation or explanation["type"] != "negation" or "contradictions" not in explanation:
                logger.warning("Negation contradictions not available for visualization")
                return None

            claim1 = explanation.get("claim1", "")
            claim2 = explanation.get("claim2", "")

            contradictions = explanation["contradictions"]

            plt.figure(figsize=(10, 6))

            plt.suptitle("Negation-based Contradiction Explanation", fontsize=16)

            plt.figtext(0.5, 0.92, f"Claim 1: {claim1}", ha="center", fontsize=10, wrap=True)
            plt.figtext(0.5, 0.88, f"Claim 2: {claim2}", ha="center", fontsize=10, wrap=True)

            ax = plt.subplot(111)
            ax.set_title("Negated Elements Comparison", fontsize=14)

            elements = []
            negated_in_text1 = []
            negated_in_text2 = []

            for contradiction in contradictions:
                if "entity" in contradiction:
                    elements.append(contradiction["entity"])
                elif "word" in contradiction:
                    elements.append(contradiction["word"])
                else:
                    continue

                negated_in_text1.append(contradiction.get("negated_in_text1", False))
                negated_in_text2.append(contradiction.get("negated_in_text2", False))

            x = np.arange(len(elements))
            width = 0.35

            ax.bar(x - width/2, [1 if neg else 0 for neg in negated_in_text1], width, label="Negated in Claim 1")
            ax.bar(x + width/2, [1 if neg else 0 for neg in negated_in_text2], width, label="Negated in Claim 2")

            ax.set_xticks(x)
            ax.set_xticklabels(elements)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["No", "Yes"])
            ax.set_ylabel("Is Negated")
            ax.legend()

            if "summary" in explanation:
                summary = explanation["summary"]
                if len(summary) > 500:
                    summary = summary[:500] + "..."
                plt.figtext(0.5, 0.05, summary, ha="center", fontsize=10, wrap=True)

            plt.tight_layout(rect=[0, 0.1, 1, 0.85])

            if output_path is None:
                output_path = os.path.join(self.output_dir, "negation_explanation.png")

            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()

            logger.info(f"Negation visualization saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error visualizing negation explanation: {e}")
            return None

    def visualize_multimodal_explanation(self, explanation: Dict[str, Any], output_path: Optional[str] = None) -> Optional[str]:
        """
        Visualize multimodal-based explanation.

        Args:
            explanation: Explanation dictionary
            output_path: Path to save the visualization

        Returns:
            Path to the saved visualization or None if visualization failed
        """
        try:
            if "type" not in explanation or explanation["type"] != "multimodal" or "study_design_comparison" not in explanation:
                logger.warning("Study design comparison not available for visualization")
                return None

            claim1 = explanation.get("claim1", "")
            claim2 = explanation.get("claim2", "")

            comparison = explanation["study_design_comparison"]

            plt.figure(figsize=(10, 8))

            plt.suptitle("Multimodal Contradiction Explanation", fontsize=16)

            plt.figtext(0.5, 0.92, f"Claim 1: {claim1}", ha="center", fontsize=10, wrap=True)
            plt.figtext(0.5, 0.88, f"Claim 2: {claim2}", ha="center", fontsize=10, wrap=True)

            ax1 = plt.subplot(211)
            ax1.set_title("Study Design Comparison", fontsize=14)

            claim1_data = comparison["claim1"]
            claim2_data = comparison["claim2"]

            study_design1 = claim1_data.get("study_design", "unknown")
            design_score1 = claim1_data.get("design_score", 0.0)

            study_design2 = claim2_data.get("study_design", "unknown")
            design_score2 = claim2_data.get("design_score", 0.0)

            x = [0, 1]
            design_scores = [design_score1, design_score2]

            ax1.bar(x, design_scores)
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"Claim 1\n({study_design1})", f"Claim 2\n({study_design2})"])
            ax1.set_ylabel("Study Design Score")
            ax1.set_ylim(0, 5.5)

            ax2 = plt.subplot(212)
            ax2.set_title("Sample Size Comparison", fontsize=14)

            sample_size1 = claim1_data.get("sample_size", 0)
            sample_size2 = claim2_data.get("sample_size", 0)

            sample_sizes = [sample_size1, sample_size2]

            ax2.bar(x, sample_sizes)
            ax2.set_xticks(x)
            ax2.set_xticklabels(["Claim 1", "Claim 2"])
            ax2.set_ylabel("Sample Size")

            if max(sample_sizes) / (min(sample_sizes) + 1) > 10:
                ax2.set_yscale("log")

            if "summary" in explanation:
                summary = explanation["summary"]
                if len(summary) > 500:
                    summary = summary[:500] + "..."
                plt.figtext(0.5, 0.05, summary, ha="center", fontsize=10, wrap=True)

            plt.tight_layout(rect=[0, 0.1, 1, 0.85])

            if output_path is None:
                output_path = os.path.join(self.output_dir, "multimodal_explanation.png")

            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()

            logger.info(f"Multimodal visualization saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error visualizing multimodal explanation: {e}")
            return None

    def visualize_explanation(self, explanation: Dict[str, Any], output_path: Optional[str] = None) -> Optional[str]:
        """
        Visualize contradiction explanation.

        Args:
            explanation: Explanation dictionary
            output_path: Path to save the visualization

        Returns:
            Path to the saved visualization or None if visualization failed
        """
        explanation_type = explanation.get("type", "unknown")

        if explanation_type == "shap":
            return self.visualize_shap_explanation(explanation, output_path)
        elif explanation_type == "negation":
            return self.visualize_negation_explanation(explanation, output_path)
        elif explanation_type == "multimodal":
            return self.visualize_multimodal_explanation(explanation, output_path)
        else:
            logger.warning(f"Unsupported explanation type: {explanation_type}")
            return None

    def visualize_contradictions(self, contradictions: List[Dict[str, Any]], output_dir: Optional[str] = None) -> List[str]:
        """
        Visualize multiple contradiction explanations.

        Args:
            contradictions: List of contradiction dictionaries
            output_dir: Directory to save visualizations

        Returns:
            List of paths to saved visualizations
        """
        if output_dir is None:
            output_dir = self.output_dir

        os.makedirs(output_dir, exist_ok=True)

        visualization_paths = []

        for i, contradiction in enumerate(contradictions):
            if "explanation" not in contradiction:
                continue

            explanation = contradiction["explanation"]

            output_path = os.path.join(output_dir, f"contradiction_{i+1}.png")

            vis_path = self.visualize_explanation(explanation, output_path)

            if vis_path:
                visualization_paths.append(vis_path)

        return visualization_paths
