"""
SHAP-based Explainability for Contradiction Analysis
This module provides SHAP-based explainability for contradiction analysis
in the ASF framework. It uses SHAP (SHapley Additive exPlanations) values
to identify the most influential words and phrases that contributed to
the contradiction detection.
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass, field
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logging.warning("SHAP not installed. Explainability features will be limited.")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("shap-explainer")
@dataclass
class ContradictionExplanation:
    """Explanation for a contradiction between two claims."""
    claim1: str
    claim2: str
    contradiction_score: float
    explanation_type: str
    influential_words: Dict[str, float] = field(default_factory=dict)
    weighted_influential_words: Dict[str, float] = field(default_factory=dict)
    negation_patterns: List[Dict[str, Any]] = field(default_factory=list)
    multimodal_factors: Dict[str, Any] = field(default_factory=dict)
    visualization_data: Dict[str, Any] = field(default_factory=dict)
    precision_weights: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation to dictionary."""
        return {
            "claim1": self.claim1,
            "claim2": self.claim2,
            "contradiction_score": self.contradiction_score,
            "explanation_type": self.explanation_type,
            "influential_words": self.influential_words,
            "weighted_influential_words": self.weighted_influential_words,
            "negation_patterns": self.negation_patterns,
            "multimodal_factors": self.multimodal_factors,
            "visualization_data": self.visualization_data,
            "precision_weights": self.precision_weights
        }
    def to_json(self) -> str:
        """Convert explanation to JSON string.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        return json.dumps(self.to_dict(), indent=2)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContradictionExplanation':
        """Create explanation from dictionary."""
        return cls(
            claim1=data["claim1"],
            claim2=data["claim2"],
            contradiction_score=data["contradiction_score"],
            explanation_type=data["explanation_type"],
            influential_words=data.get("influential_words", {}),
            weighted_influential_words=data.get("weighted_influential_words", {}),
            negation_patterns=data.get("negation_patterns", []),
            multimodal_factors=data.get("multimodal_factors", {}),
            visualization_data=data.get("visualization_data", {}),
            precision_weights=data.get("precision_weights", {})
        )
    @classmethod
    def from_json(cls, json_str: str) -> 'ContradictionExplanation':
        """Create explanation from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
class ContradictionExplainer:
    """
    SHAP-based explainer for contradiction analysis.
    This class provides methods for explaining why two medical claims
    were determined to be contradictory using SHAP values.
        Initialize the contradiction explainer.
        Args:
            model: Model for contradiction detection
            tokenizer: Tokenizer for the model
            device: Device to run the model on
            use_precision_weighting: Whether to use precision weighting
        Predict contradiction scores for a list of texts.
        Args:
            texts: List of texts to predict
        Returns:
            Array of contradiction scores
        Explain why two claims are contradictory.
        Args:
            claim1: First claim
            claim2: Second claim
            contradiction_score: Contradiction score between the claims
            use_shap: Whether to use SHAP for explanation
            use_negation_detection: Whether to use negation detection
            use_multimodal_factors: Whether to use multimodal factors
            metadata: Additional metadata for explanation
        Returns:
            Contradiction explanation
        Detect negation patterns between two claims.
        Args:
            claim1: First claim
            claim2: Second claim
        Returns:
            List of detected negation patterns
        Get context around a word in a text.
        Args:
            text: Text to search in
            word: Word to find
            window: Context window size
        Returns:
            Context around the word
        Analyze multimodal factors for contradiction.
        Args:
            metadata: Metadata for the claims
        Returns:
            Dictionary of multimodal factors
    Visualizer for contradiction explanations.
    This class provides methods for visualizing contradiction explanations
    using SHAP and other visualization techniques.
        self.has_shap = HAS_SHAP
    def visualize_shap(self, explanation: ContradictionExplanation, output_path: Optional[str] = None):
        """
        Visualize SHAP values for a contradiction explanation.
        Args:
            explanation: Contradiction explanation
            output_path: Path to save the visualization
        Generate an HTML report for a contradiction explanation.
        Args:
            explanation: Contradiction explanation
            output_path: Path to save the HTML report
            <!DOCTYPE html>
            <html>
            <head>
                <title>Contradiction Explanation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .container {{ max-width: 800px; margin: 0 auto; }}
                    .header {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
                    .claims {{ margin: 20px 0; }}
                    .claim {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                    .score {{ font-weight: bold; color: #d9534f; }}
                    .section {{ margin: 20px 0; }}
                    .word-influence {{ display: flex; flex-wrap: wrap; }}
                    .word {{
                        margin: 5px; padding: 5px 10px; border-radius: 15px;
                        display: inline-block; font-size: 14px;
                    }}
                    .positive {{ background-color: #dff0d8; color: #3c763d; }}
                    .negative {{ background-color: #f2dede; color: #a94442; }}
                    .neutral {{ background-color: #f5f5f5; color: #777; }}
                    .negation {{ background-color: #fcf8e3; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                    .multimodal {{ background-color: #d9edf7; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Contradiction Explanation Report</h1>
                        <p>Explanation Type: {explanation.explanation_type}</p>
                        <p>Contradiction Score: <span class="score">{explanation.contradiction_score:.4f}</span></p>
                    </div>
                    <div class="claims">
                        <h2>Claims</h2>
                        <div class="claim">
                            <h3>Claim 1:</h3>
                            <p>{explanation.claim1}</p>
                        </div>
                        <div class="claim">
                            <h3>Claim 2:</h3>
                            <p>{explanation.claim2}</p>
                        </div>
                    </div>
                    <div class="section">
                        <h2>Influential Words</h2>
                        <p>These words had the most influence on the contradiction detection:</p>
                        <div class="word-influence">
                            <div class="word {css_class}">
                                {word} ({value:.4f})
                            </div>
                        </div>
                    </div>
                    <div class="section">
                        <h2>Weighted Influential Words</h2>
                        <p>These words had the most influence after applying precision weighting:</p>
                        <div class="word-influence">
                            <div class="word {css_class}">
                                {word} ({value:.4f})
                            </div>
                        </div>
                    </div>
                        <div class="section">
                            <h2>Precision Weights</h2>
                            <p>These weights were used to adjust the influence of different feature types:</p>
                            <table class="weights-table">
                                <tr>
                                    <th>Feature Type</th>
                                    <th>Weight</th>
                                </tr>
                                <tr>
                                    <td>{feature_type}</td>
                                    <td>{weight:.4f}</td>
                                </tr>
                            </table>
                            <p>Global Precision: {explanation.precision_weights.get('global_precision', 0.0):.4f}</p>
                        </div>
                    <div class="section">
                        <h2>Negation Patterns</h2>
                        <p>The following negation patterns were detected:</p>
                            <div class="negation">
                                <h3>Direct Negation</h3>
                                <p>Negation word: <strong>{pattern["word"]}</strong></p>
                                <p>Found in: Claim {pattern["claim"].split("claim")[1]}</p>
                                <p>Context: "{pattern["context"]}"</p>
                            </div>
                            <div class="negation">
                                <h3>Antonym Pair</h3>
                                <p>Words: <strong>{pattern["word1"]}</strong> vs <strong>{pattern["word2"]}</strong></p>
                                <p>Context in Claim 1: "{pattern["context1"]}"</p>
                                <p>Context in Claim 2: "{pattern["context2"]}"</p>
                            </div>
                    </div>
                    <div class="section">
                        <h2>Multimodal Factors</h2>
                        <p>The following multimodal factors contributed to the contradiction:</p>
                        <table>
                            <tr>
                                <th>Factor</th>
                                <th>Details</th>
                                <th>Impact</th>
                            </tr>
                            <tr>
                                <td>{factor_name}</td>
                                <td>{factor_details}</td>
                                <td>{details['impact'].title()}</td>
                            </tr>
                        </table>
                    </div>
                </div>
            </body>
            </html>