"""
SHAP-based Explainability for Contradiction Analysis

This module provides SHAP-based explainability for contradiction analysis
in the ASF framework. It uses SHAP (SHapley Additive exPlanations) values
to identify the most influential words and phrases that contributed to
the contradiction detection.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import re
from dataclasses import dataclass, field

# Import SHAP for explainability
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logging.warning("SHAP not installed. Explainability features will be limited.")

# Configure logging
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
        """Convert explanation to JSON string."""
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
    """

    def __init__(self, model, tokenizer, device: str = "cuda" if torch.cuda.is_available() else "cpu", use_precision_weighting: bool = True):
        """
        Initialize the contradiction explainer.

        Args:
            model: Model for contradiction detection
            tokenizer: Tokenizer for the model
            device: Device to run the model on
            use_precision_weighting: Whether to use precision weighting
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_precision_weighting = use_precision_weighting

        # Initialize SHAP explainer if available
        self.shap_explainer = None
        if HAS_SHAP:
            try:
                self.shap_explainer = shap.Explainer(
                    model=self._model_predict,
                    masker=shap.maskers.Text(tokenizer)
                )
                logger.info("SHAP explainer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SHAP explainer: {e}")

        # Initialize precision weighter if requested
        self.precision_weighter = None
        if self.use_precision_weighting:
            try:
                from asf.medical.models.precision_weighting import PrecisionWeighter
                self.precision_weighter = PrecisionWeighter()
                logger.info("Precision weighter initialized successfully")
            except ImportError as e:
                logger.warning(f"Failed to import precision weighter: {e}. Continuing without precision weighting.")
                self.use_precision_weighting = False
            except Exception as e:
                logger.warning(f"Failed to initialize precision weighter: {e}. Continuing without precision weighting.")
                self.use_precision_weighting = False

    def _model_predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict contradiction scores for a list of texts.

        Args:
            texts: List of texts to predict

        Returns:
            Array of contradiction scores
        """
        # This is a wrapper function for the model that SHAP can use
        # It should return a numpy array of scores

        # Process texts in batches to avoid memory issues
        batch_size = 8
        all_scores = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenize the texts
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Assuming the model outputs logits for contradiction
                # Adjust this based on your model's output format
                if hasattr(outputs, "logits"):
                    scores = outputs.logits[:, 1].cpu().numpy()  # Assuming index 1 is contradiction
                else:
                    scores = outputs.cpu().numpy()

            all_scores.append(scores)

        return np.concatenate(all_scores)

    def explain_contradiction(
        self,
        claim1: str,
        claim2: str,
        contradiction_score: float,
        use_shap: bool = True,
        use_negation_detection: bool = True,
        use_multimodal_factors: bool = False,
        metadata: Dict[str, Any] = None
    ) -> ContradictionExplanation:
        """
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
        """
        explanation = ContradictionExplanation(
            claim1=claim1,
            claim2=claim2,
            contradiction_score=contradiction_score,
            explanation_type="combined"
        )

        # Use SHAP for explanation
        if use_shap and HAS_SHAP and self.shap_explainer is not None:
            try:
                # Combine claims for SHAP analysis
                combined_text = f"{claim1} [SEP] {claim2}"

                # Compute SHAP values
                shap_values = self.shap_explainer([combined_text])

                # Extract influential words
                influential_words = {}

                # Process SHAP values
                for i, word in enumerate(shap_values.data[0]):
                    if word.strip():
                        influential_words[word] = float(shap_values.values[0][i])

                # Sort words by absolute SHAP value
                sorted_words = sorted(
                    influential_words.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )

                # Keep top 10 influential words
                explanation.influential_words = dict(sorted_words[:10])

                # Apply precision weighting if available
                if self.use_precision_weighting and self.precision_weighter is not None:
                    try:
                        # Weight SHAP values based on feature precision
                        weighted_values = self.precision_weighter.weight_shap_values(explanation.influential_words)

                        # Sort weighted values
                        sorted_weighted = sorted(
                            weighted_values.items(),
                            key=lambda x: abs(x[1]),
                            reverse=True
                        )

                        # Keep top 10 weighted influential words
                        explanation.weighted_influential_words = dict(sorted_weighted[:10])

                        # Add precision weights to explanation
                        explanation.precision_weights = self.precision_weighter.get_precision_weights().to_dict()

                        # Update prediction errors
                        prediction_error = abs(contradiction_score - 0.5) * 2  # Scale to [0, 1]
                        self.precision_weighter.update_precision_weights(
                            prediction_error=prediction_error,
                            features=list(explanation.influential_words.keys())
                        )

                        logger.info("Precision weighting applied to SHAP values")
                    except Exception as e:
                        logger.error(f"Failed to apply precision weighting: {e}")

                # Prepare visualization data
                explanation.visualization_data["shap_values"] = {
                    "values": shap_values.values[0].tolist(),
                    "data": shap_values.data[0]
                }

                logger.info(f"SHAP explanation generated for claims: {claim1[:30]}... and {claim2[:30]}...")

            except Exception as e:
                logger.error(f"Failed to generate SHAP explanation: {e}")

        # Use negation detection
        if use_negation_detection:
            try:
                negation_patterns = self._detect_negation_patterns(claim1, claim2)
                explanation.negation_patterns = negation_patterns

                if negation_patterns:
                    logger.info(f"Negation patterns detected: {len(negation_patterns)}")

            except Exception as e:
                logger.error(f"Failed to detect negation patterns: {e}")

        # Use multimodal factors
        if use_multimodal_factors and metadata:
            try:
                multimodal_factors = self._analyze_multimodal_factors(metadata)
                explanation.multimodal_factors = multimodal_factors

                if multimodal_factors:
                    logger.info(f"Multimodal factors analyzed: {len(multimodal_factors)}")

            except Exception as e:
                logger.error(f"Failed to analyze multimodal factors: {e}")

        return explanation

    def _detect_negation_patterns(self, claim1: str, claim2: str) -> List[Dict[str, Any]]:
        """
        Detect negation patterns between two claims.

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            List of detected negation patterns
        """
        negation_patterns = []

        # Simple negation words
        negation_words = ["not", "no", "never", "neither", "nor", "cannot", "can't", "doesn't", "don't", "didn't"]

        # Check for direct negations
        for word in negation_words:
            if word in claim1.lower() and word in claim2.lower():
                continue  # Both claims have the same negation, not interesting

            if word in claim1.lower():
                # Find the context around the negation
                pattern = re.compile(r'\b\w*\s*' + re.escape(word) + r'\s*\w*\b', re.IGNORECASE)
                matches = pattern.finditer(claim1.lower())

                for match in matches:
                    context = claim1[max(0, match.start() - 20):min(len(claim1), match.end() + 20)]
                    negation_patterns.append({
                        "type": "direct_negation",
                        "word": word,
                        "claim": "claim1",
                        "context": context
                    })

            if word in claim2.lower():
                # Find the context around the negation
                pattern = re.compile(r'\b\w*\s*' + re.escape(word) + r'\s*\w*\b', re.IGNORECASE)
                matches = pattern.finditer(claim2.lower())

                for match in matches:
                    context = claim2[max(0, match.start() - 20):min(len(claim2), match.end() + 20)]
                    negation_patterns.append({
                        "type": "direct_negation",
                        "word": word,
                        "claim": "claim2",
                        "context": context
                    })

        # Check for antonyms (simplified approach)
        antonym_pairs = [
            ("increase", "decrease"),
            ("positive", "negative"),
            ("high", "low"),
            ("more", "less"),
            ("greater", "smaller"),
            ("better", "worse"),
            ("improve", "worsen"),
            ("effective", "ineffective"),
            ("significant", "insignificant"),
            ("beneficial", "harmful")
        ]

        for word1, word2 in antonym_pairs:
            if (word1 in claim1.lower() and word2 in claim2.lower()) or (word2 in claim1.lower() and word1 in claim2.lower()):
                negation_patterns.append({
                    "type": "antonym",
                    "word1": word1,
                    "word2": word2,
                    "context1": self._get_context(claim1, word1) if word1 in claim1.lower() else self._get_context(claim1, word2),
                    "context2": self._get_context(claim2, word2) if word2 in claim2.lower() else self._get_context(claim2, word1)
                })

        return negation_patterns

    def _get_context(self, text: str, word: str, window: int = 20) -> str:
        """
        Get context around a word in a text.

        Args:
            text: Text to search in
            word: Word to find
            window: Context window size

        Returns:
            Context around the word
        """
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        match = pattern.search(text.lower())

        if match:
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            return text[start:end]

        return ""

    def _analyze_multimodal_factors(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze multimodal factors for contradiction.

        Args:
            metadata: Metadata for the claims

        Returns:
            Dictionary of multimodal factors
        """
        multimodal_factors = {}

        # Check for study design differences
        if "study_design1" in metadata and "study_design2" in metadata:
            design1 = metadata["study_design1"]
            design2 = metadata["study_design2"]

            if design1 != design2:
                multimodal_factors["study_design_difference"] = {
                    "design1": design1,
                    "design2": design2,
                    "impact": "high" if design1 in ["RCT", "meta-analysis"] or design2 in ["RCT", "meta-analysis"] else "medium"
                }

        # Check for sample size differences
        if "sample_size1" in metadata and "sample_size2" in metadata:
            size1 = metadata["sample_size1"]
            size2 = metadata["sample_size2"]

            if size1 and size2 and abs(size1 - size2) > 100:
                multimodal_factors["sample_size_difference"] = {
                    "size1": size1,
                    "size2": size2,
                    "ratio": max(size1, size2) / min(size1, size2) if min(size1, size2) > 0 else float('inf'),
                    "impact": "high" if max(size1, size2) / min(size1, size2) > 10 else "medium"
                }

        # Check for publication date differences
        if "publication_date1" in metadata and "publication_date2" in metadata:
            date1 = metadata["publication_date1"]
            date2 = metadata["publication_date2"]

            if date1 and date2:
                try:
                    year1 = int(date1.split("-")[0])
                    year2 = int(date2.split("-")[0])

                    if abs(year1 - year2) > 5:
                        multimodal_factors["publication_date_difference"] = {
                            "date1": date1,
                            "date2": date2,
                            "year_difference": abs(year1 - year2),
                            "impact": "medium" if abs(year1 - year2) > 10 else "low"
                        }
                except (ValueError, IndexError):
                    pass

        # Check for population differences
        if "population1" in metadata and "population2" in metadata:
            pop1 = metadata["population1"]
            pop2 = metadata["population2"]

            if pop1 and pop2 and pop1 != pop2:
                multimodal_factors["population_difference"] = {
                    "population1": pop1,
                    "population2": pop2,
                    "impact": "high"
                }

        return multimodal_factors

class ContradictionVisualizer:
    """
    Visualizer for contradiction explanations.

    This class provides methods for visualizing contradiction explanations
    using SHAP and other visualization techniques.
    """

    def __init__(self):
        """Initialize the contradiction visualizer."""
        self.has_shap = HAS_SHAP

    def visualize_shap(self, explanation: ContradictionExplanation, output_path: Optional[str] = None):
        """
        Visualize SHAP values for a contradiction explanation.

        Args:
            explanation: Contradiction explanation
            output_path: Path to save the visualization
        """
        if not self.has_shap:
            logger.warning("SHAP not installed. Cannot visualize SHAP values.")
            return

        if "shap_values" not in explanation.visualization_data:
            logger.warning("No SHAP values found in explanation.")
            return

        try:
            # Extract SHAP values and data
            values = np.array(explanation.visualization_data["shap_values"]["values"])
            data = explanation.visualization_data["shap_values"]["data"]

            # Create a SHAP object for visualization
            shap_values = shap.Explanation(
                values=values.reshape(1, -1),
                data=np.array([data]),
                feature_names=data
            )

            # Create visualization
            plt = shap.plots.text(shap_values, display=False)

            # Save or display the visualization
            if output_path:
                plt.savefig(output_path)
                logger.info(f"SHAP visualization saved to {output_path}")
            else:
                plt.show()

        except Exception as e:
            logger.error(f"Failed to visualize SHAP values: {e}")

    def generate_html_report(self, explanation: ContradictionExplanation, output_path: str):
        """
        Generate an HTML report for a contradiction explanation.

        Args:
            explanation: Contradiction explanation
            output_path: Path to save the HTML report
        """
        try:
            # Create HTML content
            html_content = f"""
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
            """

            # Add influential words section
            if explanation.influential_words:
                html_content += """
                    <div class="section">
                        <h2>Influential Words</h2>
                        <p>These words had the most influence on the contradiction detection:</p>
                        <div class="word-influence">
                """

                for word, value in explanation.influential_words.items():
                    css_class = "positive" if value > 0 else "negative" if value < 0 else "neutral"
                    html_content += f"""
                            <div class="word {css_class}">
                                {word} ({value:.4f})
                            </div>
                    """

                html_content += """
                        </div>
                    </div>
                """

            # Add weighted influential words section
            if hasattr(explanation, 'weighted_influential_words') and explanation.weighted_influential_words:
                html_content += """
                    <div class="section">
                        <h2>Weighted Influential Words</h2>
                        <p>These words had the most influence after applying precision weighting:</p>
                        <div class="word-influence">
                """

                for word, value in explanation.weighted_influential_words.items():
                    css_class = "positive" if value > 0 else "negative" if value < 0 else "neutral"
                    html_content += f"""
                            <div class="word {css_class}">
                                {word} ({value:.4f})
                            </div>
                    """

                html_content += """
                        </div>
                    </div>
                """

                # Add precision weights section
                if hasattr(explanation, 'precision_weights') and explanation.precision_weights:
                    html_content += """
                        <div class="section">
                            <h2>Precision Weights</h2>
                            <p>These weights were used to adjust the influence of different feature types:</p>
                            <table class="weights-table">
                                <tr>
                                    <th>Feature Type</th>
                                    <th>Weight</th>
                                </tr>
                    """

                    if 'feature_weights' in explanation.precision_weights:
                        for feature_type, weight in explanation.precision_weights['feature_weights'].items():
                            html_content += f"""
                                <tr>
                                    <td>{feature_type}</td>
                                    <td>{weight:.4f}</td>
                                </tr>
                            """

                    html_content += """
                            </table>
                            <p>Global Precision: {explanation.precision_weights.get('global_precision', 0.0):.4f}</p>
                        </div>
                    """

            # Add negation patterns section
            if explanation.negation_patterns:
                html_content += """
                    <div class="section">
                        <h2>Negation Patterns</h2>
                        <p>The following negation patterns were detected:</p>
                """

                for pattern in explanation.negation_patterns:
                    if pattern["type"] == "direct_negation":
                        html_content += f"""
                            <div class="negation">
                                <h3>Direct Negation</h3>
                                <p>Negation word: <strong>{pattern["word"]}</strong></p>
                                <p>Found in: Claim {pattern["claim"].split("claim")[1]}</p>
                                <p>Context: "{pattern["context"]}"</p>
                            </div>
                        """
                    elif pattern["type"] == "antonym":
                        html_content += f"""
                            <div class="negation">
                                <h3>Antonym Pair</h3>
                                <p>Words: <strong>{pattern["word1"]}</strong> vs <strong>{pattern["word2"]}</strong></p>
                                <p>Context in Claim 1: "{pattern["context1"]}"</p>
                                <p>Context in Claim 2: "{pattern["context2"]}"</p>
                            </div>
                        """

                html_content += """
                    </div>
                """

            # Add multimodal factors section
            if explanation.multimodal_factors:
                html_content += """
                    <div class="section">
                        <h2>Multimodal Factors</h2>
                        <p>The following multimodal factors contributed to the contradiction:</p>
                        <table>
                            <tr>
                                <th>Factor</th>
                                <th>Details</th>
                                <th>Impact</th>
                            </tr>
                """

                for factor, details in explanation.multimodal_factors.items():
                    factor_name = factor.replace("_", " ").title()
                    factor_details = ""

                    if factor == "study_design_difference":
                        factor_details = f"Claim 1: {details['design1']}, Claim 2: {details['design2']}"
                    elif factor == "sample_size_difference":
                        factor_details = f"Claim 1: {details['size1']}, Claim 2: {details['size2']}, Ratio: {details['ratio']:.2f}"
                    elif factor == "publication_date_difference":
                        factor_details = f"Claim 1: {details['date1']}, Claim 2: {details['date2']}, Difference: {details['year_difference']} years"
                    elif factor == "population_difference":
                        factor_details = f"Claim 1: {details['population1']}, Claim 2: {details['population2']}"

                    html_content += f"""
                            <tr>
                                <td>{factor_name}</td>
                                <td>{factor_details}</td>
                                <td>{details['impact'].title()}</td>
                            </tr>
                    """

                html_content += """
                        </table>
                    </div>
                """

            # Close HTML
            html_content += """
                </div>
            </body>
            </html>
            """

            # Write HTML to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"HTML report saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
