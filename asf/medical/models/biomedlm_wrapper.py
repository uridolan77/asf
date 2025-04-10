"""
BioMedLM Wrapper for Contradiction Scoring

This module provides a wrapper for Microsoft's BioMedLM model to score
contradictions between medical claims.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("biomedlm-wrapper")

class BioMedLMScorer:
    """
    Wrapper for Microsoft's BioMedLM model to score contradictions between medical claims.
    """

    def __init__(self, model_name: str = "microsoft/BioMedLM-2024", use_negation_detection: bool = True, use_multimodal_fusion: bool = True, use_shap_explainer: bool = True, use_tsmixer: bool = True, use_lorentz: bool = True, use_temporal_confidence: bool = True):
        """
        Initialize the BioMedLM scorer.

        Args:
            model_name: Name of the BioMedLM model to use
            use_negation_detection: Whether to use negation detection
            use_multimodal_fusion: Whether to use multimodal fusion
            use_shap_explainer: Whether to use SHAP-based explainability
            use_tsmixer: Whether to use TSMixer for temporal analysis
            use_lorentz: Whether to use Lorentz embeddings for hierarchical analysis
            use_temporal_confidence: Whether to use temporal confidence scoring
        """
        self.model_name = model_name
        self.use_negation_detection = use_negation_detection
        self.use_multimodal_fusion = use_multimodal_fusion
        self.use_shap_explainer = use_shap_explainer
        self.use_tsmixer = use_tsmixer
        self.use_lorentz = use_lorentz
        self.use_temporal_confidence = use_temporal_confidence
        self.negation_detector = None
        self.contradiction_detector = None
        self.multimodal_detector = None
        self.contradiction_explainer = None
        self.tsmixer_detector = None
        self.lorentz_detector = None
        self.temporal_confidence_scorer = None

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

            # Load model and tokenizer
            logger.info(f"Loading BioMedLM model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            logger.info("BioMedLM model loaded successfully")

            # Initialize negation detection if requested
            if self.use_negation_detection:
                try:
                    from asf.medical.models.negation_detector import NegationDetector, NegationAwareContradictionDetector

                    self.negation_detector = NegationDetector()
                    self.contradiction_detector = NegationAwareContradictionDetector(
                        negation_detector=self.negation_detector,
                        biomedlm_scorer=self
                    )

                    logger.info("Negation detection initialized successfully")
                except ImportError as e:
                    logger.warning(f"Failed to import negation detection: {e}. Continuing without negation detection.")
                    self.use_negation_detection = False
                except Exception as e:
                    logger.warning(f"Failed to initialize negation detection: {e}. Continuing without negation detection.")
                    self.use_negation_detection = False

            # Initialize multimodal fusion if requested
            if self.use_multimodal_fusion:
                try:
                    from asf.medical.models.multimodal_fusion import MetadataExtractor, MultimodalContradictionDetector

                    metadata_extractor = MetadataExtractor()
                    self.multimodal_detector = MultimodalContradictionDetector(
                        biomedlm_scorer=self,
                        metadata_extractor=metadata_extractor
                    )

                    logger.info("Multimodal fusion initialized successfully")
                except ImportError as e:
                    logger.warning(f"Failed to import multimodal fusion: {e}. Continuing without multimodal fusion.")
                    self.use_multimodal_fusion = False
                except Exception as e:
                    logger.warning(f"Failed to initialize multimodal fusion: {e}. Continuing without multimodal fusion.")
                    self.use_multimodal_fusion = False

            # Initialize SHAP-based explainer if requested
            if self.use_shap_explainer:
                try:
                    from asf.medical.models.shap_explainer import ContradictionExplainer

                    self.contradiction_explainer = ContradictionExplainer(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=self.device
                    )

                    logger.info("SHAP-based contradiction explainer initialized successfully")
                except ImportError as e:
                    logger.warning(f"Failed to import SHAP explainer: {e}. Continuing without SHAP explainability.")
                    self.use_shap_explainer = False
                except Exception as e:
                    logger.warning(f"Failed to initialize SHAP explainer: {e}. Continuing without SHAP explainability.")
                    self.use_shap_explainer = False

            # Initialize TSMixer detector if requested
            if self.use_tsmixer:
                try:
                    from asf.medical.models.tsmixer_contradiction_detector import TSMixerContradictionDetector

                    self.tsmixer_detector = TSMixerContradictionDetector(
                        biomedlm_scorer=self,
                        device=self.device,
                        config={"use_tsmixer": True}
                    )

                    logger.info("TSMixer contradiction detector initialized successfully")
                except ImportError as e:
                    logger.warning(f"Failed to import TSMixer detector: {e}. Continuing without TSMixer analysis.")
                    self.use_tsmixer = False
                except Exception as e:
                    logger.warning(f"Failed to initialize TSMixer detector: {e}. Continuing without TSMixer analysis.")
                    self.use_tsmixer = False

            # Initialize Lorentz embedding detector if requested
            if self.use_lorentz:
                try:
                    from asf.medical.models.lorentz_embedding_detector import LorentzEmbeddingContradictionDetector

                    self.lorentz_detector = LorentzEmbeddingContradictionDetector(
                        biomedlm_scorer=self,
                        device=self.device,
                        config={"use_lorentz": True}
                    )

                    logger.info("Lorentz embedding contradiction detector initialized successfully")
                except ImportError as e:
                    logger.warning(f"Failed to import Lorentz detector: {e}. Continuing without Lorentz analysis.")
                    self.use_lorentz = False
                except Exception as e:
                    logger.warning(f"Failed to initialize Lorentz detector: {e}. Continuing without Lorentz analysis.")
                    self.use_lorentz = False

            # Initialize temporal confidence scorer if requested
            if self.use_temporal_confidence:
                try:
                    from asf.medical.models.temporal_confidence_scorer import TemporalConfidenceScorer
                    import re

                    self.temporal_confidence_scorer = TemporalConfidenceScorer()

                    logger.info("Temporal confidence scorer initialized successfully")
                except ImportError as e:
                    logger.warning(f"Failed to import temporal confidence scorer: {e}. Continuing without temporal confidence scoring.")
                    self.use_temporal_confidence = False
                except Exception as e:
                    logger.warning(f"Failed to initialize temporal confidence scorer: {e}. Continuing without temporal confidence scoring.")
                    self.use_temporal_confidence = False
        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            raise ImportError(
                "Please install the required libraries: pip install transformers torch"
            )
        except Exception as e:
            logger.error(f"Failed to initialize BioMedLM model: {e}")
            raise RuntimeError(f"Failed to initialize BioMedLM model: {e}")

    def get_score(self, claim1: str, claim2: str) -> float:
        """
        Get contradiction score between two claims.

        Args:
            claim1: First medical claim
            claim2: Second medical claim

        Returns:
            Contradiction score between 0 and 1, where higher values indicate
            stronger contradiction
        """
        try:
            import torch

            # Prepare input
            inputs = self.tokenizer(
                claim1, claim2, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get contradiction score (assuming binary classification)
            # For models with multiple classes, adjust accordingly
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            contradiction_score = probabilities[0, 1].item()  # Assuming class 1 is contradiction

            return contradiction_score
        except Exception as e:
            logger.error(f"Error getting contradiction score: {e}")
            # Return a default score in case of error
            return 0.5

    def get_detailed_scores(self, claim1: str, claim2: str) -> Dict[str, float]:
        """
        Get detailed contradiction scores between two claims.

        Args:
            claim1: First medical claim
            claim2: Second medical claim

        Returns:
            Dictionary with detailed scores
        """
        try:
            import torch

            # Prepare input
            inputs = self.tokenizer(
                claim1, claim2, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get contradiction scores
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

            # Assuming binary classification (contradiction vs. non-contradiction)
            contradiction_score = probabilities[0, 1].item()
            agreement_score = probabilities[0, 0].item()

            # Create base scores
            scores = {
                "contradiction_score": contradiction_score,
                "agreement_score": agreement_score,
                "confidence": max(contradiction_score, agreement_score)
            }

            return scores
        except Exception as e:
            logger.error(f"Error getting detailed contradiction scores: {e}")
            # Return default scores in case of error
            return {
                "contradiction_score": 0.5,
                "agreement_score": 0.5,
                "confidence": 0.0
            }

    def detect_contradiction_with_negation(self, claim1: str, claim2: str) -> Dict[str, Any]:
        """
        Detect contradiction between claims using negation awareness.

        This method combines BioMedLM scoring with negation detection for
        more accurate contradiction detection, especially in cases where
        negation is involved.

        Args:
            claim1: First medical claim
            claim2: Second medical claim

        Returns:
            Dictionary with contradiction detection results
        """
        # If negation detection is not enabled, fall back to basic scoring
        if not self.use_negation_detection or self.contradiction_detector is None:
            scores = self.get_detailed_scores(claim1, claim2)
            return {
                "text1": claim1,
                "text2": claim2,
                "has_contradiction": scores["contradiction_score"] > 0.7,
                "contradiction_score": scores["contradiction_score"],
                "agreement_score": scores["agreement_score"],
                "confidence": scores["confidence"],
                "method": "biomedlm_only",
                "negation_aware": False
            }

        # Use negation-aware contradiction detection
        try:
            result = self.contradiction_detector.detect_contradiction(claim1, claim2)
            return result
        except Exception as e:
            logger.error(f"Error in negation-aware contradiction detection: {e}")
            # Fall back to basic scoring
            scores = self.get_detailed_scores(claim1, claim2)
            return {
                "text1": claim1,
                "text2": claim2,
                "has_contradiction": scores["contradiction_score"] > 0.7,
                "contradiction_score": scores["contradiction_score"],
                "agreement_score": scores["agreement_score"],
                "confidence": scores["confidence"],
                "method": "biomedlm_only",
                "negation_aware": False,
                "error": str(e)
            }

    def detect_contradiction_multimodal(self, claim1: str, claim2: str) -> Dict[str, Any]:
        """
        Detect contradiction between claims using multimodal fusion.

        This method combines BioMedLM scoring with metadata extraction and
        multimodal fusion for more accurate contradiction detection, especially
        in cases where study design and other metadata are important.

        Args:
            claim1: First medical claim
            claim2: Second medical claim

        Returns:
            Dictionary with contradiction detection results
        """
        # If multimodal fusion is not enabled, fall back to negation-aware detection
        if not self.use_multimodal_fusion or self.multimodal_detector is None:
            return self.detect_contradiction_with_negation(claim1, claim2)

        # Use multimodal contradiction detection
        try:
            result = self.multimodal_detector.detect_contradiction(claim1, claim2)
            return result
        except Exception as e:
            logger.error(f"Error in multimodal contradiction detection: {e}")
            # Fall back to negation-aware detection
            return self.detect_contradiction_with_negation(claim1, claim2)

    def detect_contradiction(self, claim1: str, claim2: str) -> Dict[str, Any]:
        """
        Detect contradiction between claims using the best available method.

        This method selects the most advanced contradiction detection method
        available based on the configuration:
        1. TSMixer for temporal analysis (if enabled)
        2. Lorentz embeddings for hierarchical analysis (if enabled)
        3. Multimodal fusion (if enabled)
        4. Negation-aware detection (if enabled)
        5. Basic BioMedLM scoring (fallback)

        Args:
            claim1: First medical claim
            claim2: Second medical claim

        Returns:
            Dictionary with contradiction detection results
        """
        # Initialize result with basic information
        result = {
            "text1": claim1,
            "text2": claim2,
            "has_contradiction": False,
            "contradiction_score": 0.0,
            "methods_used": []
        }

        # Get base contradiction score from BioMedLM
        scores = self.get_detailed_scores(claim1, claim2)
        result["contradiction_score"] = scores["contradiction_score"]
        result["agreement_score"] = scores["agreement_score"]
        result["confidence"] = scores["confidence"]
        result["methods_used"].append("biomedlm")

        # Use TSMixer for temporal analysis if available
        if self.use_tsmixer and self.tsmixer_detector is not None:
            try:
                tsmixer_result = self.tsmixer_detector.detect_contradiction(claim1, claim2)
                result["tsmixer_result"] = tsmixer_result

                # Update contradiction score with TSMixer result
                if "contradiction_score" in tsmixer_result:
                    result["contradiction_score"] = max(
                        result["contradiction_score"],
                        tsmixer_result["contradiction_score"]
                    )

                result["methods_used"].append("tsmixer")
            except Exception as e:
                logger.error(f"Error in TSMixer contradiction detection: {e}")

        # Use Lorentz embeddings for hierarchical analysis if available
        if self.use_lorentz and self.lorentz_detector is not None:
            try:
                lorentz_result = self.lorentz_detector.detect_contradiction(claim1, claim2)
                result["lorentz_result"] = lorentz_result

                # Update contradiction score with Lorentz result
                if "contradiction_score" in lorentz_result:
                    result["contradiction_score"] = max(
                        result["contradiction_score"],
                        lorentz_result["contradiction_score"]
                    )

                result["methods_used"].append("lorentz")
            except Exception as e:
                logger.error(f"Error in Lorentz embedding contradiction detection: {e}")

        # Use multimodal fusion if available
        if self.use_multimodal_fusion and self.multimodal_detector is not None:
            try:
                multimodal_result = self.multimodal_detector.detect_contradiction(claim1, claim2)
                result["multimodal_result"] = multimodal_result

                # Update contradiction score with multimodal result
                if "contradiction_score" in multimodal_result:
                    result["contradiction_score"] = max(
                        result["contradiction_score"],
                        multimodal_result["contradiction_score"]
                    )

                result["methods_used"].append("multimodal")
            except Exception as e:
                logger.error(f"Error in multimodal contradiction detection: {e}")

        # Use negation-aware detection if available
        if self.use_negation_detection and self.contradiction_detector is not None:
            try:
                negation_result = self.contradiction_detector.detect_contradiction(claim1, claim2)
                result["negation_result"] = negation_result

                # Update contradiction score with negation result
                if "contradiction_score" in negation_result:
                    result["contradiction_score"] = max(
                        result["contradiction_score"],
                        negation_result["contradiction_score"]
                    )

                result["methods_used"].append("negation")
            except Exception as e:
                logger.error(f"Error in negation-aware contradiction detection: {e}")

        # Apply temporal confidence weighting if available
        if self.use_temporal_confidence and self.temporal_confidence_scorer is not None:
            try:
                # Extract metadata from claims
                claim1_metadata = self.temporal_confidence_scorer.get_claim_metadata(claim1)
                claim2_metadata = self.temporal_confidence_scorer.get_claim_metadata(claim2)

                # Calculate current confidence for each claim
                claim1_confidence = self.temporal_confidence_scorer.calculate_confidence(
                    initial_confidence=claim1_metadata["initial_confidence"],
                    domain=claim1_metadata["domain"],
                    creation_time=claim1_metadata["creation_time"]
                )

                claim2_confidence = self.temporal_confidence_scorer.calculate_confidence(
                    initial_confidence=claim2_metadata["initial_confidence"],
                    domain=claim2_metadata["domain"],
                    creation_time=claim2_metadata["creation_time"]
                )

                # Weight contradiction score based on temporal confidence
                weighted_score = self.temporal_confidence_scorer.weight_contradiction_score(
                    contradiction_score=result["contradiction_score"],
                    claim1_confidence=claim1_confidence,
                    claim2_confidence=claim2_confidence
                )

                # Add temporal confidence information to result
                result["temporal_confidence"] = {
                    "claim1_confidence": claim1_confidence,
                    "claim2_confidence": claim2_confidence,
                    "original_score": result["contradiction_score"],
                    "weighted_score": weighted_score
                }

                # Update contradiction score with weighted score
                result["contradiction_score"] = weighted_score
                result["methods_used"].append("temporal_confidence")
            except Exception as e:
                logger.error(f"Error applying temporal confidence weighting: {e}")

        # Determine if contradiction is detected
        result["has_contradiction"] = result["contradiction_score"] > 0.7

        return result

    def explain_contradiction(self, claim1: str, claim2: str, generate_visualization: bool = False, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Explain why two claims were determined to be contradictory.

        This method uses SHAP-based explainability to provide insights into
        why the model detected a contradiction between the claims.

        Args:
            claim1: First medical claim
            claim2: Second medical claim
            generate_visualization: Whether to generate a visualization
            output_path: Path to save the visualization

        Returns:
            Dictionary with explanation information
        """
        # First, detect contradiction
        contradiction_result = self.detect_contradiction(claim1, claim2)

        # If no contradiction detected, return early
        if not contradiction_result.get("has_contradiction", False):
            return {
                "claim1": claim1,
                "claim2": claim2,
                "contradiction_detected": False,
                "explanation": "No contradiction detected between the claims.",
                "methods_used": contradiction_result.get("methods_used", [])
            }

        # Check if SHAP-based explainer is available
        if self.use_shap_explainer and self.contradiction_explainer is not None:
            try:
                # Use SHAP explainer
                explanation = self.contradiction_explainer.explain_contradiction(
                    claim1=claim1,
                    claim2=claim2,
                    contradiction_score=contradiction_result.get("contradiction_score", 0.0),
                    use_shap=True,
                    use_negation_detection=True,
                    use_multimodal_factors="multimodal" in contradiction_result.get("methods_used", [])
                )

                # Generate visualization if requested
                if generate_visualization and output_path:
                    try:
                        from asf.medical.models.shap_explainer import ContradictionVisualizer
                        visualizer = ContradictionVisualizer()
                        visualizer.generate_html_report(explanation, output_path)
                    except Exception as e:
                        logger.error(f"Error generating visualization: {e}")

                # Return explanation as dictionary
                explanation_dict = explanation.to_dict()

                # Add methods used
                explanation_dict["methods_used"] = contradiction_result.get("methods_used", [])
                explanation_dict["methods_used"].append("shap")

                return explanation_dict
            except Exception as e:
                logger.error(f"Error generating SHAP explanation: {e}")

        # If SHAP explainer is not available or fails, check for other explainers
        if "tsmixer" in contradiction_result.get("methods_used", []) and "tsmixer_result" in contradiction_result:
            # Extract temporal explanation from TSMixer result
            try:
                tsmixer_result = contradiction_result["tsmixer_result"]
                if "temporal_analysis" in tsmixer_result and tsmixer_result["temporal_analysis"]:
                    return {
                        "claim1": claim1,
                        "claim2": claim2,
                        "contradiction_detected": True,
                        "contradiction_score": contradiction_result.get("contradiction_score", 0.0),
                        "explanation": "Temporal pattern contradiction detected.",
                        "temporal_analysis": tsmixer_result["temporal_analysis"],
                        "methods_used": contradiction_result.get("methods_used", [])
                    }
            except Exception as e:
                logger.error(f"Error extracting TSMixer explanation: {e}")

        # Fall back to basic explanation
        return {
            "claim1": claim1,
            "claim2": claim2,
            "contradiction_detected": True,
            "contradiction_score": contradiction_result.get("contradiction_score", 0.0),
            "explanation": "Contradiction detected between the claims.",
            "methods_used": contradiction_result.get("methods_used", [])
        }
