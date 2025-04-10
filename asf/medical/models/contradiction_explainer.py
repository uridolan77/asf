"""
Contradiction Explainer

This module provides SHAP-based explainability for contradiction analysis results,
helping users understand why two medical claims were determined to be contradictory.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("contradiction-explainer")

class ContradictionExplainer:
    """
    SHAP-based explainer for contradiction detection results.
    
    This class provides methods for explaining why two medical claims were
    determined to be contradictory, using SHAP (SHapley Additive exPlanations)
    values to identify the most influential words and phrases.
    """
    
    def __init__(self, biomedlm_scorer=None, use_shap: bool = True):
        """
        Initialize the contradiction explainer.
        
        Args:
            biomedlm_scorer: BioMedLMScorer instance
            use_shap: Whether to use SHAP for explanations
        """
        self.biomedlm_scorer = biomedlm_scorer
        self.use_shap = use_shap
        self.shap_explainer = None
        
        # Initialize SHAP if requested
        if self.use_shap:
            try:
                import shap
                logger.info("SHAP imported successfully")
                
                # Initialize SHAP explainer if BioMedLM scorer is provided
                if self.biomedlm_scorer is not None:
                    self._initialize_shap_explainer()
            except ImportError:
                logger.warning("SHAP not available. Install with: pip install shap")
                self.use_shap = False
    
    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer for the BioMedLM model."""
        try:
            import shap
            
            # Check if BioMedLM scorer is available
            if self.biomedlm_scorer is None or not hasattr(self.biomedlm_scorer, 'model'):
                logger.warning("BioMedLM scorer not available. SHAP explainer not initialized.")
                return
            
            # Create a wrapper function for the model
            def model_predict(texts):
                # Process batch of text pairs
                results = []
                for text_pair in texts:
                    # Split into claim1 and claim2
                    claim1, claim2 = text_pair
                    
                    # Get model prediction
                    inputs = self.biomedlm_scorer.tokenizer(
                        claim1, claim2, return_tensors="pt", padding=True, truncation=True, max_length=512
                    )
                    inputs = {k: v.to(self.biomedlm_scorer.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.biomedlm_scorer.model(**inputs)
                    
                    # Get contradiction scores
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=1)
                    
                    # Get contradiction score (class 1)
                    contradiction_score = probabilities[0, 1].item()
                    results.append(contradiction_score)
                
                return np.array(results)
            
            # Initialize SHAP explainer
            logger.info("Initializing SHAP explainer...")
            self.shap_explainer = shap.Explainer(model_predict, shap.maskers.Text)
            logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
            self.shap_explainer = None
    
    def explain_contradiction(self, claim1: str, claim2: str) -> Dict[str, Any]:
        """
        Explain why two claims were determined to be contradictory.
        
        Args:
            claim1: First medical claim
            claim2: Second medical claim
            
        Returns:
            Dictionary with explanation information
        """
        # Check if BioMedLM scorer is available
        if self.biomedlm_scorer is None:
            return {
                "error": "BioMedLM scorer not available",
                "claim1": claim1,
                "claim2": claim2,
                "explanation": "Cannot explain contradiction without BioMedLM scorer."
            }
        
        # Get contradiction result
        contradiction_result = self.biomedlm_scorer.detect_contradiction(claim1, claim2)
        
        # Check if contradiction was detected
        if not contradiction_result.get("has_contradiction", False):
            return {
                "claim1": claim1,
                "claim2": claim2,
                "contradiction_detected": False,
                "contradiction_score": contradiction_result.get("contradiction_score", 0.0),
                "explanation": "No contradiction detected between the claims."
            }
        
        # Get explanation based on contradiction type
        contradiction_type = contradiction_result.get("contradiction_type", "unknown")
        
        if contradiction_type == "negation" and "negation_analysis" in contradiction_result:
            # Explain negation-based contradiction
            explanation = self._explain_negation_contradiction(contradiction_result)
        elif contradiction_type == "multimodal" and "metadata1" in contradiction_result and "metadata2" in contradiction_result:
            # Explain multimodal-based contradiction
            explanation = self._explain_multimodal_contradiction(contradiction_result)
        elif self.use_shap and self.shap_explainer is not None:
            # Use SHAP for explanation
            explanation = self._explain_with_shap(claim1, claim2)
        else:
            # Fallback to keyword-based explanation
            explanation = self._explain_with_keywords(claim1, claim2)
        
        # Create result
        result = {
            "claim1": claim1,
            "claim2": claim2,
            "contradiction_detected": True,
            "contradiction_score": contradiction_result.get("contradiction_score", 0.0),
            "contradiction_type": contradiction_type,
            "explanation": explanation
        }
        
        # Add SHAP values if available
        if "shap_values" in explanation:
            result["shap_values"] = explanation["shap_values"]
        
        return result
    
    def _explain_negation_contradiction(self, contradiction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain negation-based contradiction.
        
        Args:
            contradiction_result: Contradiction detection result
            
        Returns:
            Dictionary with explanation information
        """
        negation_analysis = contradiction_result.get("negation_analysis", {})
        contradictions = negation_analysis.get("contradictions", [])
        
        if not contradictions:
            return {
                "type": "negation",
                "summary": "Contradiction detected based on negation, but specific contradictory elements could not be identified."
            }
        
        # Create explanation
        explanation_text = "Contradiction detected based on negation. The following elements are contradictory:\n"
        
        for i, contradiction in enumerate(contradictions):
            if "entity" in contradiction:
                entity = contradiction["entity"]
                negated_in_text1 = contradiction.get("negated_in_text1", False)
                negated_in_text2 = contradiction.get("negated_in_text2", False)
                
                explanation_text += f"{i+1}. The entity '{entity}' is "
                if negated_in_text1:
                    explanation_text += "negated in the first claim but affirmed in the second claim."
                else:
                    explanation_text += "affirmed in the first claim but negated in the second claim."
            elif "word" in contradiction:
                word = contradiction["word"]
                negated_in_text1 = contradiction.get("negated_in_text1", False)
                negated_in_text2 = contradiction.get("negated_in_text2", False)
                
                explanation_text += f"{i+1}. The word '{word}' is "
                if negated_in_text1:
                    explanation_text += "negated in the first claim but affirmed in the second claim."
                else:
                    explanation_text += "affirmed in the first claim but negated in the second claim."
        
        return {
            "type": "negation",
            "summary": explanation_text,
            "contradictions": contradictions
        }
    
    def _explain_multimodal_contradiction(self, contradiction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain multimodal-based contradiction.
        
        Args:
            contradiction_result: Contradiction detection result
            
        Returns:
            Dictionary with explanation information
        """
        metadata1 = contradiction_result.get("metadata1", {})
        metadata2 = contradiction_result.get("metadata2", {})
        
        # Extract study design information
        study_design1 = metadata1.get("study_design", {}).get("study_design", "unknown")
        study_design_score1 = metadata1.get("study_design", {}).get("design_score", 0.0)
        
        study_design2 = metadata2.get("study_design", {}).get("study_design", "unknown")
        study_design_score2 = metadata2.get("study_design", {}).get("design_score", 0.0)
        
        # Extract sample size information
        sample_size1 = metadata1.get("sample_size", {}).get("sample_size", 0)
        sample_size2 = metadata2.get("sample_size", {}).get("sample_size", 0)
        
        # Create explanation
        explanation_text = "Contradiction detected based on study design and sample size differences:\n"
        
        # Compare study designs
        if study_design_score1 > study_design_score2:
            explanation_text += f"1. The first claim is based on a {study_design1} (score: {study_design_score1:.1f}), "
            explanation_text += f"which has higher methodological quality than the {study_design2} (score: {study_design_score2:.1f}) "
            explanation_text += "in the second claim.\n"
        elif study_design_score2 > study_design_score1:
            explanation_text += f"1. The second claim is based on a {study_design2} (score: {study_design_score2:.1f}), "
            explanation_text += f"which has higher methodological quality than the {study_design1} (score: {study_design_score1:.1f}) "
            explanation_text += "in the first claim.\n"
        
        # Compare sample sizes
        if sample_size1 > sample_size2 * 2:  # First sample size is at least twice as large
            explanation_text += f"2. The first claim is based on a larger sample size (n={sample_size1}) "
            explanation_text += f"compared to the second claim (n={sample_size2}).\n"
        elif sample_size2 > sample_size1 * 2:  # Second sample size is at least twice as large
            explanation_text += f"2. The second claim is based on a larger sample size (n={sample_size2}) "
            explanation_text += f"compared to the first claim (n={sample_size1}).\n"
        
        # Add conclusion
        explanation_text += "\nThese differences in study design and sample size suggest that "
        if (study_design_score1 > study_design_score2 and sample_size1 >= sample_size2) or \
           (study_design_score1 >= study_design_score2 and sample_size1 > sample_size2):
            explanation_text += "the first claim may be more reliable."
        elif (study_design_score2 > study_design_score1 and sample_size2 >= sample_size1) or \
             (study_design_score2 >= study_design_score1 and sample_size2 > sample_size1):
            explanation_text += "the second claim may be more reliable."
        else:
            explanation_text += "one claim may be more reliable than the other, leading to a potential contradiction."
        
        return {
            "type": "multimodal",
            "summary": explanation_text,
            "study_design_comparison": {
                "claim1": {
                    "study_design": study_design1,
                    "design_score": study_design_score1,
                    "sample_size": sample_size1
                },
                "claim2": {
                    "study_design": study_design2,
                    "design_score": study_design_score2,
                    "sample_size": sample_size2
                }
            }
        }
    
    def _explain_with_shap(self, claim1: str, claim2: str) -> Dict[str, Any]:
        """
        Explain contradiction using SHAP values.
        
        Args:
            claim1: First medical claim
            claim2: Second medical claim
            
        Returns:
            Dictionary with explanation information
        """
        try:
            # Generate SHAP values
            shap_values = self.shap_explainer([[claim1, claim2]])
            
            # Get the most influential words
            top_words = []
            
            # Process SHAP values for the text pair
            values = shap_values.values[0]
            data = shap_values.data[0]
            
            # Combine the two texts
            combined_text = data[0] + " " + data[1]
            
            # Split into words
            words = re.findall(r'\b\w+\b', combined_text.lower())
            
            # Calculate word importance
            word_importance = defaultdict(float)
            
            # Map SHAP values to words
            for i, word in enumerate(words):
                if i < len(values):
                    word_importance[word] += abs(values[i])
            
            # Sort words by importance
            sorted_words = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Get top 10 words
            top_words = sorted_words[:10]
            
            # Create explanation
            explanation_text = "Contradiction detected based on the following key terms:\n"
            
            for i, (word, importance) in enumerate(top_words):
                explanation_text += f"{i+1}. '{word}' (importance: {importance:.4f})\n"
            
            # Add interpretation
            explanation_text += "\nThese terms contribute most significantly to the contradiction detection. "
            explanation_text += "They represent concepts that are presented differently or in conflict between the two claims."
            
            return {
                "type": "shap",
                "summary": explanation_text,
                "top_words": top_words,
                "shap_values": {
                    "values": values.tolist(),
                    "data": data
                }
            }
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            # Fall back to keyword-based explanation
            return self._explain_with_keywords(claim1, claim2)
    
    def _explain_with_keywords(self, claim1: str, claim2: str) -> Dict[str, Any]:
        """
        Explain contradiction using keyword analysis.
        
        Args:
            claim1: First medical claim
            claim2: Second medical claim
            
        Returns:
            Dictionary with explanation information
        """
        # Define contradiction indicators
        contradiction_indicators = [
            "not", "no", "never", "none", "neither", "nor", "without", "absence", "absent",
            "lack", "lacking", "fails", "failed", "negative", "negatively", "opposite",
            "contrary", "contrast", "different", "differently", "disagree", "disagrees",
            "inconsistent", "inconsistency", "conflict", "conflicts", "contradict", "contradicts",
            "dispute", "disputes", "challenge", "challenges", "refute", "refutes"
        ]
        
        # Define opposing term pairs
        opposing_terms = [
            ("increase", "decrease"),
            ("increased", "decreased"),
            ("increases", "decreases"),
            ("increasing", "decreasing"),
            ("higher", "lower"),
            ("high", "low"),
            ("more", "less"),
            ("positive", "negative"),
            ("significant", "insignificant"),
            ("effective", "ineffective"),
            ("beneficial", "harmful"),
            ("benefit", "harm"),
            ("improve", "worsen"),
            ("improves", "worsens"),
            ("improved", "worsened"),
            ("improvement", "deterioration"),
            ("better", "worse"),
            ("good", "bad"),
            ("safe", "unsafe"),
            ("recommended", "not recommended"),
            ("should", "should not"),
            ("can", "cannot"),
            ("present", "absent"),
            ("presence", "absence"),
            ("with", "without")
        ]
        
        # Tokenize claims
        claim1_words = set(re.findall(r'\b\w+\b', claim1.lower()))
        claim2_words = set(re.findall(r'\b\w+\b', claim2.lower()))
        
        # Find contradiction indicators
        indicators_in_claim1 = [word for word in contradiction_indicators if word in claim1_words]
        indicators_in_claim2 = [word for word in contradiction_indicators if word in claim2_words]
        
        # Find opposing terms
        opposing_pairs = []
        for term1, term2 in opposing_terms:
            if term1 in claim1_words and term2 in claim2_words:
                opposing_pairs.append((term1, term2))
            elif term2 in claim1_words and term1 in claim2_words:
                opposing_pairs.append((term2, term1))
        
        # Create explanation
        explanation_text = "Contradiction detected based on keyword analysis:\n"
        
        # Add contradiction indicators
        if indicators_in_claim1 or indicators_in_claim2:
            explanation_text += "1. Contradiction indicators found:\n"
            if indicators_in_claim1:
                explanation_text += f"   - In first claim: {', '.join(indicators_in_claim1)}\n"
            if indicators_in_claim2:
                explanation_text += f"   - In second claim: {', '.join(indicators_in_claim2)}\n"
        
        # Add opposing terms
        if opposing_pairs:
            explanation_text += "2. Opposing terms found:\n"
            for i, (term1, term2) in enumerate(opposing_pairs):
                explanation_text += f"   - '{term1}' in one claim vs. '{term2}' in the other\n"
        
        # Add common words
        common_words = claim1_words.intersection(claim2_words)
        if common_words:
            explanation_text += "3. The claims discuss similar topics, as evidenced by common terms:\n"
            explanation_text += f"   - Common terms: {', '.join(list(common_words)[:10])}\n"
        
        # Add interpretation
        explanation_text += "\nThese linguistic patterns suggest that the claims present contradictory information "
        explanation_text += "about the same or related topics."
        
        return {
            "type": "keyword",
            "summary": explanation_text,
            "contradiction_indicators": {
                "claim1": indicators_in_claim1,
                "claim2": indicators_in_claim2
            },
            "opposing_terms": opposing_pairs,
            "common_terms": list(common_words)
        }
    
    def generate_visualization(self, explanation: Dict[str, Any], output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate visualization for the explanation.
        
        Args:
            explanation: Explanation dictionary
            output_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization or None if visualization failed
        """
        try:
            import matplotlib.pyplot as plt
            import shap
            
            # Check if SHAP values are available
            if explanation.get("type") != "shap" or "shap_values" not in explanation:
                logger.warning("SHAP values not available for visualization")
                return None
            
            # Extract SHAP values
            shap_data = explanation["shap_values"]
            values = np.array(shap_data["values"])
            data = shap_data["data"]
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            
            # Create SHAP force plot
            shap_values = shap.Explanation(values=values, data=data)
            shap.plots.text(shap_values)
            
            # Save visualization if output path is provided
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
                logger.info(f"Visualization saved to {output_path}")
                return output_path
            else:
                # Display visualization
                plt.show()
                return None
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return None
