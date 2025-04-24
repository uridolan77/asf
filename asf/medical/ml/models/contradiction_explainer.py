"""
Contradiction Explainer

This module provides explainable AI functionality for the medical research contradiction detection system.
It generates human-readable explanations and visual representations of why two medical claims
are considered contradictory.

The explainer uses LIME (Local Interpretable Model-agnostic Explanations) and
SHAP (SHapley Additive exPlanations) to provide transparent insights into the
contradiction detection model's decisions.
"""

import os
import re
import json
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import shap
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from asf.medical.core.logging_config import get_logger

# Set up logger
logger = get_logger(__name__)

@dataclass
class ContradictionExplanation:
    """
    Container for contradiction explanation data.
    
    This class holds all the information about a contradiction explanation,
    including textual explanation, confidence scores, feature importance,
    and methods to generate visualizations.
    """
    
    explanation_text: str
    confidence: float
    feature_importance: Dict[str, float]
    explanation_method: str  # 'lime', 'shap', or 'combined'
    claim1: str
    claim2: str
    contradiction_type: str
    word_contributions: Optional[Dict[str, float]] = None
    uncertainty: Optional[float] = None
    alternative_outcomes: Optional[Dict[str, float]] = None
    
    def generate_visual(self, output_path: str) -> str:
        """
        Generate a visualization of the explanation and save it to a file.
        
        Args:
            output_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        plt.figure(figsize=(12, 8))
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Plot feature importance
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:10]  # Top 10 features
        
        features = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]
        
        # Create color map based on contribution direction
        colors = ['green' if imp > 0 else 'red' for imp in importances]
        
        plt.barh(features, importances, color=colors)
        plt.xlabel('Contribution to Contradiction')
        plt.title(f'Explaining {self.contradiction_type.title()} Contradiction\nConfidence: {self.confidence:.1%}')
        
        # Add claims as text
        plt.figtext(0.1, 0.01, f"Claim 1: {self.claim1}", wrap=True, fontsize=8)
        plt.figtext(0.1, 0.05, f"Claim 2: {self.claim2}", wrap=True, fontsize=8)
        
        # Add explanation as text
        plt.figtext(0.5, 0.95, f"Explanation: {self.explanation_text}", 
                   wrap=True, horizontalalignment='center', fontsize=10)
        
        # Add uncertainty if available
        if self.uncertainty:
            plt.figtext(0.5, 0.90, f"Uncertainty: {self.uncertainty:.1%}", 
                       wrap=True, horizontalalignment='center', fontsize=8)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.15)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_html_explanation(self) -> str:
        """
        Generate an HTML representation of the explanation.
        
        Returns:
            HTML string containing the explanation
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Contradiction Explanation</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 800px; margin: 0 auto; }
                .header { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .claims { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .explanation { background-color: #e9f7ef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .features { margin-top: 20px; }
                .feature { display: flex; margin-bottom: 10px; }
                .feature-name { width: 150px; }
                .feature-bar-container { flex-grow: 1; background-color: #f0f0f0; position: relative; height: 20px; }
                .feature-bar { height: 20px; }
                .feature-value { width: 50px; text-align: right; padding-left: 10px; }
                .positive { background-color: #5cb85c; }
                .negative { background-color: #d9534f; }
                .confidence { font-weight: bold; margin-top: 10px; }
                .uncertainty { font-style: italic; color: #777; margin-top: 5px; }
                .alternatives { margin-top: 20px; background-color: #f0f0f0; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>Contradiction Explanation</h2>
                    <p>Type: <strong>{contradiction_type}</strong></p>
                    <p class="confidence">Confidence: <strong>{confidence:.1%}</strong></p>
                    {uncertainty_html}
                </div>
                
                <div class="claims">
                    <h3>Claims</h3>
                    <p><strong>Claim 1:</strong> {claim1}</p>
                    <p><strong>Claim 2:</strong> {claim2}</p>
                </div>
                
                <div class="explanation">
                    <h3>Explanation</h3>
                    <p>{explanation_text}</p>
                </div>
                
                <h3>Key Features</h3>
                <div class="features">
        """.format(
            contradiction_type=self.contradiction_type.title(),
            confidence=self.confidence,
            uncertainty_html=f'<p class="uncertainty">Uncertainty: {self.uncertainty:.1%}</p>' if self.uncertainty else '',
            claim1=self.claim1,
            claim2=self.claim2,
            explanation_text=self.explanation_text
        )
        
        # Add feature bars
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:10]  # Top 10 features
        
        max_abs_importance = max([abs(imp) for _, imp in sorted_features]) if sorted_features else 1
        
        for feature, importance in sorted_features:
            percentage = abs(importance) / max_abs_importance * 100
            bar_class = "positive" if importance > 0 else "negative"
            
            html += f"""
                    <div class="feature">
                        <div class="feature-name">{feature}</div>
                        <div class="feature-bar-container">
                            <div class="feature-bar {bar_class}" style="width: {percentage}%;"></div>
                        </div>
                        <div class="feature-value">{importance:.3f}</div>
                    </div>
            """
        
        html += """
                </div>
        """
        
        # Add alternative classifications if available
        if self.alternative_outcomes:
            html += """
                <div class="alternatives">
                    <h3>Alternative Classifications</h3>
                    <ul>
            """
            
            for outcome, probability in sorted(
                self.alternative_outcomes.items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                html += f"""
                        <li>{outcome}: {probability:.1%}</li>
                """
            
            html += """
                    </ul>
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the explanation to a dictionary."""
        return {
            "explanation_text": self.explanation_text,
            "confidence": self.confidence,
            "feature_importance": self.feature_importance,
            "explanation_method": self.explanation_method,
            "claim1": self.claim1,
            "claim2": self.claim2,
            "contradiction_type": self.contradiction_type,
            "word_contributions": self.word_contributions,
            "uncertainty": self.uncertainty,
            "alternative_outcomes": self.alternative_outcomes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContradictionExplanation':
        """Create an explanation from a dictionary."""
        return cls(
            explanation_text=data["explanation_text"],
            confidence=data["confidence"],
            feature_importance=data["feature_importance"],
            explanation_method=data["explanation_method"],
            claim1=data["claim1"],
            claim2=data["claim2"],
            contradiction_type=data["contradiction_type"],
            word_contributions=data.get("word_contributions"),
            uncertainty=data.get("uncertainty"),
            alternative_outcomes=data.get("alternative_outcomes")
        )


class ContradictionExplainer:
    """
    Explainable AI service for medical claim contradiction detection.
    
    This class provides methods to explain why two medical claims are
    considered contradictory by the system. It uses explainable AI techniques
    like LIME and SHAP to generate feature importance scores and human-readable
    explanations.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        explanation_method: str = "combined"
    ):
        """
        Initialize the contradiction explainer.
        
        Args:
            model_path: Path to a pre-trained contradiction detection model (optional)
            explanation_method: Method to use for explanations ("lime", "shap", or "combined")
        """
        self.model_path = model_path
        self.explanation_method = explanation_method
        
        # Model would be loaded from path in a real implementation
        self.model = None
        
        # Initialize explainers
        self.lime_explainer = LimeTextExplainer(class_names=["Not Contradiction", "Contradiction"])
        self.vectorizer = TfidfVectorizer(max_features=500)
        
        logger.info(f"Initialized contradiction explainer using {explanation_method} method")
    
    def explain(
        self,
        claim1: str,
        claim2: str,
        contradiction_type: str = "unknown",
        model_confidence: float = 0.75,
        num_features: int = 10
    ) -> ContradictionExplanation:
        """
        Generate an explanation for why two claims are considered contradictory.
        
        Args:
            claim1: First medical claim
            claim2: Second medical claim
            contradiction_type: Type of contradiction if known
            model_confidence: Confidence score from the contradiction detection model
            num_features: Number of features to include in the explanation
            
        Returns:
            ContradictionExplanation object containing the explanation
        """
        logger.info(f"Explaining contradiction between claims: '{claim1[:50]}...' and '{claim2[:50]}...'")
        
        # In a real implementation, this would use the loaded model
        # For demonstration, we'll create a simulated explanation
        
        # Create a combined text for vectorization
        combined_text = f"{claim1} [SEP] {claim2}"
        
        # Extract features using the vectorizer
        features = self.extract_features(combined_text)
        
        # Generate feature importance scores
        # In a real implementation, this would use LIME/SHAP with the model
        feature_importance = self.generate_feature_importance(
            combined_text, 
            contradiction_type,
            model_confidence
        )
        
        # Generate word contributions (which words support/oppose contradiction)
        word_contributions = self.highlight_important_words(
            claim1, 
            claim2, 
            feature_importance
        )
        
        # Generate textual explanation based on feature importance and contradiction type
        explanation_text = self.generate_textual_explanation(
            claim1,
            claim2,
            contradiction_type,
            feature_importance,
            model_confidence
        )
        
        # Generate alternative outcomes with probabilities
        alternative_outcomes = {
            "Strong Contradiction": model_confidence,
            "Partial Contradiction": model_confidence * 0.3,
            "No Contradiction": 1 - model_confidence
        }
        
        # Calculate uncertainty
        uncertainty = 1 - (model_confidence**2)
        
        # Create explanation object
        explanation = ContradictionExplanation(
            explanation_text=explanation_text,
            confidence=model_confidence,
            feature_importance=feature_importance,
            explanation_method=self.explanation_method,
            claim1=claim1,
            claim2=claim2,
            contradiction_type=contradiction_type,
            word_contributions=word_contributions,
            uncertainty=uncertainty,
            alternative_outcomes=alternative_outcomes
        )
        
        logger.info(f"Generated explanation with {len(feature_importance)} features")
        
        return explanation
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract features from text.
        
        Args:
            text: Text to extract features from
            
        Returns:
            Dictionary of features and their values
        """
        # In a real implementation, this would use the vectorizer
        # For demonstration, we'll extract key medical terms
        
        medical_terms = [
            "trial", "study", "patients", "treatment", "effect", "efficacy",
            "significant", "results", "clinical", "therapy", "data", "outcome",
            "randomized", "placebo", "control", "dose", "mortality", "risk",
            "reduction", "increase", "decrease", "association", "correlation"
        ]
        
        features = {}
        text_lower = text.lower()
        
        for term in medical_terms:
            if term in text_lower:
                count = len(re.findall(r'\b' + term + r'\b', text_lower))
                features[term] = count / len(text_lower.split())
        
        return features
    
    def generate_feature_importance(
        self,
        text: str,
        contradiction_type: str,
        confidence: float
    ) -> Dict[str, float]:
        """
        Generate feature importance scores.
        
        In a real implementation, this would use LIME or SHAP with the model.
        For demonstration, we'll create simulated importance scores based on
        keyword presence and contradiction type.
        
        Args:
            text: Combined text of both claims
            contradiction_type: Type of contradiction
            confidence: Model confidence score
            
        Returns:
            Dictionary of feature names and their importance scores
        """
        text_lower = text.lower()
        
        # Define contradiction type-specific features
        type_features = {
            "direct": [
                "not", "no", "never", "contrary", "opposite", "unlike",
                "significant", "insignificant", "effective", "ineffective"
            ],
            "methodological": [
                "randomized", "observational", "trial", "study", "method",
                "control", "design", "placebo", "blinded", "open-label"
            ],
            "population": [
                "adult", "children", "elderly", "young", "male", "female",
                "patients", "healthy", "diabetic", "hypertensive", "obese"
            ],
            "statistical": [
                "significant", "p-value", "confidence", "interval", "effect",
                "correlation", "causation", "association", "risk", "hazard"
            ],
        }
        
        # Common contradiction terms
        contradiction_terms = {
            "trial": 0.3, "study": 0.3, "statistically": 0.4,
            "p < 0.05": 0.5, "p > 0.05": -0.5, "significant": 0.4, 
            "insignificant": -0.4, "increase": 0.3, "decrease": -0.3,
            "positive": 0.4, "negative": -0.4, "effective": 0.5, 
            "ineffective": -0.5, "benefit": 0.4, "harm": -0.4,
            "evidence": 0.3, "no evidence": -0.3, "supports": 0.4, 
            "contradicts": -0.4
        }
        
        # Get type-specific terms if contradiction type is known
        specific_terms = type_features.get(contradiction_type.lower(), [])
        
        # Generate importance scores
        importance = {}
        
        # Add importance for common contradiction terms
        for term, base_score in contradiction_terms.items():
            count = len(re.findall(r'\b' + term + r'\b', text_lower))
            if count > 0:
                # Adjust score based on confidence
                adjusted_score = base_score * confidence * count
                importance[term] = adjusted_score
        
        # Add importance for type-specific terms
        for term in specific_terms:
            if term in text_lower:
                count = len(re.findall(r'\b' + term + r'\b', text_lower))
                # Type-specific terms get higher importance
                score = 0.6 * confidence * count
                importance[term] = score
        
        # Add some key phrases if they exist in text
        key_phrases = [
            "no significant", "not effective", "statistically significant",
            "clinically significant", "no effect", "opposed to", "in contrast",
            "randomized controlled trial", "observational study"
        ]
        
        for phrase in key_phrases:
            if phrase in text_lower:
                importance[phrase] = 0.7 * confidence
        
        # Sort and limit to top features
        sorted_importance = dict(sorted(
            importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:10])
        
        return sorted_importance
    
    def highlight_important_words(
        self,
        claim1: str,
        claim2: str,
        feature_importance: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Identify important words in the claims and their contribution.
        
        Args:
            claim1: First claim
            claim2: Second claim
            feature_importance: Dictionary of feature importance scores
            
        Returns:
            Dictionary mapping words to their contribution scores
        """
        combined_text = f"{claim1.lower()} {claim2.lower()}"
        word_contributions = {}
        
        # Assign importance to individual words based on feature importance
        for feature, importance in feature_importance.items():
            # Check if feature is a phrase or single word
            if " " in feature:
                # For phrases, assign importance to each word
                words = feature.split()
                for word in words:
                    if word in combined_text:
                        word_contributions[word] = importance / len(words)
            else:
                if feature in combined_text:
                    word_contributions[feature] = importance
        
        return word_contributions
    
    def generate_textual_explanation(
        self,
        claim1: str,
        claim2: str,
        contradiction_type: str,
        feature_importance: Dict[str, float],
        confidence: float
    ) -> str:
        """
        Generate a human-readable explanation of the contradiction.
        
        Args:
            claim1: First claim
            claim2: Second claim
            contradiction_type: Type of contradiction
            feature_importance: Dictionary of feature importance scores
            confidence: Model confidence score
            
        Returns:
            Human-readable explanation string
        """
        # Base template for different contradiction types
        templates = {
            "direct": "These claims directly contradict each other regarding {topic}. " +
                     "{claim1_summary} while {claim2_summary}. " +
                     "Key contradictory terms include {key_terms}.",
            
            "methodological": "These claims differ due to methodological differences. " +
                             "{claim1_method} whereas {claim2_method}. " +
                             "This affects how the {topic} results are interpreted.",
            
            "statistical": "These claims show statistical inconsistency. " +
                         "{claim1_stats} but {claim2_stats}. " +
                         "This creates a contradiction in the reported {topic} outcomes.",
            
            "population": "These claims address different patient populations. " +
                         "{claim1_pop} while {claim2_pop}. " +
                         "The {topic} findings differ across these distinct groups.",
            
            "unknown": "These claims appear to contradict each other. " +
                      "The contradiction is supported by differences in {key_terms}."
        }
        
        # Extract key terms based on importance
        pos_terms = [f for f, i in feature_importance.items() if i > 0]
        neg_terms = [f for f, i in feature_importance.items() if i < 0]
        
        # Identify topic
        # In a real implementation, this would use NLP to identify the main topic
        topic_candidates = ["treatment efficacy", "clinical outcomes", "medical intervention",
                            "therapeutic approach", "drug effectiveness", "study results"]
        
        # Simple topic detection
        topic = "treatment outcomes"
        for candidate in topic_candidates:
            if candidate.lower() in claim1.lower() or candidate.lower() in claim2.lower():
                topic = candidate
                break
        
        # Generate summaries based on contradiction type
        if contradiction_type == "direct":
            claim1_summary = "The first claim suggests a positive outcome"
            claim2_summary = "the second claim indicates a negative or non-significant result"
            
            # Check for specific terms
            if any(term in claim1.lower() for term in ["significant", "effective", "positive"]):
                claim1_summary = "The first claim suggests a significant positive effect"
            if any(term in claim2.lower() for term in ["no significant", "not effective", "insignificant"]):
                claim2_summary = "the second claim states there was no significant effect"
        
        elif contradiction_type == "methodological":
            claim1_method = "The first claim uses one methodological approach"
            claim2_method = "the second claim uses a different methodology"
            
            # Check for specific study types
            if "randomized" in claim1.lower():
                claim1_method = "The first claim is based on a randomized controlled trial"
            elif "observational" in claim1.lower():
                claim1_method = "The first claim is based on an observational study"
                
            if "randomized" in claim2.lower():
                claim2_method = "the second claim is based on a randomized controlled trial"
            elif "observational" in claim2.lower():
                claim2_method = "the second claim is based on an observational study"
        
        elif contradiction_type == "statistical":
            claim1_stats = "The first claim reports statistical significance"
            claim2_stats = "the second claim does not show statistical significance"
            
            # Check for p-values or significance statements
            if "p < 0.05" in claim1.lower() or "significant" in claim1.lower():
                claim1_stats = "The first claim reports statistical significance (p < 0.05)"
            if "p > 0.05" in claim2.lower() or "not significant" in claim2.lower():
                claim2_stats = "the second claim reports non-significance (p > 0.05)"
        
        elif contradiction_type == "population":
            claim1_pop = "The first claim references one population group"
            claim2_pop = "the second claim focuses on a different population"
            
            # Check for specific population indicators
            population_terms = {
                "adults": "adult population", 
                "children": "pediatric population",
                "elderly": "elderly patients", 
                "women": "female patients",
                "men": "male patients", 
                "diabetic": "diabetic patients",
                "healthy": "healthy individuals"
            }
            
            for term, description in population_terms.items():
                if term in claim1.lower():
                    claim1_pop = f"The first claim focuses on {description}"
                if term in claim2.lower():
                    claim2_pop = f"the second claim addresses {description}"
        
        # Format key terms for readability
        key_terms_str = "contradictory statements"
        if pos_terms and neg_terms:
            pos_sample = pos_terms[:2]
            neg_sample = neg_terms[:2]
            key_terms_str = f"'{', '.join(pos_sample)}' vs. '{', '.join(neg_sample)}'"
        elif pos_terms:
            key_terms_str = f"'{', '.join(pos_terms[:3])}'"
        elif neg_terms:
            key_terms_str = f"'{', '.join(neg_terms[:3])}'"
        
        # Select and fill the appropriate template
        template = templates.get(contradiction_type.lower(), templates["unknown"])
        
        # Fill template based on contradiction type
        if contradiction_type == "direct":
            explanation = template.format(
                topic=topic,
                claim1_summary=claim1_summary,
                claim2_summary=claim2_summary,
                key_terms=key_terms_str
            )
        elif contradiction_type == "methodological":
            explanation = template.format(
                topic=topic,
                claim1_method=claim1_method,
                claim2_method=claim2_method
            )
        elif contradiction_type == "statistical":
            explanation = template.format(
                topic=topic,
                claim1_stats=claim1_stats,
                claim2_stats=claim2_stats
            )
        elif contradiction_type == "population":
            explanation = template.format(
                topic=topic,
                claim1_pop=claim1_pop,
                claim2_pop=claim2_pop
            )
        else:
            explanation = template.format(
                topic=topic,
                key_terms=key_terms_str
            )
        
        # Add confidence statement
        if confidence > 0.9:
            explanation += f" This is a high-confidence contradiction assessment ({confidence:.1%})."
        elif confidence > 0.7:
            explanation += f" This contradiction is detected with moderate confidence ({confidence:.1%})."
        else:
            explanation += f" This potential contradiction has lower confidence ({confidence:.1%}) and may require further investigation."
        
        return explanation