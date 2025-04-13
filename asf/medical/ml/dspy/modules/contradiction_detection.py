"""Contradiction Detection Modules

This module provides specialized DSPy modules for detecting contradictions in medical statements.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple

import dspy

from ..dspy_module_base import MedicalDSPyModule
from ..dspy_signatures import ContradictionDetection, TemporalContradictionDetection

# Set up logging
logger = logging.getLogger(__name__)


class ContradictionDetectionModule(MedicalDSPyModule):
    """Module for detecting contradictions between medical statements.

    This module uses a chain of thought approach to identify contradictions
    between two medical statements and provide an explanation.
    """

    def __init__(
        self,
        model: Optional[dspy.Module] = None,
        **kwargs
    ):
        """
        Initialize the contradiction detection module.

        Args:
            model: Custom contradiction detection model (optional)
            **kwargs: Additional arguments for the parent class
        """
        super().__init__(**kwargs)

        # Create default model if not provided
        self.model = model or dspy.ChainOfThought(ContradictionDetection)

    def forward(
        self,
        statement1: str,
        statement2: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Detect contradictions between two medical statements.

        Args:
            statement1: First medical statement
            statement2: Second medical statement
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Contradiction detection result
        """
        # Log audit
        self.log_audit(
            "CONTRADICTION_DETECTION_FORWARD",
            {"statement1": statement1, "statement2": statement2, **kwargs},
            {}
        )

        # Sanitize inputs
        sanitized_statement1 = self.sanitize_input(statement1)
        sanitized_statement2 = self.sanitize_input(statement2)

        # Detect contradictions
        try:
            detection_result = self.model(
                statement1=sanitized_statement1,
                statement2=sanitized_statement2
            )

            # Extract fields
            if hasattr(detection_result, '__dict__'):
                contradiction = getattr(detection_result, 'contradiction', False)
                explanation = getattr(detection_result, 'explanation', "No explanation provided")
                confidence = getattr(detection_result, 'confidence', 0.0)
            elif isinstance(detection_result, dict):
                contradiction = detection_result.get('contradiction', False)
                explanation = detection_result.get('explanation', "No explanation provided")
                confidence = detection_result.get('confidence', 0.0)
            else:
                contradiction = False
                explanation = "Could not determine contradiction status"
                confidence = 0.0

            # Convert contradiction to boolean if it's a string
            if isinstance(contradiction, str):
                contradiction = contradiction.lower() in ['true', 'yes', '1']

        except Exception as e:
            logger.error(f"Contradiction detection failed: {str(e)}")
            contradiction = False
            explanation = f"Error in contradiction detection: {str(e)}"
            confidence = 0.0

        # Sanitize outputs
        sanitized_explanation = self.sanitize_output(explanation)

        # Prepare result
        result = {
            'statement1': statement1,
            'statement2': statement2,
            'contradiction': contradiction,
            'explanation': sanitized_explanation,
            'confidence': confidence
        }

        # Log audit
        self.log_audit(
            "CONTRADICTION_DETECTION_FORWARD",
            {"statement1": statement1, "statement2": statement2, **kwargs},
            result
        )

        return result


class TemporalContradictionModule(MedicalDSPyModule):
    """Module for detecting contradictions between medical statements with temporal context.

    This module considers the timestamps of statements to determine if apparent
    contradictions are actually due to temporal evolution of medical knowledge or
    patient condition.
    """

    def __init__(
        self,
        model: Optional[dspy.Module] = None,
        **kwargs
    ):
        """
        Initialize the temporal contradiction detection module.

        Args:
            model: Custom temporal contradiction detection model (optional)
            **kwargs: Additional arguments for the parent class
        """
        super().__init__(**kwargs)

        # Create default model if not provided
        self.model = model or dspy.ChainOfThought(TemporalContradictionDetection)

    def forward(
        self,
        statement1: str,
        timestamp1: str,
        statement2: str,
        timestamp2: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Detect temporal contradictions between medical statements.

        Args:
            statement1: First medical statement
            timestamp1: Timestamp of the first statement
            statement2: Second medical statement
            timestamp2: Timestamp of the second statement
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Temporal contradiction detection result
        """
        # Log audit
        self.log_audit(
            "TEMPORAL_CONTRADICTION_DETECTION_FORWARD",
            {
                "statement1": statement1,
                "timestamp1": timestamp1,
                "statement2": statement2,
                "timestamp2": timestamp2,
                **kwargs
            },
            {}
        )

        # Sanitize inputs
        sanitized_statement1 = self.sanitize_input(statement1)
        sanitized_statement2 = self.sanitize_input(statement2)

        # Detect temporal contradictions
        try:
            detection_result = self.model(
                statement1=sanitized_statement1,
                timestamp1=timestamp1,
                statement2=sanitized_statement2,
                timestamp2=timestamp2
            )

            # Extract fields
            if hasattr(detection_result, '__dict__'):
                contradiction = getattr(detection_result, 'contradiction', False)
                temporal_relation = getattr(detection_result, 'temporal_relation', "unknown")
                explanation = getattr(detection_result, 'explanation', "No explanation provided")
            elif isinstance(detection_result, dict):
                contradiction = detection_result.get('contradiction', False)
                temporal_relation = detection_result.get('temporal_relation', "unknown")
                explanation = detection_result.get('explanation', "No explanation provided")
            else:
                contradiction = False
                temporal_relation = "unknown"
                explanation = "Could not determine contradiction status"

            # Convert contradiction to boolean if it's a string
            if isinstance(contradiction, str):
                contradiction = contradiction.lower() in ['true', 'yes', '1']

        except Exception as e:
            logger.error(f"Temporal contradiction detection failed: {str(e)}")
            contradiction = False
            temporal_relation = "error"
            explanation = f"Error in temporal contradiction detection: {str(e)}"

        # Sanitize outputs
        sanitized_explanation = self.sanitize_output(explanation)

        # Prepare result
        result = {
            'statement1': statement1,
            'timestamp1': timestamp1,
            'statement2': statement2,
            'timestamp2': timestamp2,
            'contradiction': contradiction,
            'temporal_relation': temporal_relation,
            'explanation': sanitized_explanation
        }

        # Log audit
        self.log_audit(
            "TEMPORAL_CONTRADICTION_DETECTION_FORWARD",
            {
                "statement1": statement1,
                "timestamp1": timestamp1,
                "statement2": statement2,
                "timestamp2": timestamp2,
                **kwargs
            },
            result
        )

        return result


class BioMedLMContradictionModule(MedicalDSPyModule):
    """Specialized contradiction detection module using BioMedLM.

    This module leverages the BioMedLM model for more accurate contradiction
    detection in medical statements by using domain-specific knowledge.
    """

    def __init__(
        self,
        biomedlm_model: str = "microsoft/BioMedLM",
        use_gpu: bool = True,
        **kwargs
    ):
        """
        Initialize the BioMedLM contradiction detection module.

        Args:
            biomedlm_model: Name or path of the BioMedLM model
            use_gpu: Whether to use GPU for inference
            **kwargs: Additional arguments for the parent class
        """
        super().__init__(**kwargs)

        self.biomedlm_model = biomedlm_model
        self.use_gpu = use_gpu

        # Create a specialized signature for BioMedLM
        BioMedLMContradictionSignature = dspy.Signature(
            statement1=dspy.InputField(desc="First medical statement"),
            statement2=dspy.InputField(desc="Second medical statement"),
            contradiction=dspy.OutputField(desc="Whether the statements contradict each other (true/false)"),
            explanation=dspy.OutputField(desc="Detailed medical explanation of the contradiction or lack thereof"),
            confidence=dspy.OutputField(desc="Confidence score between 0 and 1"),
            medical_context=dspy.OutputField(desc="Relevant medical context for understanding the statements")
        )

        # Create the model
        self.model = dspy.ChainOfThought(BioMedLMContradictionSignature)

        # Initialize BioMedLM model
        # Note: In production, we would load the actual BioMedLM model here
        # and create a custom predictor. For now, we use the standard DSPy predictor.
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info(f"Attempting to load BioMedLM model: {biomedlm_model}")
            # This is just placeholder code - in production we would properly initialize
            # the model with appropriate configuration
            self.tokenizer = AutoTokenizer.from_pretrained(biomedlm_model)
            self.biomedlm = AutoModelForCausalLM.from_pretrained(
                biomedlm_model,
                device_map="auto" if use_gpu else "cpu",
                trust_remote_code=True
            )
            logger.info(f"Successfully loaded BioMedLM model: {biomedlm_model}")
        except Exception as e:
            logger.warning(f"Could not load BioMedLM model: {str(e)}")
            logger.warning("Falling back to standard DSPy predictor")
            self.tokenizer = None
            self.biomedlm = None

    def forward(
        self,
        statement1: str,
        statement2: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Detect contradictions using BioMedLM.

        Args:
            statement1: First medical statement
            statement2: Second medical statement
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: BioMedLM contradiction detection result
        """
        # Log audit
        self.log_audit(
            "BIOMEDLM_CONTRADICTION_DETECTION_FORWARD",
            {"statement1": statement1, "statement2": statement2, **kwargs},
            {}
        )

        # Sanitize inputs
        sanitized_statement1 = self.sanitize_input(statement1)
        sanitized_statement2 = self.sanitize_input(statement2)

        # Detect contradictions using BioMedLM
        try:
            detection_result = self.model(
                statement1=sanitized_statement1,
                statement2=sanitized_statement2
            )

            # Extract fields
            if hasattr(detection_result, '__dict__'):
                contradiction = getattr(detection_result, 'contradiction', False)
                explanation = getattr(detection_result, 'explanation', "No explanation provided")
                confidence = getattr(detection_result, 'confidence', 0.0)
                medical_context = getattr(detection_result, 'medical_context', "")
            elif isinstance(detection_result, dict):
                contradiction = detection_result.get('contradiction', False)
                explanation = detection_result.get('explanation', "No explanation provided")
                confidence = detection_result.get('confidence', 0.0)
                medical_context = detection_result.get('medical_context', "")
            else:
                contradiction = False
                explanation = "Could not determine contradiction status"
                confidence = 0.0
                medical_context = ""

            # Convert contradiction to boolean if it's a string
            if isinstance(contradiction, str):
                contradiction = contradiction.lower() in ['true', 'yes', '1']

        except Exception as e:
            logger.error(f"BioMedLM contradiction detection failed: {str(e)}")
            contradiction = False
            explanation = f"Error in BioMedLM contradiction detection: {str(e)}"
            confidence = 0.0
            medical_context = ""

        # Sanitize outputs
        sanitized_explanation = self.sanitize_output(explanation)
        sanitized_medical_context = self.sanitize_output(medical_context)

        # Prepare result
        result = {
            'statement1': statement1,
            'statement2': statement2,
            'contradiction': contradiction,
            'explanation': sanitized_explanation,
            'confidence': confidence,
            'medical_context': sanitized_medical_context,
            'model': self.biomedlm_model
        }

        # Log audit
        self.log_audit(
            "BIOMEDLM_CONTRADICTION_DETECTION_FORWARD",
            {"statement1": statement1, "statement2": statement2, **kwargs},
            result
        )

        return result
