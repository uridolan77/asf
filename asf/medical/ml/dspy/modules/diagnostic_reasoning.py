"""Diagnostic Reasoning Module

This module provides a specialized DSPy module for medical diagnostic reasoning,
implementing a structured approach to differential diagnosis generation with
confidence scoring and evidence-based explanations.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import re

import dspy

from ..enhanced_module_base import EnhancedDSPyModuleBase
from ..enhanced_signatures import DiagnosticReasoning

# Set up logging
logger = logging.getLogger(__name__)


class DiagnosticReasoningModule(EnhancedDSPyModuleBase):
    """Enhanced diagnostic reasoning module for medical case analysis.
    
    This module implements a structured approach to medical diagnostic reasoning:
    1. Initial case analysis and key finding extraction
    2. Differential diagnosis generation with probabilities
    3. Recommended tests and next steps
    4. Evidence-based reasoning and confidence scoring
    
    The module follows medical best practices for diagnostic reasoning and
    includes comprehensive PHI detection, audit logging, and error handling.
    """
    
    def __init__(
        self,
        reasoning_model: Optional[dspy.Module] = None,
        max_diagnoses: int = 5,
        include_rare_conditions: bool = True,
        include_confidence_scores: bool = True,
        **kwargs
    ):
        """Initialize the diagnostic reasoning module.
        
        Args:
            reasoning_model: Custom reasoning model (optional)
            max_diagnoses: Maximum number of diagnoses to include in differential
            include_rare_conditions: Whether to include rare conditions in differential
            include_confidence_scores: Whether to include confidence scores
            **kwargs: Additional arguments for the parent class
        """
        super().__init__(**kwargs)
        
        # Create default reasoning model if not provided
        self.reasoning_model = reasoning_model or dspy.ChainOfThought(DiagnosticReasoning)
        
        # Store configuration
        self.max_diagnoses = max_diagnoses
        self.include_rare_conditions = include_rare_conditions
        self.include_confidence_scores = include_confidence_scores
        
        # Compile regex patterns for medical case preprocessing
        self._age_pattern = re.compile(r'\b(\d+)[- ](?:year|yr)[- ]old\b', re.IGNORECASE)
        self._gender_pattern = re.compile(r'\b(male|female|man|woman|boy|girl|non-binary|transgender)\b', re.IGNORECASE)
        self._symptom_patterns = [
            re.compile(r'\b(?:complains|presenting|reports|has|with)\s+(?:of|with)?\s+([^.]+)', re.IGNORECASE),
            re.compile(r'\b(?:symptoms|signs)\s+(?:include|of|are)\s+([^.]+)', re.IGNORECASE)
        ]
        
        # Initialize medical knowledge base references
        self._icd_codes = {}  # Would be populated with ICD-10 codes in a real implementation
        self._medical_guidelines = {}  # Would reference medical guidelines
    
    def _preprocess_case(self, case_description: str) -> Dict[str, Any]:
        """Extract structured information from the case description.
        
        Args:
            case_description: The medical case description
            
        Returns:
            Dict[str, Any]: Structured case information
        """
        # Extract age if present
        age_match = self._age_pattern.search(case_description)
        age = int(age_match.group(1)) if age_match else None
        
        # Extract gender if present
        gender_match = self._gender_pattern.search(case_description)
        gender = gender_match.group(1).lower() if gender_match else None
        
        # Normalize gender terms
        if gender in ['man', 'boy']:
            gender = 'male'
        elif gender in ['woman', 'girl']:
            gender = 'female'
        
        # Extract potential symptoms
        symptoms = []
        for pattern in self._symptom_patterns:
            matches = pattern.finditer(case_description)
            for match in matches:
                symptom_text = match.group(1).strip()
                if symptom_text and len(symptom_text) > 3:  # Avoid very short matches
                    symptoms.append(symptom_text)
        
        # Return structured information
        return {
            'age': age,
            'gender': gender,
            'symptoms': symptoms,
            'full_text': case_description
        }
    
    def _validate_reasoning_output(self, result: Any) -> Tuple[bool, Dict[str, Any]]:
        """Validate the reasoning model output and extract structured information.
        
        Args:
            result: The raw output from the reasoning model
            
        Returns:
            Tuple[bool, Dict[str, Any]]: Validation success flag and structured output
        """
        # Initialize structured output
        structured_output = {
            'differential_diagnosis': [],
            'recommended_tests': [],
            'reasoning': "",
            'confidence': 0.0
        }
        
        # Extract fields based on result type
        if hasattr(result, '__dict__'):
            # Handle object with attributes
            differential = getattr(result, 'differential_diagnosis', [])
            tests = getattr(result, 'recommended_tests', [])
            reasoning = getattr(result, 'reasoning', "")
            confidence = getattr(result, 'confidence', 0.0)
        elif isinstance(result, dict):
            # Handle dictionary
            differential = result.get('differential_diagnosis', [])
            tests = result.get('recommended_tests', [])
            reasoning = result.get('reasoning', "")
            confidence = result.get('confidence', 0.0)
        else:
            # Unrecognized format
            logger.warning(f"Unrecognized reasoning output format: {type(result)}")
            return False, structured_output
        
        # Validate differential diagnosis
        if not differential or not isinstance(differential, (list, str)):
            logger.warning("Missing or invalid differential diagnosis")
            return False, structured_output
        
        # Convert string differential to list if needed
        if isinstance(differential, str):
            # Split by common separators and clean up
            differential = [d.strip() for d in re.split(r'[,;]|\d+\.\s*', differential) if d.strip()]
        
        # Validate recommended tests
        if not tests or not isinstance(tests, (list, str)):
            logger.warning("Missing or invalid recommended tests")
            return False, structured_output
        
        # Convert string tests to list if needed
        if isinstance(tests, str):
            # Split by common separators and clean up
            tests = [t.strip() for t in re.split(r'[,;]|\d+\.\s*', tests) if t.strip()]
        
        # Validate reasoning
        if not reasoning or not isinstance(reasoning, str):
            logger.warning("Missing or invalid reasoning")
            return False, structured_output
        
        # Validate confidence
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            # Try to convert string to float if needed
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence)
                except ValueError:
                    confidence = 0.5  # Default to middle confidence
            else:
                confidence = 0.5  # Default to middle confidence
        
        # Update structured output
        structured_output['differential_diagnosis'] = differential[:self.max_diagnoses]
        structured_output['recommended_tests'] = tests
        structured_output['reasoning'] = reasoning
        structured_output['confidence'] = float(confidence)
        
        return True, structured_output
    
    def forward(self, case_description: str, **kwargs) -> Dict[str, Any]:
        """Execute the diagnostic reasoning process.
        
        Args:
            case_description: The medical case description
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: Diagnostic reasoning results
        """
        # Log audit event for the start of processing
        self.log_audit(
            "DIAGNOSTIC_REASONING_START",
            {"case_description": case_description[:100] + "..." if len(case_description) > 100 else case_description},
            {}
        )
        
        # Sanitize input
        sanitized_case = self.sanitize_input(case_description)
        
        # Preprocess the case to extract structured information
        structured_case = self._preprocess_case(sanitized_case)
        
        # Prepare additional context for the reasoning model
        context = {
            "include_rare_conditions": self.include_rare_conditions,
            "max_diagnoses": self.max_diagnoses,
            **kwargs
        }
        
        # Call the reasoning model
        try:
            result = self.reasoning_model(
                case_description=sanitized_case,
                age=structured_case['age'],
                gender=structured_case['gender'],
                **context
            )
            
            # Validate and structure the output
            valid, structured_output = self._validate_reasoning_output(result)
            
            if not valid:
                # Fallback to a simpler approach if validation fails
                logger.warning("Validation failed, using fallback approach")
                fallback_result = dspy.Predict(DiagnosticReasoning)(
                    case_description=sanitized_case
                )
                valid, structured_output = self._validate_reasoning_output(fallback_result)
                
                if not valid:
                    # If still invalid, return a minimal valid response
                    structured_output = {
                        'differential_diagnosis': ["Insufficient information for diagnosis"],
                        'recommended_tests': ["Complete medical history", "Physical examination"],
                        'reasoning': "The provided case description lacks sufficient detail for a comprehensive differential diagnosis.",
                        'confidence': 0.3
                    }
        except Exception as e:
            # Handle exceptions gracefully
            logger.error(f"Error in diagnostic reasoning: {str(e)}")
            structured_output = {
                'differential_diagnosis': ["Error in diagnostic process"],
                'recommended_tests': ["Complete medical history", "Physical examination"],
                'reasoning': f"An error occurred during the diagnostic reasoning process: {str(e)}",
                'confidence': 0.1
            }
        
        # Sanitize outputs
        sanitized_output = {
            'differential_diagnosis': structured_output['differential_diagnosis'],
            'recommended_tests': structured_output['recommended_tests'],
            'reasoning': self.sanitize_output(structured_output['reasoning']),
            'confidence': structured_output['confidence']
        }
        
        # Add the structured case information to the output
        output = {
            'case_summary': {
                'age': structured_case['age'],
                'gender': structured_case['gender'],
                'key_symptoms': structured_case['symptoms'][:5] if structured_case['symptoms'] else []
            },
            **sanitized_output
        }
        
        # Log audit event for the completed processing
        self.log_audit(
            "DIAGNOSTIC_REASONING_COMPLETE",
            {"case_description": case_description[:100] + "..." if len(case_description) > 100 else case_description},
            output
        )
        
        return output


class SpecialistConsultModule(EnhancedDSPyModuleBase):
    """Specialist consultation module for complex medical cases.
    
    This module extends the basic diagnostic reasoning with specialist-specific
    knowledge and reasoning patterns for complex or specialized medical cases.
    It can be configured for different medical specialties like cardiology,
    neurology, oncology, etc.
    """
    
    def __init__(
        self,
        specialty: str,
        base_reasoning_module: Optional[DiagnosticReasoningModule] = None,
        specialist_model: Optional[dspy.Module] = None,
        **kwargs
    ):
        """Initialize the specialist consultation module.
        
        Args:
            specialty: Medical specialty (e.g., "cardiology", "neurology")
            base_reasoning_module: Base diagnostic reasoning module
            specialist_model: Specialist-specific reasoning model
            **kwargs: Additional arguments for the parent class
        """
        super().__init__(**kwargs)
        
        # Store specialty
        self.specialty = specialty.lower()
        
        # Create base reasoning module if not provided
        self.base_reasoning = base_reasoning_module or DiagnosticReasoningModule()
        
        # Create specialist signature
        SpecialistSignature = dspy.Signature(
            case_description=dspy.InputField(desc="Medical case description"),
            initial_assessment=dspy.InputField(desc="Initial diagnostic assessment"),
            specialty=dspy.InputField(desc="Medical specialty for consultation"),
            specialist_assessment=dspy.OutputField(desc=f"Detailed {specialty} specialist assessment"),
            specialist_diagnosis=dspy.OutputField(desc=f"Differential diagnosis from {specialty} perspective"),
            specialist_recommendations=dspy.OutputField(desc=f"Recommended tests and treatments from {specialty} specialist"),
            confidence=dspy.OutputField(desc="Confidence score (0-1)")
        )
        
        # Create specialist model if not provided
        self.specialist_model = specialist_model or dspy.ChainOfThought(SpecialistSignature)
    
    def forward(self, case_description: str, **kwargs) -> Dict[str, Any]:
        """Execute the specialist consultation process.
        
        Args:
            case_description: The medical case description
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: Specialist consultation results
        """
        # First get the base diagnostic reasoning
        base_assessment = self.base_reasoning(case_description, **kwargs)
        
        # Log audit event
        self.log_audit(
            "SPECIALIST_CONSULTATION_START",
            {
                "case_description": case_description[:100] + "..." if len(case_description) > 100 else case_description,
                "specialty": self.specialty
            },
            {}
        )
        
        # Call the specialist model
        try:
            result = self.specialist_model(
                case_description=case_description,
                initial_assessment=base_assessment,
                specialty=self.specialty
            )
            
            # Extract fields
            if hasattr(result, '__dict__'):
                specialist_assessment = getattr(result, 'specialist_assessment', "")
                specialist_diagnosis = getattr(result, 'specialist_diagnosis', [])
                specialist_recommendations = getattr(result, 'specialist_recommendations', [])
                confidence = getattr(result, 'confidence', 0.5)
            elif isinstance(result, dict):
                specialist_assessment = result.get('specialist_assessment', "")
                specialist_diagnosis = result.get('specialist_diagnosis', [])
                specialist_recommendations = result.get('specialist_recommendations', [])
                confidence = result.get('confidence', 0.5)
            else:
                specialist_assessment = str(result)
                specialist_diagnosis = []
                specialist_recommendations = []
                confidence = 0.5
            
            # Convert string diagnoses to list if needed
            if isinstance(specialist_diagnosis, str):
                specialist_diagnosis = [d.strip() for d in re.split(r'[,;]|\d+\.\s*', specialist_diagnosis) if d.strip()]
            
            # Convert string recommendations to list if needed
            if isinstance(specialist_recommendations, str):
                specialist_recommendations = [r.strip() for r in re.split(r'[,;]|\d+\.\s*', specialist_recommendations) if r.strip()]
            
        except Exception as e:
            # Handle exceptions gracefully
            logger.error(f"Error in specialist consultation: {str(e)}")
            specialist_assessment = f"Error in specialist consultation: {str(e)}"
            specialist_diagnosis = []
            specialist_recommendations = []
            confidence = 0.1
        
        # Sanitize outputs
        sanitized_assessment = self.sanitize_output(specialist_assessment)
        
        # Prepare output
        output = {
            'base_assessment': base_assessment,
            'specialty': self.specialty,
            'specialist_assessment': sanitized_assessment,
            'specialist_diagnosis': specialist_diagnosis,
            'specialist_recommendations': specialist_recommendations,
            'confidence': float(confidence)
        }
        
        # Log audit event
        self.log_audit(
            "SPECIALIST_CONSULTATION_COMPLETE",
            {
                "case_description": case_description[:100] + "..." if len(case_description) > 100 else case_description,
                "specialty": self.specialty
            },
            output
        )
        
        return output
