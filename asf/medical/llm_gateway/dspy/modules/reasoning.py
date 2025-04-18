"""
Reasoning Module

This module provides DSPy modules for general-purpose reasoning tasks.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import dspy

from .base import DSPyModuleBase
from .enhanced_base import EnhancedDSPyModuleBase

# Set up logging
logger = logging.getLogger(__name__)


class ReasoningSignature(dspy.Signature):
    """Signature for general reasoning."""
    
    problem_description = dspy.InputField(desc="Description of the problem or situation")
    reasoning_steps = dspy.OutputField(desc="Step-by-step reasoning process")
    solution = dspy.OutputField(desc="Proposed solution to the problem")
    alternatives = dspy.OutputField(desc="Alternative solutions to consider")
    confidence = dspy.OutputField(desc="Confidence in the solution (high, medium, low)")


class ExpertReasoningSignature(dspy.Signature):
    """Signature for expert reasoning."""
    
    problem_description = dspy.InputField(desc="Description of the problem or situation")
    domain = dspy.InputField(desc="Domain of expertise (e.g., 'software', 'finance')")
    expert_reasoning = dspy.OutputField(desc="Expert reasoning process specific to the domain")
    expert_solution = dspy.OutputField(desc="Expert solution to the problem")
    expert_alternatives = dspy.OutputField(desc="Expert alternative solutions to consider")
    confidence = dspy.OutputField(desc="Confidence in the solution (high, medium, low)")
    references = dspy.OutputField(desc="References to domain knowledge or best practices")


class ReasoningModule(DSPyModuleBase):
    """Module for general-purpose reasoning."""
    
    def __init__(
        self,
        max_solutions: int = 1,
        include_alternatives: bool = True,
    ):
        """
        Initialize the reasoning module.
        
        Args:
            max_solutions: Maximum number of solutions to include
            include_alternatives: Whether to include alternative solutions
        """
        super().__init__()
        self.max_solutions = max_solutions
        self.include_alternatives = include_alternatives
        
        # Create predictor
        self.predictor = dspy.Predict(ReasoningSignature)
    
    async def predict(
        self,
        problem_description: str
    ) -> Dict[str, Any]:
        """
        Perform reasoning on a problem.
        
        Args:
            problem_description: Description of the problem
            
        Returns:
            Dict[str, Any]: Reasoning result
        """
        # Log the request
        logger.info(f"Performing reasoning on problem: {problem_description[:100]}...")
        
        try:
            # Call predictor
            response = await self.call_predictor(
                self.predictor,
                problem_description=problem_description
            )
            
            # Process response
            result = {
                "reasoning_steps": response.reasoning_steps,
                "solution": response.solution,
                "confidence": response.confidence,
            }
            
            # Add alternatives if requested
            if self.include_alternatives:
                result["alternatives"] = response.alternatives
            
            return result
            
        except Exception as e:
            logger.error(f"Error in reasoning: {str(e)}")
            raise


class ExpertReasoningModule(EnhancedDSPyModuleBase):
    """Module for expert-level reasoning in specific domains."""
    
    def __init__(
        self,
        domain: str,
        base_reasoning_module: Optional[ReasoningModule] = None,
        include_references: bool = True,
    ):
        """
        Initialize the expert reasoning module.
        
        Args:
            domain: Domain of expertise (e.g., 'software', 'finance')
            base_reasoning_module: Base reasoning module to extend
            include_references: Whether to include references to domain knowledge
        """
        super().__init__()
        self.domain = domain
        self.base_reasoning_module = base_reasoning_module or ReasoningModule()
        self.include_references = include_references
        
        # Create predictor
        self.predictor = dspy.Predict(ExpertReasoningSignature)
    
    async def predict(
        self,
        problem_description: str
    ) -> Dict[str, Any]:
        """
        Perform expert reasoning on a problem.
        
        Args:
            problem_description: Description of the problem
            
        Returns:
            Dict[str, Any]: Expert reasoning result
        """
        # Log the request
        logger.info(f"Performing expert reasoning in {self.domain} domain on problem: {problem_description[:100]}...")
        
        try:
            # First get base reasoning if available
            base_result = None
            if self.base_reasoning_module:
                try:
                    base_result = await self.base_reasoning_module.predict(problem_description)
                except Exception as e:
                    logger.warning(f"Base reasoning failed: {str(e)}")
            
            # Call expert predictor
            response = await self.call_predictor(
                self.predictor,
                problem_description=problem_description,
                domain=self.domain
            )
            
            # Process response
            result = {
                "domain": self.domain,
                "expert_reasoning": response.expert_reasoning,
                "expert_solution": response.expert_solution,
                "expert_alternatives": response.expert_alternatives,
                "confidence": response.confidence,
            }
            
            # Add references if requested
            if self.include_references:
                result["references"] = response.references
            
            # Add base reasoning if available
            if base_result:
                result["general_reasoning"] = base_result.get("reasoning_steps")
                result["general_solution"] = base_result.get("solution")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in expert reasoning: {str(e)}")
            raise


# Define common domain-specific reasoning modules
class SoftwareDevelopmentReasoning(ExpertReasoningModule):
    """Module for software development reasoning."""
    
    def __init__(self, **kwargs):
        """Initialize the software development reasoning module."""
        super().__init__(domain="software_development", **kwargs)


class DataScienceReasoning(ExpertReasoningModule):
    """Module for data science reasoning."""
    
    def __init__(self, **kwargs):
        """Initialize the data science reasoning module."""
        super().__init__(domain="data_science", **kwargs)


class BusinessStrategyReasoning(ExpertReasoningModule):
    """Module for business strategy reasoning."""
    
    def __init__(self, **kwargs):
        """Initialize the business strategy reasoning module."""
        super().__init__(domain="business_strategy", **kwargs)


# Export
__all__ = [
    "ReasoningSignature",
    "ExpertReasoningSignature",
    "ReasoningModule",
    "ExpertReasoningModule",
    "SoftwareDevelopmentReasoning",
    "DataScienceReasoning",
    "BusinessStrategyReasoning",
]