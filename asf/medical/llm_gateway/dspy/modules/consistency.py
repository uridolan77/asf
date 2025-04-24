"""
Consistency Checker Module

This module provides DSPy modules for detecting inconsistencies in data.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import dspy

from .base import DSPyModuleBase
from .enhanced_base import EnhancedDSPyModuleBase

# Set up logging
logger = logging.getLogger(__name__)


class ConsistencyCheckSignature(dspy.Signature):
    """Signature for general consistency checking."""
    
    statement1 = dspy.InputField(desc="First statement to compare")
    statement2 = dspy.InputField(desc="Second statement to compare")
    is_consistent = dspy.OutputField(desc="Whether the statements are consistent with each other (yes/no)")
    reason = dspy.OutputField(desc="Explanation of why the statements are consistent or inconsistent")


class TemporalConsistencySignature(dspy.Signature):
    """Signature for temporal consistency checking."""
    
    statement1 = dspy.InputField(desc="First statement with timestamp")
    statement2 = dspy.InputField(desc="Second statement with timestamp")
    timestamp1 = dspy.InputField(desc="Timestamp of first statement")
    timestamp2 = dspy.InputField(desc="Timestamp of second statement")
    is_consistent = dspy.OutputField(desc="Whether the statements are consistent over time (yes/no)")
    reason = dspy.OutputField(desc="Explanation of why the statements are temporally consistent or inconsistent")
    natural_progression = dspy.OutputField(desc="Whether any inconsistency could be explained by natural change over time")


class ConsistencyChecker(DSPyModuleBase):
    """Module for checking consistency between statements."""
    
    def __init__(
        self,
        detailed_reasons: bool = True,
    ):
        """
        Initialize the consistency checker.
        
        Args:
            detailed_reasons: Whether to include detailed reasoning
        """
        super().__init__()
        self.detailed_reasons = detailed_reasons
        
        # Create predictor
        self.predictor = dspy.Predict(ConsistencyCheckSignature)
    
    async def predict(
        self,
        statement1: str,
        statement2: str
    ) -> Dict[str, Any]:
        """
        Check consistency between two statements.
        
        Args:
            statement1: First statement
            statement2: Second statement
            
        Returns:
            Dict[str, Any]: Consistency check result
        """
        # Log the request
        logger.info(f"Checking consistency between statements")
        
        try:
            # Call predictor
            response = await self.call_predictor(
                self.predictor,
                statement1=statement1,
                statement2=statement2
            )
            
            # Process response
            result = {
                "is_consistent": response.is_consistent.lower() == "yes",
                "reason": response.reason
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in consistency checking: {str(e)}")
            raise


class TemporalConsistencyChecker(EnhancedDSPyModuleBase):
    """Module for checking consistency between statements over time."""
    
    def __init__(
        self,
        detailed_reasons: bool = True,
        check_natural_progression: bool = True,
    ):
        """
        Initialize the temporal consistency checker.
        
        Args:
            detailed_reasons: Whether to include detailed reasoning
            check_natural_progression: Whether to check if inconsistencies could be natural changes over time
        """
        super().__init__()
        self.detailed_reasons = detailed_reasons
        self.check_natural_progression = check_natural_progression
        
        # Create predictor
        self.predictor = dspy.Predict(TemporalConsistencySignature)
    
    async def _predict_impl(
        self,
        statement1: str,
        statement2: str,
        timestamp1: Optional[str] = None,
        timestamp2: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check consistency between two statements considering timestamps.
        
        Args:
            statement1: First statement
            statement2: Second statement
            timestamp1: Timestamp of first statement
            timestamp2: Timestamp of second statement
            
        Returns:
            Dict[str, Any]: Temporal consistency check result
        """
        # Default timestamps if not provided
        timestamp1 = timestamp1 or "unknown"
        timestamp2 = timestamp2 or "unknown"
        
        # Call predictor
        response = await self.call_predictor(
            self.predictor,
            statement1=statement1,
            statement2=statement2,
            timestamp1=timestamp1,
            timestamp2=timestamp2
        )
        
        # Process response
        result = {
            "is_consistent": response.is_consistent.lower() == "yes",
            "reason": response.reason,
        }
        
        # Add natural progression if requested
        if self.check_natural_progression:
            result["natural_progression"] = response.natural_progression
        
        return result


class MultiStatementConsistencyChecker(DSPyModuleBase):
    """Module for checking consistency across multiple statements."""
    
    def __init__(
        self,
        base_checker: Optional[ConsistencyChecker] = None,
        detailed_reasons: bool = True,
    ):
        """
        Initialize the multi-statement consistency checker.
        
        Args:
            base_checker: Base consistency checker to use
            detailed_reasons: Whether to include detailed reasoning
        """
        super().__init__()
        self.base_checker = base_checker or ConsistencyChecker(detailed_reasons=detailed_reasons)
        self.detailed_reasons = detailed_reasons
    
    async def predict(
        self,
        statements: List[str]
    ) -> Dict[str, Any]:
        """
        Check consistency across multiple statements.
        
        Args:
            statements: List of statements to check
            
        Returns:
            Dict[str, Any]: Consistency check result
        """
        # Log the request
        logger.info(f"Checking consistency across {len(statements)} statements")
        
        try:
            # Perform pairwise comparisons
            results = []
            inconsistencies = []
            
            for i in range(len(statements)):
                for j in range(i + 1, len(statements)):
                    result = await self.base_checker.predict(
                        statement1=statements[i],
                        statement2=statements[j]
                    )
                    
                    pair_result = {
                        "statement1_index": i,
                        "statement2_index": j,
                        "statement1": statements[i],
                        "statement2": statements[j],
                        "is_consistent": result["is_consistent"],
                        "reason": result["reason"]
                    }
                    
                    results.append(pair_result)
                    
                    if not result["is_consistent"]:
                        inconsistencies.append(pair_result)
            
            # Create overall result
            overall_result = {
                "total_statements": len(statements),
                "total_pairs": len(results),
                "inconsistent_pairs": len(inconsistencies),
                "all_consistent": len(inconsistencies) == 0,
                "detailed_results": results
            }
            
            if inconsistencies:
                overall_result["inconsistencies"] = inconsistencies
            
            return overall_result
            
        except Exception as e:
            logger.error(f"Error in multi-statement consistency checking: {str(e)}")
            raise


# Export
__all__ = [
    "ConsistencyCheckSignature",
    "TemporalConsistencySignature",
    "ConsistencyChecker",
    "TemporalConsistencyChecker",
    "MultiStatementConsistencyChecker",
]