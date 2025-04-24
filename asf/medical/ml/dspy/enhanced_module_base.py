"""Enhanced DSPy Module Base Classes

This module provides enhanced base classes for DSPy modules with improved
security, error handling, and audit logging for medical research applications.
"""

import logging
import time
import re
import traceback
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable, Type, TypeVar, Generic
from datetime import datetime
import uuid
import json
import inspect
from functools import wraps

import dspy

from .enhanced_audit_logging import get_audit_logger

# Set up logging
logger = logging.getLogger(__name__)

# Type variables for input/output
T_Input = TypeVar('T_Input')
T_Output = TypeVar('T_Output')


def log_execution_time(func):
    """Decorator to log execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper


def handle_exceptions(func):
    """Decorator to handle exceptions in a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            
            # Log error to audit logger
            audit_logger = get_audit_logger()
            audit_logger.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                context={"function": func.__name__}
            )
            
            # Re-raise the exception
            raise
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            
            # Log error to audit logger
            audit_logger = get_audit_logger()
            audit_logger.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                context={"function": func.__name__}
            )
            
            # Re-raise the exception
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper


class EnhancedDSPyModuleBase(Generic[T_Input, T_Output]):
    """Enhanced base class for DSPy modules with improved security and error handling.
    
    This class provides a foundation for building DSPy modules with enhanced
    security, error handling, and audit logging for medical research applications.
    """
    
    def __init__(self, **kwargs):
        """Initialize the enhanced DSPy module base.
        
        Args:
            **kwargs: Additional arguments
        """
        # Initialize audit logger
        self.audit_logger = get_audit_logger()
        
        # Initialize PHI detection patterns
        self.phi_patterns = [
            # Patient identifiers
            r'\b(?:patient|subject|participant)\s*(?:id|number|#)\s*[:=]?\s*[a-zA-Z0-9_-]{4,}',
            
            # Names
            r'\b(?:[A-Z][a-z]+\s+){1,2}[A-Z][a-z]+\b',  # Simple name pattern (e.g., John Smith)
            
            # Dates
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
            
            # Phone numbers
            r'\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # (123) 456-7890 or 123-456-7890
            
            # SSN
            r'\b\d{3}[-]\d{2}[-]\d{4}\b',  # 123-45-6789
            
            # MRN (Medical Record Number)
            r'\b(?:MRN|Medical Record Number)\s*[:=]?\s*[a-zA-Z0-9_-]{4,}\b',
            
            # Email addresses
            r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
        ]
        
        # Compile PHI patterns
        self._compiled_phi_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.phi_patterns]
        
        # Initialize operation ID for tracking
        self._operation_id = None
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize input text to prevent prompt injection and other security issues.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            str: Sanitized text
        """
        if not isinstance(text, str):
            return text
        
        # Remove potential control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Remove potential prompt injection patterns
        sanitized = re.sub(r'(?i)ignore\s+(?:previous|above).*instructions', '[FILTERED]', sanitized)
        sanitized = re.sub(r'(?i)disregard\s+(?:previous|above).*instructions', '[FILTERED]', sanitized)
        
        return sanitized
    
    def sanitize_output(self, text: str) -> str:
        """Sanitize output text to prevent leakage of sensitive information.
        
        Args:
            text: Output text to sanitize
            
        Returns:
            str: Sanitized text
        """
        if not isinstance(text, str):
            return text
        
        sanitized = text
        
        # Apply PHI redaction for each sensitive pattern
        for i, pattern in enumerate(self._compiled_phi_patterns):
            sanitized = pattern.sub(f"[REDACTED-{i}]", sanitized)
        
        return sanitized
    
    def log_audit(self, event_type: str, inputs: Dict[str, Any], outputs: Dict[str, Any], error: Optional[str] = None):
        """Log an audit event.
        
        Args:
            event_type: Type of event
            inputs: Input data
            outputs: Output data
            error: Error message if applicable
        """
        # Generate operation ID if not already set
        if self._operation_id is None:
            self._operation_id = str(uuid.uuid4())
        
        # Log event
        self.audit_logger.log_event(
            event_type=event_type,
            component=self.__class__.__name__,
            inputs=inputs,
            outputs=outputs,
            error=error,
            correlation_id=self._operation_id
        )
    
    @log_execution_time
    @handle_exceptions
    def __call__(self, *args, **kwargs) -> T_Output:
        """Call the module.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            T_Output: Module output
        """
        # Generate new operation ID for this call
        self._operation_id = str(uuid.uuid4())
        
        # Get start time
        start_time = time.time()
        
        # Log audit event for the start of processing
        self.log_audit(
            f"{self.__class__.__name__}_START",
            {"args": str(args), "kwargs": {k: str(v) for k, v in kwargs.items()}},
            {}
        )
        
        # Call forward method
        result = self.forward(*args, **kwargs)
        
        # Get end time
        end_time = time.time()
        
        # Log audit event for the end of processing
        self.log_audit(
            f"{self.__class__.__name__}_COMPLETE",
            {"args": str(args), "kwargs": {k: str(v) for k, v in kwargs.items()}},
            {"result": result, "execution_time": end_time - start_time}
        )
        
        return result
    
    def forward(self, *args, **kwargs) -> T_Output:
        """Forward method to be implemented by subclasses.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            T_Output: Module output
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_prompt_templates(self) -> Dict[str, str]:
        """Get prompt templates used by this module.
        
        Returns:
            Dict[str, str]: Dictionary of prompt templates
        """
        templates = {}
        
        # Extract from this module if it's a ChainOfThought
        if isinstance(self, dspy.ChainOfThought):
            if hasattr(self, "prompt_template"):
                templates["cot_template"] = getattr(self, "prompt_template", "")
        
        # Extract from predict modules
        for name, attr in self.__dict__.items():
            if isinstance(attr, dspy.Predict):
                if hasattr(attr, "prompt_template"):
                    templates[f"{name}_template"] = getattr(attr, "prompt_template", "")
        
        # For nested modules, recursively extract
        for name, attr in self.__dict__.items():
            if isinstance(attr, dspy.Module) and not isinstance(attr, dspy.Predict):
                try:
                    # Try using the protocol method if available
                    if hasattr(attr, "get_prompt_templates"):
                        nested_templates = attr.get_prompt_templates()
                    # Fallback to introspection
                    else:
                        nested_templates = self._extract_prompts_recursively(attr)
                        
                    for nested_name, nested_template in nested_templates.items():
                        templates[f"{name}.{nested_name}"] = nested_template
                except Exception as e:
                    logger.warning(f"Failed to extract templates from nested module {name}: {str(e)}")
        
        return templates
    
    def _extract_prompts_recursively(self, module: dspy.Module) -> Dict[str, str]:
        """Extract prompts recursively from a module.
        
        Args:
            module: Module to extract prompts from
            
        Returns:
            Dict[str, str]: Dictionary of prompt templates
        """
        templates = {}
        
        # Extract from this module if it's a ChainOfThought
        if isinstance(module, dspy.ChainOfThought):
            if hasattr(module, "prompt_template"):
                templates["cot_template"] = getattr(module, "prompt_template", "")
        
        # Extract from predict modules
        for name, attr in module.__dict__.items():
            if isinstance(attr, dspy.Predict):
                if hasattr(attr, "prompt_template"):
                    templates[f"{name}_template"] = getattr(attr, "prompt_template", "")
        
        # For nested modules, recursively extract
        for name, attr in module.__dict__.items():
            if isinstance(attr, dspy.Module) and not isinstance(attr, dspy.Predict):
                try:
                    # Try using the protocol method if available
                    if hasattr(attr, "get_prompt_templates"):
                        nested_templates = attr.get_prompt_templates()
                    # Fallback to introspection
                    else:
                        nested_templates = self._extract_prompts_recursively(attr)
                        
                    for nested_name, nested_template in nested_templates.items():
                        templates[f"{name}.{nested_name}"] = nested_template
                except Exception as e:
                    logger.warning(f"Failed to extract templates from nested module {name}: {str(e)}")
        
        return templates
    
    def get_signature_info(self) -> Dict[str, Any]:
        """Get signature information for this module.
        
        Returns:
            Dict[str, Any]: Signature information
        """
        signature_info = {
            'inputs': [],
            'outputs': [],
            'constraints': {},
            'description': getattr(self, '__doc__', '')
        }
        
        # Extract from signature if available
        if hasattr(self, '_signature'):
            sig = getattr(self, '_signature', None)
            if sig:
                # Extract input and output fields
                if hasattr(sig, 'input_fields'):
                    signature_info['inputs'] = [f.name for f in sig.input_fields]
                if hasattr(sig, 'output_fields'):
                    signature_info['outputs'] = [f.name for f in sig.output_fields]
                
                # Add any field descriptions or constraints
                for field in getattr(sig, 'fields', []):
                    if hasattr(field, 'description') and field.description:
                        signature_info.setdefault('field_descriptions', {})[field.name] = field.description
                    
                    # Extract constraints like max_length if present
                    for attr_name in ['max_length', 'min_length', 'allowed_values']:
                        if hasattr(field, attr_name):
                            value = getattr(field, attr_name, None)
                            if value is not None:
                                signature_info['constraints'].setdefault(field.name, {})[attr_name] = value
        
        return signature_info


class EnhancedRAGModule(EnhancedDSPyModuleBase):
    """Enhanced RAG module base class.
    
    This class provides a foundation for building RAG modules with enhanced
    security, error handling, and audit logging for medical research applications.
    """
    
    def __init__(self, retriever: Optional[dspy.Module] = None, **kwargs):
        """Initialize the enhanced RAG module.
        
        Args:
            retriever: Retriever module
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.retriever = retriever
    
    @log_execution_time
    @handle_exceptions
    async def forward(self, question: str, **kwargs) -> Dict[str, Any]:
        """Forward method to be implemented by subclasses.
        
        Args:
            question: Question to answer
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: RAG result
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement forward method")


class EnhancedChainOfThoughtModule(EnhancedDSPyModuleBase):
    """Enhanced chain-of-thought module base class.
    
    This class provides a foundation for building chain-of-thought modules with enhanced
    security, error handling, and audit logging for medical research applications.
    """
    
    def __init__(self, signature: Type[dspy.Signature], **kwargs):
        """Initialize the enhanced chain-of-thought module.
        
        Args:
            signature: DSPy signature
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.signature = signature
        self.cot = dspy.ChainOfThought(signature)
    
    @log_execution_time
    @handle_exceptions
    def forward(self, **kwargs) -> Dict[str, Any]:
        """Forward method to be implemented by subclasses.
        
        Args:
            **kwargs: Keyword arguments
            
        Returns:
            Dict[str, Any]: Chain-of-thought result
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement forward method")
