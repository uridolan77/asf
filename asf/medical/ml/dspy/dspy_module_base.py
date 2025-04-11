"""
DSPy Module Base Classes

This module provides base classes and protocols for DSPy modules.
It defines standard interfaces and implementations for different types of DSPy modules.
"""

import logging
from typing import Dict, Any, List, Optional, Protocol, Type, TypeVar, Generic, runtime_checkable
from abc import ABC, abstractmethod

import dspy

# Set up logging
logger = logging.getLogger(__name__)

# Define type variables for input/output
T_Input = TypeVar('T_Input')
T_Output = TypeVar('T_Output')


@runtime_checkable
class DSPyModuleProtocol(Protocol, Generic[T_Input, T_Output]):
    """
    Protocol defining the expected interface of DSPy modules.
    
    All custom DSPy modules should implement this protocol to ensure
    consistent behavior with the DSPyClient framework.
    """
    
    def get_prompt_templates(self) -> Dict[str, str]:
        """
        Return a dictionary of prompt templates used by this module.
        
        Returns:
            Dict[str, str]: A dictionary mapping template names to their content
        
        Examples:
            >>> module.get_prompt_templates()
            {'main_template': 'Answer the question: {question}'}
        """
        ...
    
    def get_signature_info(self) -> Dict[str, Any]:
        """
        Return metadata about the module's signature.
        
        Returns:
            Dict[str, Any]: Information about input/output fields and constraints
        
        Examples:
            >>> module.get_signature_info()
            {
                'inputs': ['question', 'context'],
                'outputs': ['answer', 'confidence'],
                'constraints': {'max_length': 500}
            }
        """
        ...
    
    def __call__(self, *args: Any, **kwargs: Any) -> T_Output:
        """
        Execute the module with the given inputs.
        
        Args:
            *args: Positional arguments to the module
            **kwargs: Keyword arguments to the module
            
        Returns:
            T_Output: The module's output
        
        Raises:
            ValueError: If required inputs are missing
            RuntimeError: If execution fails
        """
        ...


class DSPyModuleBase(dspy.Module, Generic[T_Input, T_Output]):
    """
    Base class for custom DSPy modules with standardized interface methods.
    
    This class provides default implementations of the DSPyModuleProtocol
    that use introspection to extract information from DSPy modules.
    
    Attributes:
        _signature_info (Dict[str, Any]): Cached signature information
        _prompt_templates (Dict[str, str]): Cached prompt templates
    """
    
    def __init__(self):
        """Initialize the base module with empty caches."""
        super().__init__()
        self._signature_info = None
        self._prompt_templates = None
    
    def get_prompt_templates(self) -> Dict[str, str]:
        """
        Extract prompt templates from this module using introspection.
        
        Returns:
            Dict[str, str]: A dictionary mapping template names to their content
        """
        # Implement caching to avoid repeated introspection
        if self._prompt_templates is not None:
            return self._prompt_templates
            
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
                    if isinstance(attr, DSPyModuleProtocol):
                        nested_templates = attr.get_prompt_templates()
                    # Fallback to introspection
                    else:
                        nested_templates = self._extract_prompts_recursively(attr)
                        
                    for nested_name, nested_template in nested_templates.items():
                        templates[f"{name}.{nested_name}"] = nested_template
                except Exception as e:
                    logger.warning(f"Failed to extract templates from nested module {name}: {str(e)}")
        
        # Cache the results
        self._prompt_templates = templates
        return templates
    
    def get_signature_info(self) -> Dict[str, Any]:
        """
        Extract signature information from this module.
        
        Returns:
            Dict[str, Any]: Information about input/output fields and constraints
        """
        if self._signature_info is not None:
            return self._signature_info
            
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
        
        # Cache the results
        self._signature_info = signature_info
        return signature_info
    
    def _extract_prompts_recursively(self, module: dspy.Module) -> Dict[str, str]:
        """
        Recursively extract prompt templates from a module.
        
        Args:
            module: The module to extract prompts from
            
        Returns:
            Dict[str, str]: A dictionary of prompt templates
        """
        templates = {}
        
        # Handle ChainOfThought
        if isinstance(module, dspy.ChainOfThought):
            if hasattr(module, "prompt_template"):
                templates["cot_template"] = getattr(module, "prompt_template", "")
        
        # Handle Predict modules
        for name, attr in module.__dict__.items():
            if isinstance(attr, dspy.Predict):
                if hasattr(attr, "prompt_template"):
                    templates[f"{name}_template"] = getattr(attr, "prompt_template", "")
        
        # Recursively process nested modules
        for name, attr in module.__dict__.items():
            if isinstance(attr, dspy.Module) and not isinstance(attr, dspy.Predict):
                try:
                    nested_templates = self._extract_prompts_recursively(attr)
                    for nested_name, nested_template in nested_templates.items():
                        templates[f"{name}.{nested_name}"] = nested_template
                except Exception:
                    pass  # Skip any modules that can't be processed
        
        return templates


class MedicalDSPyModule(DSPyModuleBase[T_Input, T_Output]):
    """
    Base class for medical-specific DSPy modules.
    
    This class extends DSPyModuleBase with medical-specific functionality.
    
    Attributes:
        phi_detection_enabled (bool): Whether PHI detection is enabled
        audit_logging_enabled (bool): Whether audit logging is enabled
    """
    
    def __init__(self, phi_detection_enabled: bool = True, audit_logging_enabled: bool = True):
        """
        Initialize the medical DSPy module.
        
        Args:
            phi_detection_enabled: Whether to enable PHI detection
            audit_logging_enabled: Whether to enable audit logging
        """
        super().__init__()
        self.phi_detection_enabled = phi_detection_enabled
        self.audit_logging_enabled = audit_logging_enabled
    
    def sanitize_input(self, text: str) -> str:
        """
        Sanitize input text to remove potential PHI.
        
        Args:
            text: The text to sanitize
            
        Returns:
            str: The sanitized text
        """
        if not self.phi_detection_enabled:
            return text
        
        # Implement PHI detection and redaction
        # This is a placeholder - in a real implementation, you would use a more
        # sophisticated PHI detection and redaction system
        import re
        
        # Simple pattern matching for common PHI
        patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Dates
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # Dates
            r'\b\d{5}(?:-\d{4})?\b'  # ZIP
        ]
        
        sanitized = text
        for pattern in patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized)
        
        return sanitized
    
    def sanitize_output(self, text: str) -> str:
        """
        Sanitize output text to remove potential PHI.
        
        Args:
            text: The text to sanitize
            
        Returns:
            str: The sanitized text
        """
        # Use the same sanitization logic as for input
        return self.sanitize_input(text)
    
    def log_audit(self, operation: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Log an audit record for this operation.
        
        Args:
            operation: The operation being performed
            inputs: The inputs to the operation
            outputs: The outputs from the operation
        """
        if not self.audit_logging_enabled:
            return
        
        # Sanitize inputs and outputs
        sanitized_inputs = {k: self.sanitize_input(v) if isinstance(v, str) else str(v) for k, v in inputs.items()}
        sanitized_outputs = {k: self.sanitize_output(v) if isinstance(v, str) else str(v) for k, v in outputs.items()}
        
        # Log the audit record
        logger.info(f"AUDIT: {operation} - Inputs: {sanitized_inputs} - Outputs: {sanitized_outputs}")
        
        # In a real implementation, you would log to a secure, immutable audit log
        # This might involve writing to a database, sending to a logging service, etc.


class RAGModule(MedicalDSPyModule[Dict[str, Any], Dict[str, Any]]):
    """
    Base class for RAG (Retrieval-Augmented Generation) modules.
    
    This class provides a standard interface for RAG modules.
    """
    
    def __init__(
        self,
        retriever: Optional[dspy.Module] = None,
        generator: Optional[dspy.Module] = None,
        **kwargs
    ):
        """
        Initialize the RAG module.
        
        Args:
            retriever: The retrieval module
            generator: The generation module
            **kwargs: Additional arguments to pass to the parent class
        """
        super().__init__(**kwargs)
        self.retriever = retriever or self._create_default_retriever()
        self.generator = generator or self._create_default_generator()
    
    def _create_default_retriever(self) -> dspy.Module:
        """
        Create a default retriever module.
        
        Returns:
            dspy.Module: The default retriever
        """
        return dspy.Retrieve(k=3)
    
    def _create_default_generator(self) -> dspy.Module:
        """
        Create a default generator module.
        
        Returns:
            dspy.Module: The default generator
        """
        return dspy.Generate("context, question -> answer")
    
    def forward(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the RAG pipeline.
        
        Args:
            question: The question to answer
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: The RAG output
        """
        # Log audit
        self.log_audit("RAG_FORWARD", {"question": question, **kwargs}, {})
        
        # Sanitize input
        sanitized_question = self.sanitize_input(question)
        
        # Retrieve relevant passages
        retrieval_result = self.retriever(sanitized_question)
        context = retrieval_result.passages if hasattr(retrieval_result, 'passages') else retrieval_result
        
        # Generate answer
        generation_result = self.generator(context=context, question=sanitized_question)
        
        # Extract answer
        if hasattr(generation_result, 'answer'):
            answer = generation_result.answer
        elif isinstance(generation_result, dict) and 'answer' in generation_result:
            answer = generation_result['answer']
        else:
            answer = str(generation_result)
        
        # Sanitize output
        sanitized_answer = self.sanitize_output(answer)
        
        # Prepare result
        result = {
            'question': question,
            'context': context,
            'answer': sanitized_answer
        }
        
        # Log audit
        self.log_audit("RAG_FORWARD", {"question": question, **kwargs}, result)
        
        return result


class ChainOfThoughtModule(MedicalDSPyModule[Dict[str, Any], Dict[str, Any]]):
    """
    Base class for Chain of Thought modules.
    
    This class provides a standard interface for Chain of Thought modules.
    """
    
    def __init__(
        self,
        signature: Optional[Type[dspy.Signature]] = None,
        **kwargs
    ):
        """
        Initialize the Chain of Thought module.
        
        Args:
            signature: The signature for the module
            **kwargs: Additional arguments to pass to the parent class
        """
        super().__init__(**kwargs)
        self.cot = dspy.ChainOfThought(signature) if signature else None
    
    def forward(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the Chain of Thought module.
        
        Args:
            **kwargs: Input arguments
            
        Returns:
            Dict[str, Any]: The module output
        """
        # Log audit
        self.log_audit("COT_FORWARD", kwargs, {})
        
        # Sanitize inputs
        sanitized_inputs = {k: self.sanitize_input(v) if isinstance(v, str) else v for k, v in kwargs.items()}
        
        # Execute Chain of Thought
        if self.cot:
            result = self.cot(**sanitized_inputs)
        else:
            raise ValueError("Chain of Thought module not initialized with a signature")
        
        # Convert to dictionary if needed
        if hasattr(result, '__dict__'):
            result_dict = result.__dict__
        elif isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {'result': str(result)}
        
        # Sanitize outputs
        sanitized_outputs = {k: self.sanitize_output(v) if isinstance(v, str) else v for k, v in result_dict.items()}
        
        # Log audit
        self.log_audit("COT_FORWARD", kwargs, sanitized_outputs)
        
        return sanitized_outputs
