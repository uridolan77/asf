"""Enhanced DSPy Module Base

This module provides enhanced base classes for DSPy modules with better error handling,
PHI detection, and audit logging for medical research applications.
"""

import logging
import time
import re
import traceback
from typing import Dict, Any, List, Optional, Union, Protocol, runtime_checkable
import dspy

from audit_logging import get_audit_logger
from settings import get_enhanced_settings

# Set up logging
logger = logging.getLogger(__name__)


@runtime_checkable
class EnhancedDSPyModuleProtocol(Protocol):
    """Protocol defining the expected interface of enhanced DSPy modules.

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

    def sanitize_input(self, text: str) -> str:
        """
        Sanitize input text to remove potential security issues.

        Args:
            text: Input text to sanitize

        Returns:
            str: Sanitized text
        """
        ...

    def sanitize_output(self, text: str) -> str:
        """
        Sanitize output text to remove potential PHI/PII.

        Args:
            text: Output text to sanitize

        Returns:
            str: Sanitized text
        """
        ...

    def log_audit(self, event_type: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Log an audit event for this module.

        Args:
            event_type: Type of event
            inputs: Input data
            outputs: Output data
        """
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the module with the given inputs.

        Args:
            *args: Positional arguments to the module
            **kwargs: Keyword arguments to the module

        Returns:
            Any: The module's output

        Raises:
            ValueError: If required inputs are missing
            RuntimeError: If execution fails
        """
        ...


class EnhancedDSPyModuleBase:
    """Enhanced base class for custom DSPy modules with standardized interface methods.

    This class provides default implementations of the EnhancedDSPyModuleProtocol
    that use introspection to extract information from DSPy modules.

    Attributes:
        _signature_info (Dict[str, Any]): Cached signature information
        _prompt_templates (Dict[str, str]): Cached prompt templates
    """

    def __init__(self):
        """Initialize the base module with empty caches.
        
        Initializes the module with empty caches for signature information and prompt templates.
        Sets up configuration settings and audit logging capabilities.
        """
        # Initialize caches
        self._signature_info = None
        self._prompt_templates = None
        
        # Get settings and audit logger
        self._settings = get_enhanced_settings()
        self._audit_logger = get_audit_logger()
        
        # Compile PHI patterns for output sanitization
        self._compiled_phi_patterns = [
            re.compile(pattern) for pattern in self._settings.PHI_DETECTION_PATTERNS
        ] if hasattr(self._settings, 'PHI_DETECTION_PATTERNS') else []

    def _extract_prompts_recursively(self, module) -> Dict[str, str]:
        """Recursively extract prompt templates from a module.

        Args:
            module: The module to extract from

        Returns:
            Dict[str, str]: A dictionary mapping template names to their content
        """
        templates = {}

        # Extract from ChainOfThought
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
                    nested_templates = self._extract_prompts_recursively(attr)
                    for nested_name, nested_template in nested_templates.items():
                        templates[f"{name}.{nested_name}"] = nested_template
                except Exception:
                    pass

        return templates

    def get_signature_info(self) -> Dict[str, Any]:
        """Extract signature information from this module.

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

    def sanitize_input(self, text: str) -> str:
        """Sanitize input text to remove potential security issues.

        Args:
            text: Input text to sanitize

        Returns:
            str: Sanitized text
        """
        if not text or not isinstance(text, str):
            return ""

        # Check if input validation is enabled
        if not self._settings.ENABLE_INPUT_VALIDATION:
            return text

        sanitized = text

        # Truncate if too long
        if len(sanitized) > self._settings.MAX_PROMPT_LENGTH:
            sanitized = sanitized[:self._settings.MAX_PROMPT_LENGTH]
            logger.warning(f"Input text truncated to {self._settings.MAX_PROMPT_LENGTH} characters")

        # Remove potential injection patterns
        # This is a basic implementation - in production, use a more robust solution
        injection_patterns = [
            r'<script.*?>.*?</script>',  # Basic script tags
            r'javascript:',  # JavaScript protocol
            r'data:text/html',  # Data URI
            r'<!--.*?-->',  # HTML comments
            r'<style.*?>.*?</style>',  # Style tags
            r'<iframe.*?>.*?</iframe>',  # iframes
        ]

        for pattern in injection_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)

        return sanitized

    def sanitize_output(self, text: str) -> str:
        """Sanitize output text to remove potential PHI/PII.

        Args:
            text: Output text to sanitize

        Returns:
            str: Sanitized text
        """
        if not text or not isinstance(text, str):
            return ""

        # Check if PHI detection is enabled
        if not self._settings.ENABLE_PHI_DETECTION:
            return text

        sanitized = text

        # Apply each pattern
        for i, pattern in enumerate(self._compiled_phi_patterns):
            sanitized = pattern.sub(f"[REDACTED-PHI-{i}]", sanitized)

        return sanitized

    def log_audit(self, event_type: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Log an audit event for this module.

        Args:
            event_type: Type of event
            inputs: Input data
            outputs: Output data
        """
        # Check if audit logging is enabled
        if not self._settings.ENABLE_AUDIT_LOGGING:
            return

        # Get module name and type
        module_name = self.__class__.__name__
        module_type = self.__class__.__module__ + "." + self.__class__.__name__

        # Log the event
        self._audit_logger.log_module_call(
            module_name=module_name,
            module_type=module_type,
            inputs=inputs,
            outputs=outputs,
            error=None,
            latency=None
        )

    def forward(self, *args, **kwargs) -> Any:
        """Execute the module's forward method.

        This method should be implemented by subclasses.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Any: Module output

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the module with the given inputs.

        Args:
            *args: Positional arguments to the module
            **kwargs: Keyword arguments to the module

        Returns:
            Any: The module's output

        Raises:
            ValueError: If required inputs are missing
            RuntimeError: If execution fails
        """
        # Record start time for latency measurement
        start_time = time.time()

        # Sanitize inputs
        sanitized_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                sanitized_kwargs[key] = self.sanitize_input(value)
            else:
                sanitized_kwargs[key] = value

        # Log audit event for module call
        if self._settings.ENABLE_AUDIT_LOGGING:
            self._audit_logger.log_module_call(
                module_name=self.__class__.__name__,
                module_type=self.__class__.__module__ + "." + self.__class__.__name__,
                inputs=sanitized_kwargs,
                outputs=None,
                error=None,
                latency=None
            )

        try:
            # Call the forward method
            result = self.forward(*args, **sanitized_kwargs)

            # Calculate latency
            latency = time.time() - start_time

            # Sanitize outputs if it's a dictionary
            if isinstance(result, dict):
                sanitized_result = {}
                for key, value in result.items():
                    if isinstance(value, str):
                        sanitized_result[key] = self.sanitize_output(value)
                    else:
                        sanitized_result[key] = value
                result = sanitized_result

            # Log audit event for successful module call
            if self._settings.ENABLE_AUDIT_LOGGING:
                self._audit_logger.log_module_call(
                    module_name=self.__class__.__name__,
                    module_type=self.__class__.__module__ + "." + self.__class__.__name__,
                    inputs=sanitized_kwargs,
                    outputs=result,
                    error=None,
                    latency=latency
                )

            return result
        except Exception as e:
            # Calculate latency
            latency = time.time() - start_time

            # Get stack trace
            stack_trace = traceback.format_exc()

            # Log error
            logger.error(f"Error in module {self.__class__.__name__}: {str(e)}\n{stack_trace}")

            # Log audit event for failed module call
            if self._settings.ENABLE_AUDIT_LOGGING:
                self._audit_logger.log_module_call(
                    module_name=self.__class__.__name__,
                    module_type=self.__class__.__module__ + "." + self.__class__.__name__,
                    inputs=sanitized_kwargs,
                    outputs=None,
                    error=str(e),
                    latency=latency
                )

                self._audit_logger.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    stack_trace=stack_trace,
                    context={"module_name": self.__class__.__name__, "inputs": sanitized_kwargs}
                )

            # Re-raise the exception
            raise


class EnhancedRAGModule(EnhancedDSPyModuleBase):
    """Enhanced base class for RAG (Retrieval-Augmented Generation) modules.

    This class provides a foundation for building RAG modules with
    standardized retrieval and generation components.

    Attributes:
        retriever (dspy.Module): The retrieval component of the RAG pipeline
        generator (dspy.Module): The generation component of the RAG pipeline
    """

    def __init__(
        self,
        retriever: Optional[dspy.Module] = None,
        generator: Optional[dspy.Module] = None,
        **kwargs
    ):
        """Initialize the RAG module.

        Args:
            retriever: Retriever module for document/passage retrieval
            generator: Generator module for text generation based on retrieved content
            **kwargs: Additional configuration parameters
        """
        super().__init__()
        self.retriever = retriever
        self.generator = generator


class EnhancedChainOfThoughtModule(EnhancedDSPyModuleBase):
    """Enhanced base class for Chain of Thought modules.

    This class provides a foundation for building Chain of Thought modules
    with standardized reasoning components.

    Attributes:
        signature (dspy.Signature): The DSPy signature defining inputs and outputs
        cot (dspy.ChainOfThought): The chain of thought component
    """

    def __init__(
        self,
        signature: Optional[dspy.Signature] = None,
        **kwargs
    ):
        """Initialize the Chain of Thought module.

        Args:
            signature: DSPy signature
            **kwargs: Additional arguments
        """
        super().__init__()
        self.signature = signature
        self.cot = dspy.ChainOfThought(signature) if signature else None

    def forward(self, *args, **kwargs) -> Any:
        """Execute the Chain of Thought reasoning.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Any: Chain of Thought output

        Raises:
            ValueError: If no signature or CoT module is provided
        """
        if self.cot is None:
            raise ValueError("No signature or CoT module provided")

        return self.cot(*args, **kwargs)


# Export all classes
__all__ = [
    'EnhancedDSPyModuleProtocol',
    'EnhancedDSPyModuleBase',
    'EnhancedRAGModule',
    'EnhancedChainOfThoughtModule'
]
