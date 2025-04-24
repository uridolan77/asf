"""
DSPy Client

This module provides a client for DSPy that integrates with the LLM Gateway provider system.
"""

import logging
import functools
import asyncio
from typing import Dict, Any, Optional, Union, List, Type

import dspy
from dspy import Signature, InputField, OutputField, Predict
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

from .config import get_dspy_settings, LLMProvider
from .modules.base import DSPyModuleProtocol
from .utils.circuit_breaker import CircuitBreaker, AsyncCircuitBreaker

# Set up logging
logger = logging.getLogger(__name__)


class DSPyClient:
    """Client for DSPy that integrates with the LLM Gateway provider system."""
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        circuit_breaker: Optional[Union[CircuitBreaker, AsyncCircuitBreaker]] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the DSPy client.
        
        Args:
            provider: LLM provider name
            model_name: Model name
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            circuit_breaker: Circuit breaker for API calls
            api_key: API key for the provider
            **kwargs: Additional provider-specific arguments
        """
        # Get settings
        self.settings = get_dspy_settings()
        
        # Override settings with provided arguments
        self.provider = provider or self.settings.provider
        self.model_name = model_name or self.settings.model_name
        self.temperature = temperature if temperature is not None else self.settings.temperature
        self.max_tokens = max_tokens or self.settings.max_tokens
        self.circuit_breaker = circuit_breaker
        
        # Set up language model for DSPy
        self._lm = self._create_language_model(api_key=api_key, **kwargs)
        
        # Configure DSPy with the language model
        dspy.settings.configure(lm=self._lm)
    
    def _create_language_model(self, api_key: Optional[str] = None, **kwargs) -> Any:
        """
        Create a language model for DSPy based on the provider.
        
        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Any: Language model for DSPy
        """
        provider = self.provider
        model_name = self.model_name
        
        # OpenAI
        if provider == LLMProvider.OPENAI:
            # Get API key
            api_key = api_key or self.settings.openai_api_key
            
            # Import inside function to avoid hard dependency
            from dspy.openai import OpenAI
            
            return OpenAI(
                model=model_name,
                api_key=api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
        
        # Anthropic
        elif provider == LLMProvider.ANTHROPIC:
            # Get API key
            api_key = api_key or self.settings.anthropic_api_key
            
            # Import inside function to avoid hard dependency
            from dspy.anthropic import Anthropic
            
            return Anthropic(
                model=model_name,
                api_key=api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
        
        # Azure OpenAI
        elif provider == LLMProvider.AZURE_OPENAI:
            # Get API key and endpoint
            api_key = api_key or self.settings.azure_openai_api_key
            endpoint = kwargs.get("endpoint") or self.settings.azure_openai_endpoint
            
            # Import inside function to avoid hard dependency
            from dspy.azure import AzureOpenAI
            
            return AzureOpenAI(
                deployment_name=model_name,
                api_key=api_key,
                endpoint=endpoint,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
        
        # Hugging Face
        elif provider == LLMProvider.HUGGING_FACE:
            # Get API key
            api_key = api_key or self.settings.hugging_face_api_key
            
            # Import inside function to avoid hard dependency
            try:
                from dspy.hf import HuggingFace
                
                return HuggingFace(
                    model=model_name,
                    api_key=api_key,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs
                )
            except ImportError:
                logger.error("Failed to import HuggingFace support for DSPy")
                raise ImportError("Please install the required packages for HuggingFace support")
        
        # MCP
        elif provider == LLMProvider.MCP:
            # Get endpoint
            endpoint = kwargs.get("endpoint") or self.settings.mcp_endpoint
            
            # Import inside function to avoid hard dependency
            try:
                # Assuming MCP support is through a custom adapter for DSPy
                from ..lm.mcp_lm import MCPLanguageModel
                
                return MCPLanguageModel(
                    model=model_name,
                    endpoint=endpoint,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs
                )
            except ImportError:
                logger.error("Failed to import MCP support for DSPy")
                raise ImportError("Please install the required packages for MCP support")
        
        # Mock
        elif provider == LLMProvider.MOCK:
            # Import inside function to avoid hard dependency
            from dspy.mock import MockLM
            
            return MockLM(
                **kwargs
            )
        
        # Unknown provider
        else:
            logger.error(f"Unknown provider: {provider}")
            raise ValueError(f"Unknown provider: {provider}")
    
    async def call_module(
        self,
        module_name: str,
        module: DSPyModuleProtocol,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call a DSPy module.
        
        Args:
            module_name: Name of the module for logging
            module: DSPy module to call
            **kwargs: Module arguments
            
        Returns:
            Dict[str, Any]: Module result
        """
        try:
            # Set the language model on the module if it has that attribute
            if hasattr(module, "lm") and module.lm is None:
                module.lm = self._lm
            
            # Call the module
            result = await module.predict(**kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calling module {module_name}: {str(e)}")
            raise
    
    def optimize_module(
        self,
        module_type: Type,
        metric: Any,
        train_data: List[Any],
        num_trials: int = 10,
        **kwargs
    ) -> Any:
        """
        Optimize a DSPy module using bootstrapping.
        
        Args:
            module_type: Type of module to optimize
            metric: Metric for optimization
            train_data: Training data
            num_trials: Number of trials for optimization
            **kwargs: Additional optimization arguments
            
        Returns:
            Any: Optimized module
        """
        try:
            # Create bootstrapper
            bootstrapper = BootstrapFewShotWithRandomSearch(
                metric=metric,
                num_trials=num_trials,
                **kwargs
            )
            
            # Optimize
            optimized_module = bootstrapper.compile(
                module_type,
                train_data=train_data,
                **kwargs
            )
            
            return optimized_module
            
        except Exception as e:
            logger.error(f"Error optimizing module: {str(e)}")
            raise


@functools.lru_cache()
async def get_enhanced_client() -> DSPyClient:
    """
    Get enhanced DSPy client singleton with proper configuration.
    
    Returns:
        DSPyClient: Enhanced DSPy client
    """
    # Create circuit breaker if needed
    settings = get_dspy_settings()
    
    circuit_breaker = None
    if settings.circuit_breaker_max_failures > 0:
        circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=settings.circuit_breaker_max_failures,
            recovery_timeout=settings.circuit_breaker_reset_timeout
        )
    
    # Create client
    client = DSPyClient(
        provider=settings.provider,
        model_name=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        circuit_breaker=circuit_breaker
    )
    
    return client


# Export
__all__ = [
    "DSPyClient",
    "get_enhanced_client",
]
