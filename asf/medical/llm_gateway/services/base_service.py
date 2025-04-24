"""
Base implementation of the LLM Service Interface.

This module provides a base implementation of the LLM Service Interface
with common functionality for error handling, logging, and metrics.
"""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from asf.medical.llm_gateway.interfaces.llm_service import LLMServiceInterface
from asf.medical.llm_gateway.interfaces.exceptions import (
    LLMServiceException,
    TimeoutException,
    TransportException,
    ServiceUnavailableException
)
from asf.medical.llm_gateway.observability.metrics import MetricsService
from asf.medical.llm_gateway.transport.base import TransportError, RateLimitExceededError, CircuitBreakerOpenError

logger = logging.getLogger(__name__)


class BaseService(LLMServiceInterface):
    """
    Base implementation of the LLM Service Interface.
    
    This class provides common functionality and error handling for all LLM services.
    It implements the LLMServiceInterface abstract methods with error handling
    and delegates to abstract methods that subclasses must implement.
    """
    
    def __init__(self, config: Dict[str, Any], metrics_service: Optional[MetricsService] = None):
        """
        Initialize the base service with configuration.
        
        Args:
            config: Configuration dictionary for the service
            metrics_service: Optional metrics service for instrumentation
        """
        self.config = config
        self.metrics_service = metrics_service or MetricsService()
        self.service_id = config.get('service_id', self.__class__.__name__)
    
    async def generate_text(self, prompt: str, model: str, params: Dict[str, Any]) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            model: The model to use for generation
            params: Additional parameters for the generation
            
        Returns:
            Generated text as string
            
        Raises:
            LLMServiceException: If an error occurs during text generation
        """
        try:
            with self.metrics_service.time_execution(f"{self.service_id}.generate_text"):
                return await self._generate_text_impl(prompt, model, params)
        except Exception as e:
            logger.exception(f"Error in {self.service_id}.generate_text: {str(e)}")
            raise self._translate_error(e)
    
    async def generate_stream(self, prompt: str, model: str, params: Dict[str, Any]) -> AsyncIterator[str]:
        """
        Stream text generation from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            model: The model to use for generation
            params: Additional parameters for the generation
            
        Returns:
            Asynchronous iterator of generated text chunks
            
        Raises:
            LLMServiceException: If an error occurs during text generation streaming
        """
        try:
            # Start timing
            start_time = self.metrics_service.start_timer(f"{self.service_id}.generate_stream")
            
            # Execute the streaming operation
            async for chunk in self._generate_stream_impl(prompt, model, params):
                yield chunk
                
            # Record successful completion
            self.metrics_service.record_success(f"{self.service_id}.generate_stream")
            self.metrics_service.stop_timer(start_time, f"{self.service_id}.generate_stream")
            
        except Exception as e:
            # Record failure
            self.metrics_service.record_failure(f"{self.service_id}.generate_stream")
            logger.exception(f"Error in {self.service_id}.generate_stream: {str(e)}")
            raise self._translate_error(e)
            
    async def get_embeddings(self, 
                           text: Union[str, List[str]], 
                           model: str,
                           params: Optional[Dict[str, Any]] = None) -> List[List[float]]:
        """
        Get embeddings for text.
        
        Args:
            text: The text or list of texts to get embeddings for
            model: The model to use for embeddings
            params: Additional parameters for the embedding generation
            
        Returns:
            List of embedding vectors
            
        Raises:
            LLMServiceException: If an error occurs during embedding generation
        """
        if params is None:
            params = {}
            
        try:
            with self.metrics_service.time_execution(f"{self.service_id}.get_embeddings"):
                return await self._get_embeddings_impl(text, model, params)
        except Exception as e:
            logger.exception(f"Error in {self.service_id}.get_embeddings: {str(e)}")
            raise self._translate_error(e)
    
    async def chat(self, messages: List[Dict[str, str]], model: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Have a chat-based interaction with the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model to use for the chat
            params: Additional parameters for the chat
            
        Returns:
            Dictionary containing the response
            
        Raises:
            LLMServiceException: If an error occurs during chat
        """
        try:
            with self.metrics_service.time_execution(f"{self.service_id}.chat"):
                return await self._chat_impl(messages, model, params)
        except Exception as e:
            logger.exception(f"Error in {self.service_id}.chat: {str(e)}")
            raise self._translate_error(e)
    
    async def chat_stream(self, messages: List[Dict[str, str]], model: str, params: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Have a streaming chat-based interaction with the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model to use for the chat
            params: Additional parameters for the chat
            
        Returns:
            Asynchronous iterator of response chunks
            
        Raises:
            LLMServiceException: If an error occurs during chat streaming
        """
        try:
            # Start timing
            start_time = self.metrics_service.start_timer(f"{self.service_id}.chat_stream")
            
            # Execute the streaming operation
            async for chunk in self._chat_stream_impl(messages, model, params):
                yield chunk
                
            # Record successful completion
            self.metrics_service.record_success(f"{self.service_id}.chat_stream")
            self.metrics_service.stop_timer(start_time, f"{self.service_id}.chat_stream")
            
        except Exception as e:
            # Record failure
            self.metrics_service.record_failure(f"{self.service_id}.chat_stream")
            logger.exception(f"Error in {self.service_id}.chat_stream: {str(e)}")
            raise self._translate_error(e)
    
    # Abstract methods that subclasses must implement
    
    async def _generate_text_impl(self, prompt: str, model: str, params: Dict[str, Any]) -> str:
        """
        Implementation of text generation.
        
        Subclasses must override this method to provide their implementation.
        
        Args:
            prompt: The prompt to generate text from
            model: The model to use for generation
            params: Additional parameters for the generation
            
        Returns:
            Generated text as string
        """
        raise NotImplementedError("Subclasses must implement _generate_text_impl")
    
    async def _generate_stream_impl(self, prompt: str, model: str, params: Dict[str, Any]) -> AsyncIterator[str]:
        """
        Implementation of streaming text generation.
        
        Subclasses must override this method to provide their implementation.
        
        Args:
            prompt: The prompt to generate text from
            model: The model to use for generation
            params: Additional parameters for the generation
            
        Returns:
            Asynchronous iterator of generated text chunks
        """
        raise NotImplementedError("Subclasses must implement _generate_stream_impl")
    
    async def _get_embeddings_impl(self, 
                                 text: Union[str, List[str]], 
                                 model: str,
                                 params: Dict[str, Any]) -> List[List[float]]:
        """
        Implementation of embedding generation.
        
        Subclasses must override this method to provide their implementation.
        
        Args:
            text: The text or list of texts to get embeddings for
            model: The model to use for embeddings
            params: Additional parameters for the embedding generation
            
        Returns:
            List of embedding vectors
        """
        raise NotImplementedError("Subclasses must implement _get_embeddings_impl")
    
    async def _chat_impl(self, messages: List[Dict[str, str]], model: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation of chat interaction.
        
        Subclasses must override this method to provide their implementation.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model to use for the chat
            params: Additional parameters for the chat
            
        Returns:
            Dictionary containing the response
        """
        raise NotImplementedError("Subclasses must implement _chat_impl")
    
    async def _chat_stream_impl(self, messages: List[Dict[str, str]], model: str, params: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Implementation of streaming chat interaction.
        
        Subclasses must override this method to provide their implementation.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model to use for the chat
            params: Additional parameters for the chat
            
        Returns:
            Asynchronous iterator of response chunks
        """
        raise NotImplementedError("Subclasses must implement _chat_stream_impl")
    
    def _translate_error(self, error: Exception) -> Exception:
        """
        Translate provider-specific errors to service-level exceptions.
        
        Args:
            error: The provider-specific exception
            
        Returns:
            A service-level exception
        """
        # Handle TransportError from the transport layer
        if isinstance(error, TransportError):
            if isinstance(error, RateLimitExceededError):
                from asf.medical.llm_gateway.interfaces.exceptions import RateLimitException
                return RateLimitException(f"Rate limit exceeded for {self.service_id}: {str(error)}")
            elif isinstance(error, CircuitBreakerOpenError):
                return ServiceUnavailableException(f"Circuit breaker open for {self.service_id}: {str(error)}")
            else:
                return TransportException(f"Transport error for {self.service_id}: {str(error)}")
                
        # Handle timeout errors
        if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
            return TimeoutException(f"Request timed out for {self.service_id}")
            
        # If the error is already a LLMServiceException, just return it
        if isinstance(error, LLMServiceException):
            return error
            
        # Default case: wrap in a generic LLMServiceException
        return LLMServiceException(f"Service error in {self.service_id}: {str(error)}")