"""
MCP service implementation of the LLM Service Interface.

This module provides an implementation of the LLM Service Interface for MCP,
adapting the existing MCP provider to the new service abstraction layer.
"""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import asyncio

from asf.medical.llm_gateway.services.base_service import BaseService
from asf.medical.llm_gateway.interfaces.exceptions import (
    ModelNotAvailableException,
    ServiceUnavailableException,
    AuthenticationException,
    InvalidRequestException,
    RateLimitException,
    ContextLengthException,
    ContentFilterException
)
from asf.medical.llm_gateway.mcp.mcp_provider import MCPProvider
from asf.medical.llm_gateway.mcp.errors import MCPError
from asf.medical.llm_gateway.core.models import (
    LLMRequest,
    LLMResponse,
    ContentItem,
    ProviderParameters,
    ChatMessage
)
from asf.medical.llm_gateway.observability.metrics import MetricsService

logger = logging.getLogger(__name__)


class MCPService(BaseService):
    """
    MCP service implementation of the LLM Service Interface.
    
    This class adapts the existing MCP provider to conform to the LLM service interface,
    mapping between the service abstraction and provider-specific implementation.
    """
    
    def __init__(self, config: Dict[str, Any], metrics_service: Optional[MetricsService] = None):
        """
        Initialize the MCP service with configuration.
        
        Args:
            config: Configuration dictionary for the service
            metrics_service: Optional metrics service for instrumentation
        """
        super().__init__(config, metrics_service)
        
        # Create MCP provider with configuration
        mcp_config = config.get('mcp', {})
        self.provider = MCPProvider(**mcp_config)
        
        # Get model information from configuration
        self.model_info = config.get('models', {})
        
    async def _generate_text_impl(self, prompt: str, model: str, params: Dict[str, Any]) -> str:
        """
        Implementation of text generation using MCP provider.
        
        Args:
            prompt: The prompt to generate text from
            model: The model to use for generation
            params: Additional parameters for the generation
            
        Returns:
            Generated text as string
        """
        # Create an LLMRequest from the parameters
        request = self._create_llm_request(prompt, model, params)
        
        # Call the provider's generate method
        response = await self.provider.generate(request)
        
        # Extract and return the text from the response
        return self._extract_text(response)
    
    async def _generate_stream_impl(self, prompt: str, model: str, params: Dict[str, Any]) -> AsyncIterator[str]:
        """
        Implementation of streaming text generation using MCP provider.
        
        Args:
            prompt: The prompt to generate text from
            model: The model to use for generation
            params: Additional parameters for the generation
            
        Returns:
            Asynchronous iterator of generated text chunks
        """
        # Create an LLMRequest from the parameters
        request = self._create_llm_request(prompt, model, params)
        
        # Call the provider's generate_stream method
        async for response in self.provider.generate_stream(request):
            # Extract and yield each text chunk
            yield self._extract_text(response)
    
    async def _get_embeddings_impl(self, 
                                 text: Union[str, List[str]], 
                                 model: str,
                                 params: Dict[str, Any]) -> List[List[float]]:
        """
        Implementation of embedding generation using MCP provider.
        
        Args:
            text: The text or list of texts to get embeddings for
            model: The model to use for embeddings
            params: Additional parameters for the embedding generation
            
        Returns:
            List of embedding vectors
        """
        # Create an LLMRequest for embeddings
        request = self._create_embedding_request(text, model, params)
        
        # Call the provider's get_embeddings method
        response = await self.provider.get_embeddings(request)
        
        # Return the embedding vectors
        return response.embeddings
    
    async def _chat_impl(self, messages: List[Dict[str, str]], model: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation of chat interaction using MCP provider.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model to use for the chat
            params: Additional parameters for the chat
            
        Returns:
            Dictionary containing the response
        """
        # Create an LLMRequest for chat
        request = self._create_chat_request(messages, model, params)
        
        # Call the provider's chat method
        response = await self.provider.chat(request)
        
        # Format and return the chat response
        return self._format_chat_response(response)
    
    async def _chat_stream_impl(self, messages: List[Dict[str, str]], model: str, params: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Implementation of streaming chat interaction using MCP provider.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model to use for the chat
            params: Additional parameters for the chat
            
        Returns:
            Asynchronous iterator of response chunks
        """
        # Create an LLMRequest for chat
        request = self._create_chat_request(messages, model, params)
        
        # Call the provider's chat_stream method
        async for response in self.provider.chat_stream(request):
            # Format and yield each chat chunk
            yield self._format_chat_response(response)
    
    def _create_llm_request(self, prompt: str, model: str, params: Dict[str, Any]) -> LLMRequest:
        """
        Create an LLMRequest for text generation.
        
        Args:
            prompt: The prompt to generate text from
            model: The model to use for generation
            params: Additional parameters for the generation
            
        Returns:
            An LLMRequest object
        """
        # Create ProviderParameters from the params dictionary
        provider_parameters = ProviderParameters(
            temperature=params.get('temperature', 0.7),
            top_p=params.get('top_p', 1.0),
            max_tokens=params.get('max_tokens', 1024),
            stop=params.get('stop', None),
            frequency_penalty=params.get('frequency_penalty', 0.0),
            presence_penalty=params.get('presence_penalty', 0.0),
        )
        
        # Create and return the LLMRequest
        return LLMRequest(
            model=model,
            provider_id="mcp",
            provider_parameters=provider_parameters,
            content_items=[ContentItem(content=prompt, content_type="text")]
        )
    
    def _create_embedding_request(self, text: Union[str, List[str]], model: str, params: Dict[str, Any]) -> LLMRequest:
        """
        Create an LLMRequest for embedding generation.
        
        Args:
            text: The text or list of texts to get embeddings for
            model: The model to use for embeddings
            params: Additional parameters for the embedding generation
            
        Returns:
            An LLMRequest object
        """
        # Create ProviderParameters from the params dictionary
        provider_parameters = ProviderParameters(**params)
        
        # Create content items from the input text
        if isinstance(text, str):
            content_items = [ContentItem(content=text, content_type="text")]
        else:
            content_items = [ContentItem(content=t, content_type="text") for t in text]
        
        # Create and return the LLMRequest
        return LLMRequest(
            model=model,
            provider_id="mcp",
            provider_parameters=provider_parameters,
            content_items=content_items
        )
    
    def _create_chat_request(self, messages: List[Dict[str, str]], model: str, params: Dict[str, Any]) -> LLMRequest:
        """
        Create an LLMRequest for chat interaction.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model to use for the chat
            params: Additional parameters for the chat
            
        Returns:
            An LLMRequest object
        """
        # Create ProviderParameters from the params dictionary
        provider_parameters = ProviderParameters(
            temperature=params.get('temperature', 0.7),
            top_p=params.get('top_p', 1.0),
            max_tokens=params.get('max_tokens', 1024),
            stop=params.get('stop', None),
            frequency_penalty=params.get('frequency_penalty', 0.0),
            presence_penalty=params.get('presence_penalty', 0.0),
        )
        
        # Convert message dictionaries to ChatMessage objects
        chat_messages = [
            ChatMessage(role=msg['role'], content=msg['content'])
            for msg in messages
        ]
        
        # Create and return the LLMRequest
        return LLMRequest(
            model=model,
            provider_id="mcp",
            provider_parameters=provider_parameters,
            chat_messages=chat_messages
        )
    
    def _extract_text(self, response: LLMResponse) -> str:
        """
        Extract text from an LLMResponse.
        
        Args:
            response: The LLMResponse to extract text from
            
        Returns:
            The extracted text as a string
        """
        if response.content_items and len(response.content_items) > 0:
            return response.content_items[0].content
        return ""
    
    def _format_chat_response(self, response: LLMResponse) -> Dict[str, Any]:
        """
        Format a chat response.
        
        Args:
            response: The LLMResponse to format
            
        Returns:
            A dictionary containing the formatted response
        """
        if response.chat_message:
            return {
                'role': response.chat_message.role,
                'content': response.chat_message.content
            }
        elif response.content_items and len(response.content_items) > 0:
            return {
                'role': 'assistant',
                'content': response.content_items[0].content
            }
        return {
            'role': 'assistant',
            'content': ''
        }
    
    def _translate_error(self, error: Exception) -> Exception:
        """
        Translate MCP-specific errors to service-level exceptions.
        
        Args:
            error: The MCP-specific exception
            
        Returns:
            A service-level exception
        """
        # Handle MCP-specific errors
        if isinstance(error, MCPError):
            if error.error_type == "model_not_available":
                return ModelNotAvailableException(f"Model not available: {str(error)}")
            elif error.error_type == "service_unavailable":
                return ServiceUnavailableException(f"MCP service unavailable: {str(error)}")
            elif error.error_type == "authentication_failed":
                return AuthenticationException(f"MCP authentication failed: {str(error)}")
            elif error.error_type == "invalid_request":
                return InvalidRequestException(f"Invalid MCP request: {str(error)}")
            elif error.error_type == "rate_limit_exceeded":
                return RateLimitException(f"MCP rate limit exceeded: {str(error)}")
            elif error.error_type == "context_length_exceeded":
                return ContextLengthException(f"MCP context length exceeded: {str(error)}")
            elif error.error_type == "content_filtered":
                return ContentFilterException(f"Content filtered by MCP: {str(error)}")
        
        # Fall back to parent class error translation
        return super()._translate_error(error)