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
from asf.medical.llm_gateway.mcp.errors import McpError
from asf.medical.llm_gateway.core.models import (
    LLMRequest,
    LLMResponse,
    ContentItem,
    LLMConfig,
    Message,
    MCPRole
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
        # Create LLMConfig from the params dictionary
        config = LLMConfig(
            model_identifier=model,
            temperature=params.get('temperature', 0.7),
            top_p=params.get('top_p', 1.0),
            max_tokens=params.get('max_tokens', 1024),
            stop_sequences=params.get('stop', None),
            frequency_penalty=params.get('frequency_penalty', 0.0),
            presence_penalty=params.get('presence_penalty', 0.0),
        )
        
        # Create content item from the prompt
        content_item = ContentItem.from_text(prompt)
        
        # Create and return the LLMRequest
        return LLMRequest(
            prompt_content=[content_item],
            config=config
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
        # Create LLMConfig from the params dictionary
        config = LLMConfig(
            model_identifier=model,
            **params  # Pass all parameters from params dictionary
        )
        
        # Create content items from the input text
        content_items = []
        if isinstance(text, str):
            content_items = [ContentItem.from_text(text)]
        else:
            content_items = [ContentItem.from_text(t) for t in text]
        
        # Create and return the LLMRequest
        return LLMRequest(
            prompt_content=content_items,
            config=config
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
        # Create LLMConfig from the params dictionary
        config = LLMConfig(
            model_identifier=model,
            temperature=params.get('temperature', 0.7),
            top_p=params.get('top_p', 1.0),
            max_tokens=params.get('max_tokens', 1024),
            stop_sequences=params.get('stop', None),
            frequency_penalty=params.get('frequency_penalty', 0.0),
            presence_penalty=params.get('presence_penalty', 0.0),
        )
        
        # Convert message dictionaries to Message objects with content items
        message_objects = []
        for msg in messages:
            role = msg['role']
            content_text = msg['content']
            # Create a content item for the message
            content_item = ContentItem.from_text(content_text)
            # Create a Message object
            message_objects.append(Message(
                role=MCPRole(role),  # Convert string role to MCPRole enum
                content=[content_item]  # Message expects a list of ContentItem
            ))
        
        # Create content items for the LLMRequest
        content_items = []
        for msg in messages:
            content_items.append(ContentItem.from_text(msg['content']))
        
        # Create and return the LLMRequest
        return LLMRequest(
            prompt_content=content_items,
            config=config
        )
    
    def _extract_text(self, response: LLMResponse) -> str:
        """
        Extract text from an LLMResponse.
        
        Args:
            response: The LLMResponse to extract text from
            
        Returns:
            The extracted text as a string
        """
        # Handle different response formats
        if hasattr(response, 'generated_content') and response.generated_content:
            if isinstance(response.generated_content, str):
                return response.generated_content
            elif isinstance(response.generated_content, list) and len(response.generated_content) > 0:
                # If generated_content is a list of ContentItem
                if hasattr(response.generated_content[0], 'text_content') and response.generated_content[0].text_content is not None:
                    return response.generated_content[0].text_content
        
        # Fallback for other response formats
        if hasattr(response, 'content_items') and response.content_items and len(response.content_items) > 0:
            # Access ContentItem properly
            if hasattr(response.content_items[0], 'text_content') and response.content_items[0].text_content is not None:
                return response.content_items[0].text_content
            elif hasattr(response.content_items[0], 'data') and 'text' in response.content_items[0].data:
                return response.content_items[0].data['text']
        
        # Default empty response
        return ""
    
    def _format_chat_response(self, response: LLMResponse) -> Dict[str, Any]:
        """
        Format a chat response.
        
        Args:
            response: The LLMResponse to format
            
        Returns:
            A dictionary containing the formatted response
        """
        # Handle different response formats
        if hasattr(response, 'generated_content') and response.generated_content:
            if isinstance(response.generated_content, str):
                return {
                    'role': 'assistant',
                    'content': response.generated_content
                }
            elif isinstance(response.generated_content, list) and len(response.generated_content) > 0:
                # Assuming generated_content is a list of ContentItem
                item = response.generated_content[0]
                if hasattr(item, 'text_content') and item.text_content is not None:
                    return {
                        'role': 'assistant',
                        'content': item.text_content
                    }
                elif hasattr(item, 'data') and 'text' in item.data:
                    return {
                        'role': 'assistant',
                        'content': item.data['text']
                    }
        
        # Default empty response
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
        if isinstance(error, McpError):
            if hasattr(error, 'error_type'):
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