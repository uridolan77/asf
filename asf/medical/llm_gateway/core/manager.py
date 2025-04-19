# llm_gateway/core/manager.py

import asyncio
import logging
import time
from typing import Dict, List, Optional, Type, AsyncGenerator, Any, Union, AsyncIterator

from asf.medical.llm_gateway.core.models import (
    InterventionContext, LLMRequest, LLMResponse, StreamChunk,
    PerformanceMetrics, ComplianceResult, FinishReason
)
from asf.medical.llm_gateway.core.factory import ProviderFactory
from asf.medical.llm_gateway.core.models import (
    ContentItem,
    ErrorDetails,
    ErrorLevel,
    FinishReason,
    GatewayConfig,
    InterventionContext,
    LLMConfig,
    LLMRequest,
    LLMResponse,
    MCPContentType as GatewayMCPContentType,
    MCPMetadata,
    MCPRole as GatewayMCPRole,
    PerformanceMetrics,
    ProviderConfig,
    StreamChunk,
    ToolFunction,
    ToolDefinition,
    ToolUseRequest,
    UsageStats,
    MCPUsage
)
from asf.medical.llm_gateway.interventions.base import BaseIntervention
from asf.medical.llm_gateway.interventions.factory import InterventionFactory, InterventionFactoryError
from asf.medical.llm_gateway.interventions.manager import InterventionManager
from asf.medical.llm_gateway.providers.base import BaseProvider
from asf.medical.llm_gateway.services.factory import ServiceFactory
from asf.medical.llm_gateway.interfaces.exceptions import LLMServiceException

logger = logging.getLogger(__name__)

class LLMGatewayManager:
    """
    Main orchestration layer for the LLM Gateway.
    This class is responsible for routing requests to the appropriate service.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM Gateway Manager with configuration.
        
        Args:
            config: Configuration dictionary for the manager
        """
        self.config = config
        self.service_factory = ServiceFactory(config)
    
    async def generate_text(self, prompt: str, model: str, params: Optional[Dict[str, Any]] = None) -> str:
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
        if params is None:
            params = {}
            
        try:
            service = self.service_factory.get_service_for_model(model)
            return await service.generate_text(prompt, model, params)
        except Exception as e:
            if isinstance(e, LLMServiceException):
                raise
            logger.error(f"Error during text generation: {str(e)}")
            raise LLMServiceException(f"Failed to generate text: {str(e)}")
    
    async def generate_stream(self, prompt: str, model: str, params: Optional[Dict[str, Any]] = None) -> AsyncIterator[str]:
        """
        Stream text generation from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            model: The model to use for generation
            params: Additional parameters for the generation
            
        Returns:
            Asynchronous iterator of generated text chunks
            
        Raises:
            LLMServiceException: If an error occurs during text generation
        """
        if params is None:
            params = {}
            
        try:
            service = self.service_factory.get_service_for_model(model)
            async for chunk in service.generate_stream(prompt, model, params):
                yield chunk
        except Exception as e:
            if isinstance(e, LLMServiceException):
                raise
            logger.error(f"Error during text generation stream: {str(e)}")
            raise LLMServiceException(f"Failed to stream text: {str(e)}")
            
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
            service = self.service_factory.get_service_for_model(model)
            return await service.get_embeddings(text, model, params)
        except Exception as e:
            if isinstance(e, LLMServiceException):
                raise
            logger.error(f"Error during embedding generation: {str(e)}")
            raise LLMServiceException(f"Failed to generate embeddings: {str(e)}")
    
    async def chat(self,
                  messages: List[Dict[str, str]],
                  model: str,
                  params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        if params is None:
            params = {}
            
        try:
            service = self.service_factory.get_service_for_model(model)
            return await service.chat(messages, model, params)
        except Exception as e:
            if isinstance(e, LLMServiceException):
                raise
            logger.error(f"Error during chat: {str(e)}")
            raise LLMServiceException(f"Failed to chat: {str(e)}")
    
    async def chat_stream(self,
                         messages: List[Dict[str, str]],
                         model: str,
                         params: Optional[Dict[str, Any]] = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Have a streaming chat-based interaction with the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model to use for the chat
            params: Additional parameters for the chat
            
        Returns:
            Asynchronous iterator of response chunks
            
        Raises:
            LLMServiceException: If an error occurs during chat
        """
        if params is None:
            params = {}
            
        try:
            service = self.service_factory.get_service_for_model(model)
            async for chunk in service.chat_stream(messages, model, params):
                yield chunk
        except Exception as e:
            if isinstance(e, LLMServiceException):
                raise
            logger.error(f"Error during chat stream: {str(e)}")
            raise LLMServiceException(f"Failed to stream chat: {str(e)}")

