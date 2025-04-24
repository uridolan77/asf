"""
Core LLM operations component.

This module provides the core LLM operations for the Enhanced LLM Service,
including text generation, streaming, chat, and embeddings.
"""

import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from asf.medical.llm_gateway.core.client import LLMGatewayClient
from asf.medical.llm_gateway.core.models import (
    LLMRequest, LLMResponse, LLMConfig, InterventionContext,
    MCPRole, StreamChunk, FinishReason
)
from asf.medical.llm_gateway.interfaces.exceptions import LLMServiceException

logger = logging.getLogger(__name__)

class CoreOperationsComponent:
    """
    Core LLM operations component.
    
    This class provides the core LLM operations for the Enhanced LLM Service,
    including text generation, streaming, chat, and embeddings.
    """
    
    def __init__(self, gateway_client: LLMGatewayClient):
        """
        Initialize the core operations component.
        
        Args:
            gateway_client: The LLMGatewayClient to use for LLM operations
        """
        self.gateway_client = gateway_client
    
    async def generate_text(self, prompt: str, model: str, params: Dict[str, Any]) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            model: The model to use for generation
            params: Additional parameters for the generation
            
        Returns:
            Generated text as string
        """
        # Create request ID
        request_id = str(uuid.uuid4())
        
        # Create LLM request
        llm_config = LLMConfig(
            model_identifier=model,
            temperature=params.get('temperature', 0.7),
            max_tokens=params.get('max_tokens'),
            top_p=params.get('top_p'),
            presence_penalty=params.get('presence_penalty'),
            frequency_penalty=params.get('frequency_penalty'),
            system_prompt=params.get('system_prompt')
        )
        
        # Create context
        intervention_context = InterventionContext(request_id=request_id)
        
        # Add system prompt if provided
        if params.get('system_prompt'):
            intervention_context.add_conversation_turn(
                MCPRole.SYSTEM.value,
                params.get('system_prompt')
            )
        
        # Create request
        request = LLMRequest(
            prompt_content=prompt,
            config=llm_config,
            initial_context=intervention_context
        )
        
        try:
            # Generate response
            response = await self.gateway_client.generate(request)
            
            # Extract generated content
            return response.generated_content or ""
        except Exception as e:
            # Re-raise as service exception
            raise LLMServiceException(f"Error in generate_text: {str(e)}") from e
    
    async def generate_stream(self, prompt: str, model: str, params: Dict[str, Any]) -> AsyncIterator[str]:
        """
        Stream text generation from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            model: The model to use for generation
            params: Additional parameters for the generation
            
        Returns:
            Asynchronous iterator of generated text chunks
        """
        # Create request ID
        request_id = str(uuid.uuid4())
        
        # Create LLM request
        llm_config = LLMConfig(
            model_identifier=model,
            temperature=params.get('temperature', 0.7),
            max_tokens=params.get('max_tokens'),
            top_p=params.get('top_p'),
            presence_penalty=params.get('presence_penalty'),
            frequency_penalty=params.get('frequency_penalty'),
            system_prompt=params.get('system_prompt')
        )
        
        # Create context
        intervention_context = InterventionContext(request_id=request_id)
        
        # Add system prompt if provided
        if params.get('system_prompt'):
            intervention_context.add_conversation_turn(
                MCPRole.SYSTEM.value,
                params.get('system_prompt')
            )
        
        # Create request
        request = LLMRequest(
            prompt_content=prompt,
            config=llm_config,
            initial_context=intervention_context,
            stream=True
        )
        
        try:
            # Generate stream
            async for chunk in self.gateway_client.generate_stream(request):
                # Extract chunk text
                chunk_text = chunk.delta_text or ""
                
                # Yield the chunk text
                yield chunk_text
        except Exception as e:
            # Re-raise as service exception
            raise LLMServiceException(f"Error in generate_stream: {str(e)}") from e
    
    async def chat(self, messages: List[Dict[str, str]], model: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Have a chat-based interaction with the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model to use for the chat
            params: Additional parameters for the chat
            
        Returns:
            Dictionary containing the response
        """
        # Create request ID
        request_id = str(uuid.uuid4())
        
        # Create LLM request
        llm_config = LLMConfig(
            model_identifier=model,
            temperature=params.get('temperature', 0.7),
            max_tokens=params.get('max_tokens'),
            top_p=params.get('top_p'),
            presence_penalty=params.get('presence_penalty'),
            frequency_penalty=params.get('frequency_penalty'),
            system_prompt=params.get('system_prompt')
        )
        
        # Create context
        intervention_context = InterventionContext(request_id=request_id)
        
        # Add messages to context
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            # Map role to MCPRole
            mcp_role = MCPRole.USER
            if role.lower() == 'system':
                mcp_role = MCPRole.SYSTEM
            elif role.lower() == 'assistant':
                mcp_role = MCPRole.ASSISTANT
            elif role.lower() == 'tool':
                mcp_role = MCPRole.TOOL
            
            intervention_context.add_conversation_turn(mcp_role.value, content)
        
        # Get the last user message as the prompt
        user_messages = [m for m in messages if m.get('role', '').lower() == 'user']
        prompt = user_messages[-1].get('content', '') if user_messages else ""
        
        # Create request
        request = LLMRequest(
            prompt_content=prompt,
            config=llm_config,
            initial_context=intervention_context
        )
        
        try:
            # Generate response
            response = await self.gateway_client.generate(request)
            
            # Create response dictionary
            result = {
                "id": response.request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response.generated_content
                        },
                        "finish_reason": response.finish_reason.value if response.finish_reason else "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }
            
            return result
        except Exception as e:
            # Re-raise as service exception
            raise LLMServiceException(f"Error in chat: {str(e)}") from e
    
    async def chat_stream(self, messages: List[Dict[str, str]], model: str, params: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Have a streaming chat-based interaction with the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model to use for the chat
            params: Additional parameters for the chat
            
        Returns:
            Asynchronous iterator of response chunks
        """
        # Create request ID
        request_id = str(uuid.uuid4())
        
        # Create LLM request
        llm_config = LLMConfig(
            model_identifier=model,
            temperature=params.get('temperature', 0.7),
            max_tokens=params.get('max_tokens'),
            top_p=params.get('top_p'),
            presence_penalty=params.get('presence_penalty'),
            frequency_penalty=params.get('frequency_penalty'),
            system_prompt=params.get('system_prompt')
        )
        
        # Create context
        intervention_context = InterventionContext(request_id=request_id)
        
        # Add messages to context
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            # Map role to MCPRole
            mcp_role = MCPRole.USER
            if role.lower() == 'system':
                mcp_role = MCPRole.SYSTEM
            elif role.lower() == 'assistant':
                mcp_role = MCPRole.ASSISTANT
            elif role.lower() == 'tool':
                mcp_role = MCPRole.TOOL
            
            intervention_context.add_conversation_turn(mcp_role.value, content)
        
        # Get the last user message as the prompt
        user_messages = [m for m in messages if m.get('role', '').lower() == 'user']
        prompt = user_messages[-1].get('content', '') if user_messages else ""
        
        # Create request
        request = LLMRequest(
            prompt_content=prompt,
            config=llm_config,
            initial_context=intervention_context,
            stream=True
        )
        
        try:
            # Generate stream
            chunk_count = 0
            
            async for chunk in self.gateway_client.generate_stream(request):
                chunk_count += 1
                
                # Extract chunk text
                chunk_text = chunk.delta_text or ""
                
                # Create chunk response
                chunk_response = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant" if chunk_count == 1 else None,
                                "content": chunk_text
                            },
                            "finish_reason": chunk.finish_reason.value if chunk.finish_reason else None
                        }
                    ]
                }
                
                # Yield the chunk response
                yield chunk_response
        except Exception as e:
            # Re-raise as service exception
            raise LLMServiceException(f"Error in chat_stream: {str(e)}") from e
    
    async def get_embeddings(self, text: Union[str, List[str]], model: str, params: Optional[Dict[str, Any]] = None) -> List[List[float]]:
        """
        Get embeddings for text.
        
        Args:
            text: The text or list of texts to get embeddings for
            model: The model to use for embeddings
            params: Additional parameters for the embedding generation
            
        Returns:
            List of embedding vectors
        """
        # TODO: Implement embeddings using the gateway client
        # For now, return a placeholder implementation
        
        # Create a placeholder embedding
        if isinstance(text, str):
            # Single text embedding (placeholder)
            return [[0.0] * 1536]
        else:
            # Multiple text embeddings (placeholder)
            return [[0.0] * 1536 for _ in text]
    
    async def chat_with_tools(self,
                             messages: List[Dict[str, str]],
                             tools: List[Dict[str, Any]],
                             model: str,
                             params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Have a chat-based interaction with the model with tool calling capabilities.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: List of tool definitions
            model: The model to use for the chat
            params: Additional parameters for the chat
            
        Returns:
            Dictionary containing the response with tool calls
        """
        # TODO: Implement tool calling
        # For now, return a regular chat response
        return await self.chat(messages, model, params)
