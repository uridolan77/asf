"""
Abstract interface for LLM services.

This module defines the contract that all LLM service implementations must follow,
providing a consistent interface for text generation, embedding, and chat capabilities.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional, Union

class LLMServiceInterface(ABC):
    """
    Abstract interface for LLM services.
    
    This interface defines the contract that all LLM service implementations must follow.
    It provides methods for text generation, embedding generation, and chat capabilities.
    """
    
    @abstractmethod
    async def generate_text(self, 
                           prompt: str, 
                           model: str, 
                           params: Dict[str, Any]) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            model: The model to use for generation
            params: Additional parameters for the generation
            
        Returns:
            Generated text as string
        """
        pass
    
    @abstractmethod
    async def generate_stream(self, 
                             prompt: str, 
                             model: str, 
                             params: Dict[str, Any]) -> AsyncIterator[str]:
        """
        Stream text generation from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            model: The model to use for generation
            params: Additional parameters for the generation
            
        Returns:
            Asynchronous iterator of generated text chunks
        """
        pass
        
    @abstractmethod
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
        """
        pass
    
    @abstractmethod
    async def chat(self,
                  messages: List[Dict[str, str]],
                  model: str,
                  params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Have a chat-based interaction with the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model to use for the chat
            params: Additional parameters for the chat
            
        Returns:
            Dictionary containing the response
        """
        pass
    
    @abstractmethod
    async def chat_stream(self,
                         messages: List[Dict[str, str]],
                         model: str,
                         params: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Have a streaming chat-based interaction with the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model to use for the chat
            params: Additional parameters for the chat
            
        Returns:
            Asynchronous iterator of response chunks
        """
        pass