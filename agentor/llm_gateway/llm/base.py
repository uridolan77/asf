from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel


class LLMRequest(BaseModel):
    """Request to an LLM."""
    prompt: str
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMResponse(BaseModel):
    """Response from an LLM."""
    text: str
    model: str
    usage: Dict[str, int]
    metadata: Optional[Dict[str, Any]] = None


class BaseLLM(ABC):
    """Base class for all LLM providers."""
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            request: The request to send to the LLM
            
        Returns:
            The response from the LLM
        """
        pass
