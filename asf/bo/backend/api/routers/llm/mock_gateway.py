"""
LLM Gateway Mock Components for Development.

This module provides mock implementations of LLM Gateway components
to allow the UI to function without the full LLM Gateway implementation.
"""

import os
import yaml
import uuid
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field

# Mock enums
class MCPRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class FinishReason(Enum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTERED = "content_filtered"
    ERROR = "error"

# Mock dataclasses
@dataclass
class ContentItem:
    role: str
    content: str

@dataclass
class UsageInfo:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

@dataclass
class LLMConfig:
    model_identifier: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None

@dataclass
class InterventionContext:
    session_id: str
    conversation: List[ContentItem] = field(default_factory=list)
    
    def add_conversation_turn(self, role: str, content: str) -> None:
        self.conversation.append(ContentItem(role=role, content=content))

@dataclass
class LLMRequest:
    prompt_content: str
    config: LLMConfig
    initial_context: InterventionContext
    stream: bool = False

@dataclass
class LLMResponse:
    request_id: str
    generated_content: str
    finish_reason: FinishReason
    usage: Optional[UsageInfo] = None

@dataclass
class ProviderConfig:
    provider_type: str
    display_name: str
    connection_params: Dict[str, Any]
    models: Dict[str, Dict[str, Any]]

@dataclass
class GatewayConfig:
    gateway_id: str
    default_provider: str
    allowed_providers: List[str]
    additional_config: Dict[str, Any] = field(default_factory=dict)

# Mock provider factory
class ProviderFactory:
    def __init__(self):
        pass
    
    def create_provider(self, provider_type: str, config: Dict[str, Any]) -> Any:
        return MockProvider(provider_type, config)

# Mock provider
class MockProvider:
    def __init__(self, provider_type: str, config: Dict[str, Any]):
        self.provider_type = provider_type
        self.config = config
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            request_id=str(uuid.uuid4()),
            generated_content=f"Mock response from {self.provider_type}",
            finish_reason=FinishReason.STOP,
            usage=UsageInfo(
                prompt_tokens=len(request.prompt_content.split()),
                completion_tokens=10,
                total_tokens=len(request.prompt_content.split()) + 10
            )
        )

# Mock LLM Gateway client
class LLMGatewayClient:
    def __init__(self, config: GatewayConfig, provider_factory: ProviderFactory):
        self.config = config
        self.provider_factory = provider_factory
        self.providers = {}
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            request_id=str(uuid.uuid4()),
            generated_content="This is a mock response from the LLM Gateway. The actual functionality requires the full LLM Gateway implementation.",
            finish_reason=FinishReason.STOP,
            usage=UsageInfo(
                prompt_tokens=len(request.prompt_content.split()),
                completion_tokens=20,
                total_tokens=len(request.prompt_content.split()) + 20
            )
        )