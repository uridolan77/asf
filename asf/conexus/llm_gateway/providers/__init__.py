"""
Providers package for the domain-agnostic LLM Gateway.

This package manages the various LLM provider implementations and
provides a factory for creating provider instances with proper
connection pooling and resilience patterns.
"""

import logging
from asf.conexus.llm_gateway.providers.provider_factory import register_provider
from asf.conexus.llm_gateway.providers.cached_provider import (
    CachedProviderWrapper, create_cached_provider
)

# Import provider implementations
from asf.conexus.llm_gateway.providers.openai import OpenAIClient

# Register provider implementations
register_provider("openai", OpenAIClient)

__all__ = [
    "register_provider", 
    "OpenAIClient",
    "CachedProviderWrapper",
    "create_cached_provider"
]

