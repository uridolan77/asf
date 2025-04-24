"""
OpenAI provider package for the LLM Gateway.

This package provides integration with OpenAI's API for the LLM Gateway.
"""

from asf.medical.llm_gateway.providers.openai.client import OpenAIClient

# Export the OpenAIClient class
__all__ = ["OpenAIClient"]