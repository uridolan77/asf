"""
Mock provider module for LLM Gateway.

This module re-exports the MockClient as MockProvider for backward compatibility.
"""

from asf.medical.llm_gateway.providers.mock_client import MockClient as MockProvider

__all__ = ["MockProvider"]
