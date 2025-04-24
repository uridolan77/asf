"""
LLM API routers package for BO backend.

This package contains routers for LLM-related functionality, including:
- LLM Gateway management
- LLM Service Abstraction Layer management
- DSPy integration
- BiomedLM integration
- CL-PEFT (Continual Learning with Parameter-Efficient Fine-Tuning)
- MCP (Model Context Protocol) integration
"""

from .gateway import router as gateway_router
from .service import router as service_router
from .dspy import router as dspy_router
from .biomedlm import router as biomedlm_router
from .main import router as llm_router
from .debug import router as debug_router
from .cl_peft import router as cl_peft_router
from .mcp import router as mcp_router

__all__ = ['gateway_router', 'service_router', 'dspy_router', 'biomedlm_router', 'llm_router', 'debug_router', 'cl_peft_router', 'mcp_router']
