"""
LLM API routers package for BO backend.

This package contains routers for LLM-related functionality, including:
- LLM Gateway management
- DSPy integration
- BiomedLM integration
- CL-PEFT (Continual Learning with Parameter-Efficient Fine-Tuning)
- MCP (Model Context Protocol) integration
- Progress tracking for LLM operations
"""

from .gateway import router as gateway_router
from .dspy import router as dspy_router
from .biomedlm import router as biomedlm_router
from .main import router as llm_router
from .debug import router as debug_router
from .cl_peft import router as cl_peft_router
from .mcp import router as mcp_router
from .grafana import router as grafana_router
from .progress import router as progress_router
from .enhanced_gateway import router as enhanced_gateway_router

__all__ = ['gateway_router', 'dspy_router', 'biomedlm_router', 'llm_router', 'debug_router', 'cl_peft_router', 'mcp_router', 'grafana_router', 'progress_router', 'enhanced_gateway_router']
