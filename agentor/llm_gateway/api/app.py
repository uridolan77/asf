from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from agentor.llm_gateway.api.auth import validate_api_key, get_settings, Settings, RBACMiddleware
from agentor.llm_gateway.api.middleware import InputValidationMiddleware
from agentor.llm_gateway.llm.base import LLMRequest, LLMResponse
from agentor.llm_gateway.utils.tracing import TracingMiddleware, setup_tracing


# Configure the application
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure the application on startup and shutdown."""
    # Startup
    settings = get_settings()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add input validation middleware
    app.add_middleware(InputValidationMiddleware)

    # Add RBAC middleware
    app.add_middleware(RBACMiddleware)

    # Set up tracing
    tracer = setup_tracing("llm_gateway")

    # Add tracing middleware
    app.add_middleware(TracingMiddleware, tracer=tracer)

    yield

    # Shutdown
    # Clean up resources here

# Create the app with lifespan
app = FastAPI(title="LLM Gateway API", lifespan=lifespan)


class GenerateRequest(BaseModel):
    """Request to generate text from an LLM."""
    prompt: str
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stop_sequences: Optional[list[str]] = None


class GenerateResponse(BaseModel):
    """Response from generating text from an LLM."""
    text: str
    model: str
    usage: Dict[str, int]


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    api_key: str = Depends(validate_api_key),
    settings: Settings = Depends(get_settings)
):
    """Generate text from an LLM.

    Args:
        request: The generation request
        api_key: The validated API key
        settings: The application settings

    Returns:
        The generated text
    """
    # In a real implementation, we would use the LLM gateway to generate text
    # For this example, we'll just return a dummy response
    return GenerateResponse(
        text="This is a dummy response.",
        model=request.model,
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
