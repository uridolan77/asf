"""
LLM Management API Router

This module provides FastAPI routes for managing Large Language Models (LLMs) 
including LLM Gateway, DSPy modules, and BiomedLM models.
"""

from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
from ...core.logging_config import get_logger
from ..dependencies import get_current_active_user, get_admin_user
from ...storage.models import User

logger = get_logger(__name__)

# Define the models for LLM management
class LLMProvider(BaseModel):
    """LLM provider model"""
    id: str
    name: str
    active: bool
    models: List[str]
    api_type: str
    description: Optional[str] = None

class LLMModel(BaseModel):
    """LLM model details"""
    id: str
    name: str
    provider: str
    context_length: int
    capabilities: List[str]
    description: Optional[str] = None

class LLMUsageStats(BaseModel):
    """LLM usage statistics"""
    model: str
    usage_count: int
    token_count: int
    avg_tokens_per_request: float
    avg_response_time: float
    error_rate: float
    cost_estimate: Optional[float] = None

class DSPyModule(BaseModel):
    """DSPy module model"""
    id: str
    name: str
    description: str
    task_type: str
    optimized_for: List[str]
    version: str
    created_at: str
    last_used: Optional[str] = None

class BiomedLMAdapter(BaseModel):
    """BiomedLM adapter model"""
    id: str
    name: str
    description: str
    base_model: str
    medical_domain: str
    version: str
    created_at: str
    parameters: Optional[Dict] = None

# Create the router
router = APIRouter(prefix="/api/llm", tags=["llm-management"])

@router.get("/")
async def llm_management_root():
    """
    Root endpoint for LLM management API.
    
    Returns:
        Information about available LLM management endpoints
    """
    return {
        "status": "ok", 
        "endpoints": [
            {
                "name": "gateway",
                "description": "Manage LLM Gateway providers and models",
                "routes": [
                    "/gateway/providers",
                    "/gateway/models",
                    "/gateway/test-connection"
                ]
            },
            {
                "name": "dspy",
                "description": "Manage DSPy modules for LLM programming",
                "routes": [
                    "/dspy/modules",
                    "/dspy/register",
                    "/dspy/optimize"
                ]
            },
            {
                "name": "biomedlm",
                "description": "Manage BiomedLM models and adapters",
                "routes": [
                    "/biomedlm/adapters",
                    "/biomedlm/create-adapter",
                    "/biomedlm/evaluate"
                ]
            },
            {
                "name": "usage",
                "description": "Monitor LLM usage statistics",
                "routes": [
                    "/usage/stats",
                    "/usage/by-model",
                    "/usage/by-date",
                    "/usage/by-application"
                ]
            }
        ]
    }

@router.get("/gateway/providers")
async def get_llm_providers(current_user: User = Depends(get_current_active_user)):
    """
    Get all available LLM providers.
    
    Args:
        current_user: The authenticated user
        
    Returns:
        List of LLM providers
    """
    try:
        # In a real implementation, this would query a database or service
        providers = [
            LLMProvider(
                id="openai",
                name="OpenAI",
                active=True,
                models=["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
                api_type="openai",
                description="OpenAI's models accessible through their API"
            ),
            LLMProvider(
                id="anthropic",
                name="Anthropic",
                active=True,
                models=["claude-3-opus", "claude-3-sonnet", "claude-instant"],
                api_type="anthropic",
                description="Anthropic's Claude models"
            ),
            LLMProvider(
                id="azure",
                name="Azure OpenAI",
                active=False,
                models=["gpt-4", "gpt-3.5-turbo"],
                api_type="azure_openai",
                description="OpenAI models hosted on Azure"
            ),
            LLMProvider(
                id="huggingface",
                name="Hugging Face",
                active=True,
                models=["mistralai/Mixtral-8x7B", "meta-llama/Llama-2-70b-chat"],
                api_type="huggingface",
                description="Models accessible through Hugging Face Inference API"
            )
        ]
        return providers
    except Exception as e:
        logger.error(f"Error getting LLM providers: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving LLM providers")

@router.get("/dspy/modules")
async def get_dspy_modules(current_user: User = Depends(get_current_active_user)):
    """
    Get all available DSPy modules.
    
    Args:
        current_user: The authenticated user
        
    Returns:
        List of DSPy modules
    """
    try:
        # In a real implementation, this would query a database or service
        modules = [
            DSPyModule(
                id="med-rag-1",
                name="MedicalRAG",
                description="RAG module optimized for medical information retrieval",
                task_type="retrieval",
                optimized_for=["pubmed", "clinical_trials"],
                version="1.2.0",
                created_at="2025-03-15",
                last_used="2025-04-15"
            ),
            DSPyModule(
                id="claim-extractor-2",
                name="ClaimExtractor",
                description="Extract scientific claims from medical literature",
                task_type="extraction",
                optimized_for=["research_papers", "clinical_guidelines"],
                version="2.0.1",
                created_at="2025-02-20",
                last_used="2025-04-14"
            ),
            DSPyModule(
                id="contradiction-3",
                name="ContradictionDetector",
                description="Detect contradictions between medical claims",
                task_type="comparison",
                optimized_for=["systematic_reviews", "meta_analyses"],
                version="1.5.3",
                created_at="2025-01-10",
                last_used="2025-04-16"
            )
        ]
        return modules
    except Exception as e:
        logger.error(f"Error getting DSPy modules: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving DSPy modules")

@router.get("/biomedlm/adapters")
async def get_biomedlm_adapters(current_user: User = Depends(get_current_active_user)):
    """
    Get all available BiomedLM adapters.
    
    Args:
        current_user: The authenticated user
        
    Returns:
        List of BiomedLM adapters
    """
    try:
        # In a real implementation, this would query a database or service
        adapters = [
            BiomedLMAdapter(
                id="cardio-1",
                name="CardioAdapter",
                description="Specialized adapter for cardiology domain",
                base_model="biomedlm-2-7b",
                medical_domain="cardiology",
                version="1.0.2",
                created_at="2025-03-01",
                parameters={
                    "adapter_type": "lora",
                    "rank": 16,
                    "trained_on": "cardiology_corpus_v2"
                }
            ),
            BiomedLMAdapter(
                id="neuro-2",
                name="NeuroAdapter",
                description="Specialized adapter for neurology domain",
                base_model="biomedlm-2-7b",
                medical_domain="neurology",
                version="1.1.0",
                created_at="2025-02-15",
                parameters={
                    "adapter_type": "lora",
                    "rank": 16,
                    "trained_on": "neurology_corpus_v3"
                }
            ),
            BiomedLMAdapter(
                id="general-med-3",
                name="GeneralMedAdapter",
                description="General purpose adapter for medical domain",
                base_model="biomedlm-2-7b",
                medical_domain="general",
                version="2.0.0",
                created_at="2025-01-20",
                parameters={
                    "adapter_type": "lora",
                    "rank": 32,
                    "trained_on": "pubmed_2024_corpus"
                }
            )
        ]
        return adapters
    except Exception as e:
        logger.error(f"Error getting BiomedLM adapters: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving BiomedLM adapters")

@router.get("/usage/stats")
async def get_llm_usage_stats(current_user: User = Depends(get_current_active_user)):
    """
    Get LLM usage statistics.
    
    Args:
        current_user: The authenticated user
        
    Returns:
        List of LLM usage statistics
    """
    try:
        # In a real implementation, this would query a database or monitoring system
        stats = [
            LLMUsageStats(
                model="gpt-4o",
                usage_count=2580,
                token_count=5250000,
                avg_tokens_per_request=2035,
                avg_response_time=1.2,
                error_rate=0.015,
                cost_estimate=105.00
            ),
            LLMUsageStats(
                model="claude-3-opus",
                usage_count=1420,
                token_count=3650000,
                avg_tokens_per_request=2570,
                avg_response_time=1.5,
                error_rate=0.012,
                cost_estimate=73.00
            ),
            LLMUsageStats(
                model="biomedlm-2-7b",
                usage_count=3850,
                token_count=7250000,
                avg_tokens_per_request=1883,
                avg_response_time=0.8,
                error_rate=0.022,
                cost_estimate=36.25
            ),
            LLMUsageStats(
                model="mistralai/Mixtral-8x7B",
                usage_count=980,
                token_count=1850000,
                avg_tokens_per_request=1888,
                avg_response_time=1.1,
                error_rate=0.031,
                cost_estimate=9.25
            )
        ]
        return stats
    except Exception as e:
        logger.error(f"Error getting LLM usage stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving LLM usage statistics")

@router.post("/gateway/test-connection")
async def test_llm_connection(
    provider_id: str = Body(..., embed=True),
    model_id: str = Body(..., embed=True),
    prompt: str = Body(..., embed=True),
    current_user: User = Depends(get_current_active_user)
):
    """
    Test connection to an LLM provider with a simple prompt.
    
    Args:
        provider_id: The ID of the LLM provider to test
        model_id: The ID of the model to test
        prompt: A simple prompt to send to the LLM
        current_user: The authenticated user
        
    Returns:
        The response from the LLM
    """
    try:
        # In a real implementation, this would connect to the actual LLM API
        # For now, we return a mock response
        return {
            "status": "success",
            "provider": provider_id,
            "model": model_id,
            "prompt": prompt,
            "response": f"This is a simulated response from {model_id} via {provider_id}. Your prompt was: '{prompt}'",
            "metrics": {
                "tokens_used": len(prompt.split()) * 2,
                "response_time": 0.8
            }
        }
    except Exception as e:
        logger.error(f"Error testing LLM connection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error testing connection to {provider_id}/{model_id}")