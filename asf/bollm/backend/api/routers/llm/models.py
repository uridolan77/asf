"""
Shared data models for LLM API routers.

This module provides shared Pydantic models for LLM-related API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Gateway Models
class ProviderStatus(BaseModel):
    provider_id: str
    status: str
    provider_type: str
    display_name: Optional[str] = None
    models: List[str] = []
    checked_at: str
    message: Optional[str] = None

class GatewayStatus(BaseModel):
    gateway_id: str
    status: str
    version: str
    default_provider: str
    active_providers: List[ProviderStatus]
    config_path: str
    checked_at: str

class ProviderUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    connection_params: Optional[Dict[str, Any]] = None
    models: Optional[Dict[str, Dict[str, Any]]] = None
    enabled: Optional[bool] = None

class LLMRequestModel(BaseModel):
    prompt: str
    model: str
    provider_id: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    system_prompt: Optional[str] = None
    stream: bool = False

class LLMResponseModel(BaseModel):
    request_id: str
    content: str
    model: str
    provider_id: str
    finish_reason: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    created_at: str

# DSPy Models
class DSPyModuleInfo(BaseModel):
    name: str
    description: Optional[str] = None
    signature: str
    parameters: Dict[str, Any] = {}
    registered_at: str
    last_used: Optional[str] = None
    usage_count: int = 0

class DSPyExecuteRequest(BaseModel):
    module_name: str
    inputs: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None

class DSPyOptimizeRequest(BaseModel):
    module_name: str
    metric: str
    num_trials: int = 10
    examples: List[Dict[str, Any]] = []
    config: Optional[Dict[str, Any]] = None

class DSPyExecuteResponse(BaseModel):
    module_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    execution_time_ms: float
    model_used: str
    tokens_used: Optional[int] = None
    created_at: str

# BiomedLM Models
class BiomedLMModelInfo(BaseModel):
    model_id: str
    display_name: str
    description: Optional[str] = None
    base_model: str
    parameters: int
    adapter_type: Optional[str] = None
    fine_tuned_for: Optional[List[str]] = None
    created_at: str
    last_used: Optional[str] = None
    usage_count: int = 0

class BiomedLMGenerateRequest(BaseModel):
    model_id: str
    prompt: str
    max_tokens: Optional[int] = Field(None, gt=0)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, gt=0)
    repetition_penalty: Optional[float] = Field(None, ge=0.0)
    do_sample: Optional[bool] = None

class BiomedLMFinetuneRequest(BaseModel):
    model_id: str
    adapter_name: str
    task: str
    dataset: Union[str, List[Dict[str, str]]]
    learning_rate: Optional[float] = Field(None, gt=0.0)
    batch_size: Optional[int] = Field(None, gt=0)
    num_epochs: Optional[int] = Field(None, gt=0)
    lora_r: Optional[int] = Field(None, gt=0)
    lora_alpha: Optional[int] = Field(None, gt=0)
    lora_dropout: Optional[float] = Field(None, ge=0.0, le=1.0)

class BiomedLMGenerateResponse(BaseModel):
    model_id: str
    prompt: str
    generated_text: str
    generation_time_ms: float
    tokens_generated: Optional[int] = None
    created_at: str
