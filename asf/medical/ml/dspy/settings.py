Enhanced DSPy Settings

This module provides enhanced settings for DSPy with more configuration options
and better validation for production use.

import os
import logging
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseSettings, SecretStr, Field, validator, root_validator

# Set up logging
logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    Supported LLM providers.
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    LLAMACPP = "llamacpp"
    COHERE = "cohere"
    WATSONX = "watsonx"


class CacheBackend(str, Enum):
    Supported cache backends.
    DISK = "disk"
    REDIS = "redis"
    NULL = "null"


class EnhancedDSPySettings(BaseSettings):
    Enhanced settings for DSPy with more configuration options.
    
    This class provides a comprehensive set of settings for configuring DSPy
    in a production environment, with a focus on medical research requirements.
    
    # LLM Provider Settings
    LLM_PROVIDER: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="LLM provider to use"
    )
    LLM_API_KEY: SecretStr = Field(
        default=SecretStr(""),
        description="API key for the LLM provider"
    )
    DEFAULT_MODEL: str = Field(
        default="gpt-3.5-turbo",
        description="Default model to use"
    )
    FALLBACK_MODEL: str = Field(
        default="gpt-3.5-turbo-instruct",
        description="Fallback model to use if the default model fails"
    )
    
    # Azure-specific settings
    AZURE_ENDPOINT: Optional[str] = Field(
        default=None,
        description="Azure OpenAI endpoint URL"
    )
    AZURE_DEPLOYMENT_NAME: Optional[str] = Field(
        default=None,
        description="Azure OpenAI deployment name"
    )
    AZURE_API_VERSION: str = Field(
        default="2023-05-15",
        description="Azure OpenAI API version"
    )
    
    # Model Parameters
    MAX_TOKENS: int = Field(
        default=1000,
        description="Maximum number of tokens to generate"
    )
    TEMPERATURE: float = Field(
        default=0.7,
        description="Temperature for sampling"
    )
    TOP_P: float = Field(
        default=1.0,
        description="Top-p sampling parameter"
    )
    
    # Caching Settings
    CACHE_BACKEND: CacheBackend = Field(
        default=CacheBackend.DISK,
        description="Cache backend to use"
    )
    CACHE_DIRECTORY: str = Field(
        default=".dspy_cache",
        description="Directory for disk cache"
    )
    CACHE_TTL: int = Field(
        default=3600,
        description="Default cache TTL in seconds"
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    REDIS_PASSWORD: Optional[SecretStr] = Field(
        default=None,
        description="Redis password"
    )
    REDIS_POOL_SIZE: int = Field(
        default=10,
        description="Redis connection pool size"
    )
    
    # Telemetry Settings
    ENABLE_TELEMETRY: bool = Field(
        default=True,
        description="Whether to enable telemetry"
    )
    LOG_ASYNC: bool = Field(
        default=True,
        description="Whether to log asynchronously"
    )
    MLFLOW_TRACKING_URI: Optional[str] = Field(
        default=None,
        description="MLflow tracking URI"
    )
    MLFLOW_EXPERIMENT_NAME: str = Field(
        default="dspy-experiments",
        description="MLflow experiment name"
    )
    
    # Retry Settings
    MAX_RETRIES: int = Field(
        default=3,
        description="Maximum number of retries for API calls"
    )
    RETRY_MIN_WAIT: float = Field(
        default=1.0,
        description="Minimum wait time between retries in seconds"
    )
    RETRY_MAX_WAIT: float = Field(
        default=10.0,
        description="Maximum wait time between retries in seconds"
    )
    
    # Thread Settings
    THREAD_LIMIT: int = Field(
        default=4,
        description="Maximum number of threads to use"
    )
    
    # Circuit Breaker Settings
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(
        default=5,
        description="Number of failures before opening circuit"
    )
    CIRCUIT_BREAKER_RESET_TIMEOUT: float = Field(
        default=30.0,
        description="Seconds before attempting recovery"
    )
    CIRCUIT_BREAKER_SUCCESS_THRESHOLD: int = Field(
        default=2,
        description="Successes needed in half-open state to close"
    )
    
    # Audit Logging Settings
    ENABLE_AUDIT_LOGGING: bool = Field(
        default=True,
        description="Whether to enable audit logging"
    )
    AUDIT_LOG_PATH: str = Field(
        default="audit_logs",
        description="Path to store audit logs"
    )
    ENABLE_PHI_DETECTION: bool = Field(
        default=True,
        description="Whether to enable PHI detection and redaction in logs"
    )
    
    # Security Settings
    ENABLE_INPUT_VALIDATION: bool = Field(
        default=True,
        description="Whether to enable input validation"
    )
    ENABLE_OUTPUT_FILTERING: bool = Field(
        default=True,
        description="Whether to enable output filtering"
    )
    MAX_PROMPT_LENGTH: int = Field(
        default=16000,
        description="Maximum prompt length in characters"
    )
    
    # Optimization Settings
    OPTIMIZATION_TIMEOUT: int = Field(
        default=1800,
        description="Timeout for optimization in seconds (30 minutes)"
    )
    
    # Validation
    @validator("THREAD_LIMIT")
    def validate_thread_limit(cls, v):
        Validate thread limit.
        
        Args:
            cls: Description of cls
            v: Description of v
        
        if v < 1:
            raise ValueError("THREAD_LIMIT must be at least 1")
        return v
    
    @validator("MAX_RETRIES")
    def validate_max_retries(cls, v):
        Validate max retries.
        
        Args:
            cls: Description of cls
            v: Description of v
        
        if v < 0:
            raise ValueError("MAX_RETRIES must be non-negative")
        return v
    
    @validator("TEMPERATURE")
    def validate_temperature(cls, v):
        Validate temperature.
        
        Args:
            cls: Description of cls
            v: Description of v
        
        if v < 0 or v > 2:
            raise ValueError("TEMPERATURE must be between 0 and 2")
        return v
    
    @validator("TOP_P")
    def validate_top_p(cls, v):
        Validate top_p.
        
        Args:
            cls: Description of cls
            v: Description of v
        
        if v <= 0 or v > 1:
            raise ValueError("TOP_P must be between 0 and 1")
        return v
    
    @root_validator
    def validate_azure_settings(cls, values):
        Validate Azure-specific settings.
        
        Args:
            cls: Description of cls
            values: Description of values
        
    global _enhanced_settings
    if _enhanced_settings is None:
        _enhanced_settings = EnhancedDSPySettings()
    return _enhanced_settings


# Export all classes and functions
__all__ = [
    'LLMProvider',
    'CacheBackend',
    'EnhancedDSPySettings',
    'get_enhanced_settings'
]
