"""
DSPy Settings Module

This module provides configuration settings for the DSPy integration using Pydantic.
It handles loading settings from environment variables and provides validation.
"""

import os
import secrets
from enum import Enum
from typing import Optional, List, Dict, Any, Union

from pydantic import BaseSettings, SecretStr, Field, validator, root_validator
from pydantic.env_settings import SettingsSourceCallable


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    LLAMA_CPP = "llama_cpp"
    COHERE = "cohere"


class CacheBackend(str, Enum):
    """Supported caching backends."""
    NONE = "none"
    DISK = "disk"
    REDIS = "redis"


class DSPySettings(BaseSettings):
    """
    Settings for DSPy integration.
    
    This class uses Pydantic to load and validate configuration from environment variables.
    """
    # LLM Provider Settings
    LLM_PROVIDER: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="The LLM provider to use"
    )
    LLM_API_KEY: SecretStr = Field(
        default=SecretStr(""),
        description="API key for the LLM provider"
    )
    LLM_API_BASE: Optional[str] = Field(
        default=None,
        description="Base URL for the LLM API (if not using default)"
    )
    LLM_ORGANIZATION: Optional[str] = Field(
        default=None,
        description="Organization ID for OpenAI"
    )
    DEFAULT_MODEL: str = Field(
        default="gpt-3.5-turbo",
        description="Default model to use"
    )
    FALLBACK_MODEL: Optional[str] = Field(
        default=None,
        description="Fallback model to use if the default model fails"
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
    PRESENCE_PENALTY: float = Field(
        default=0.0,
        description="Presence penalty parameter"
    )
    FREQUENCY_PENALTY: float = Field(
        default=0.0,
        description="Frequency penalty parameter"
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
        description="Time-to-live for cache entries in seconds"
    )
    REDIS_URL: Optional[str] = Field(
        default=None,
        description="Redis URL for Redis cache"
    )
    REDIS_PASSWORD: Optional[SecretStr] = Field(
        default=None,
        description="Redis password for Redis cache"
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
    
    # Medical Research Specific Settings
    MEDICAL_DATA_SOURCES: List[str] = Field(
        default=[],
        description="List of medical data sources to use"
    )
    ENABLE_PHI_DETECTION: bool = Field(
        default=True,
        description="Whether to enable PHI detection and redaction"
    )
    AUDIT_LOGGING: bool = Field(
        default=True,
        description="Whether to enable detailed audit logging"
    )
    AUDIT_LOG_PATH: str = Field(
        default="audit_logs",
        description="Path for audit logs"
    )
    
    # Validation
    @validator("MAX_TOKENS")
    def validate_max_tokens(cls, v):
        """Validate max_tokens is positive."""
        if v <= 0:
            raise ValueError("MAX_TOKENS must be positive")
        return v
    
    @validator("TEMPERATURE")
    def validate_temperature(cls, v):
        """Validate temperature is between 0 and 2."""
        if v < 0 or v > 2:
            raise ValueError("TEMPERATURE must be between 0 and 2")
        return v
    
    @validator("TOP_P")
    def validate_top_p(cls, v):
        """Validate top_p is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("TOP_P must be between 0 and 1")
        return v
    
    @validator("PRESENCE_PENALTY", "FREQUENCY_PENALTY")
    def validate_penalty(cls, v):
        """Validate penalties are between -2 and 2."""
        if v < -2 or v > 2:
            raise ValueError("Penalty must be between -2 and 2")
        return v
    
    @root_validator
    def validate_redis_settings(cls, values):
        """Validate Redis settings."""
        if values.get("CACHE_BACKEND") == CacheBackend.REDIS:
            if not values.get("REDIS_URL"):
                raise ValueError("REDIS_URL must be provided when using Redis cache")
        return values
    
    @root_validator
    def validate_llm_api_key(cls, values):
        """Validate LLM API key is provided."""
        if not values.get("LLM_API_KEY") or values.get("LLM_API_KEY").get_secret_value() == "":
            raise ValueError("LLM_API_KEY must be provided")
        return values
    
    class Config:
        """Pydantic config."""
        env_prefix = "DSPY_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create a singleton instance
dspy_settings = DSPySettings()


def get_dspy_settings() -> DSPySettings:
    """
    Get the DSPy settings.
    
    Returns:
        DSPySettings: The DSPy settings.
    """
    return dspy_settings
