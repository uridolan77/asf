"""
DSPy Configuration Settings

This module provides configuration settings for the DSPy module,
including language model provider settings, caching configuration,
and other parameters.
"""

import os
import enum
import functools
import logging
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseSettings, Field, root_validator, validator

# Set up logging
logger = logging.getLogger(__name__)


class LLMProvider(str, enum.Enum):
    """Supported LLM providers."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MCP = "mcp"
    AZURE_OPENAI = "azure_openai"
    HUGGING_FACE = "hugging_face"
    MOCK = "mock"


class CacheBackend(str, enum.Enum):
    """Supported cache backends."""
    
    DISK = "disk"
    REDIS = "redis"
    NONE = "none"


class DSPySettings(BaseSettings):
    """DSPy settings."""
    
    # LLM Provider settings
    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="LLM provider to use"
    )
    
    # API keys and endpoints
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    
    azure_openai_api_key: Optional[str] = Field(
        default=None,
        description="Azure OpenAI API key"
    )
    
    azure_openai_endpoint: Optional[str] = Field(
        default=None,
        description="Azure OpenAI endpoint"
    )
    
    hugging_face_api_key: Optional[str] = Field(
        default=None,
        description="Hugging Face API key"
    )
    
    mcp_endpoint: Optional[str] = Field(
        default=None,
        description="MCP endpoint"
    )
    
    # Model settings
    model_name: str = Field(
        default="gpt-4",
        description="Model name to use"
    )
    
    temperature: float = Field(
        default=0.7,
        description="Temperature for generation"
    )
    
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate"
    )
    
    # Cache settings
    cache_backend: CacheBackend = Field(
        default=CacheBackend.DISK,
        description="Cache backend to use"
    )
    
    cache_dir: str = Field(
        default="./cache",
        description="Directory to store cache files"
    )
    
    redis_host: Optional[str] = Field(
        default="localhost",
        description="Redis host"
    )
    
    redis_port: Optional[int] = Field(
        default=6379,
        description="Redis port"
    )
    
    redis_password: Optional[str] = Field(
        default=None,
        description="Redis password"
    )
    
    redis_db: Optional[int] = Field(
        default=0,
        description="Redis database"
    )
    
    # Circuit breaker settings
    circuit_breaker_max_failures: int = Field(
        default=3,
        description="Maximum number of failures before opening circuit breaker"
    )
    
    circuit_breaker_reset_timeout: int = Field(
        default=60,
        description="Time in seconds before trying to close circuit breaker"
    )
    
    # Retry settings
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for API calls"
    )
    
    retry_delay: float = Field(
        default=1.0,
        description="Delay between retries in seconds"
    )
    
    # Timeout settings
    timeout: float = Field(
        default=120.0,
        description="Timeout for API calls in seconds"
    )
    
    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Log level"
    )
    
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path"
    )
    
    # DSPy-specific settings
    trace_enabled: bool = Field(
        default=False,
        description="Whether to enable DSPy tracing"
    )
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Valid levels: {valid_levels}")
        return v.upper()
    
    @root_validator
    def validate_api_keys(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the appropriate API keys are provided."""
        provider = values.get("provider")
        
        if provider == LLMProvider.OPENAI and not values.get("openai_api_key"):
            values["openai_api_key"] = os.environ.get("OPENAI_API_KEY")
            if not values.get("openai_api_key"):
                logger.warning("No OpenAI API key provided")
        
        elif provider == LLMProvider.ANTHROPIC and not values.get("anthropic_api_key"):
            values["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY")
            if not values.get("anthropic_api_key"):
                logger.warning("No Anthropic API key provided")
        
        elif provider == LLMProvider.AZURE_OPENAI:
            if not values.get("azure_openai_api_key"):
                values["azure_openai_api_key"] = os.environ.get("AZURE_OPENAI_API_KEY")
                if not values.get("azure_openai_api_key"):
                    logger.warning("No Azure OpenAI API key provided")
            
            if not values.get("azure_openai_endpoint"):
                values["azure_openai_endpoint"] = os.environ.get("AZURE_OPENAI_ENDPOINT")
                if not values.get("azure_openai_endpoint"):
                    logger.warning("No Azure OpenAI endpoint provided")
        
        elif provider == LLMProvider.HUGGING_FACE and not values.get("hugging_face_api_key"):
            values["hugging_face_api_key"] = os.environ.get("HUGGING_FACE_API_KEY")
            if not values.get("hugging_face_api_key"):
                logger.warning("No Hugging Face API key provided")
        
        elif provider == LLMProvider.MCP and not values.get("mcp_endpoint"):
            values["mcp_endpoint"] = os.environ.get("MCP_ENDPOINT", "http://localhost:5000")
        
        return values
    
    class Config:
        """Pydantic config."""
        
        env_prefix = "DSPY_"
        env_file = ".env"
        case_sensitive = False


@functools.lru_cache()
def get_dspy_settings() -> DSPySettings:
    """
    Get DSPy settings singleton.
    
    Returns:
        DSPySettings: DSPy settings
    """
    settings = DSPySettings()
    
    # Set up logging based on settings
    log_level = getattr(logging, settings.log_level)
    
    logger.setLevel(log_level)
    if settings.log_file:
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return settings


# Export
__all__ = [
    "LLMProvider",
    "CacheBackend",
    "DSPySettings",
    "get_dspy_settings",
]
