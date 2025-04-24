"""
Configuration settings for the Conexus LLM Gateway.

This module provides a centralized configuration system that loads
settings from environment variables and configuration files.
"""

import json
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
from pydantic import BaseModel, BaseSettings, Field, validator
from functools import lru_cache

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONFIG_PATH = "config.json"


class EnvironmentType(str, Enum):
    """Environment type for the application."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ModeSettings(BaseModel):
    """Mode settings for the application."""
    debug_mode: bool = Field(False, description="Enable debug mode")
    environment: EnvironmentType = Field(
        EnvironmentType.DEVELOPMENT,
        description="Environment type"
    )


class DatabaseSettings(BaseModel):
    """Database settings for the application."""
    enabled: bool = Field(True, description="Enable database")
    url: str = Field(
        "sqlite:///asf_llm_gateway.db",
        description="Database connection URL"
    )
    pool_size: int = Field(5, description="Connection pool size")
    max_overflow: int = Field(10, description="Maximum connection overflow")
    pool_recycle: int = Field(3600, description="Connection pool recycle time in seconds")


class ApiSettings(BaseModel):
    """API settings for the application."""
    host: str = Field("0.0.0.0", description="API host")
    port: int = Field(8000, description="API port")
    base_path: str = Field("/api", description="Base API path")
    cors_origins: List[str] = Field(
        ["*"],
        description="CORS allowed origins"
    )
    cors_allow_credentials: bool = Field(
        True,
        description="CORS allow credentials"
    )


class LoggingSettings(BaseModel):
    """Logging settings for the application."""
    level: str = Field("INFO", description="Logging level")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file_path: Optional[str] = Field(None, description="Log file path")


class MetricsSettings(BaseModel):
    """Metrics settings for the application."""
    enabled: bool = Field(True, description="Enable metrics collection")
    prometheus_enabled: bool = Field(
        False,
        description="Enable Prometheus metrics server"
    )
    prometheus_port: int = Field(
        8001,
        description="Port for Prometheus metrics server"
    )


class SecuritySettings(BaseModel):
    """Security settings for the application."""
    auth_enabled: bool = Field(False, description="Enable authentication")
    api_key_header: str = Field(
        "X-API-Key",
        description="Header for API key"
    )
    api_keys: List[str] = Field([], description="List of valid API keys")
    jwt_secret: str = Field("", description="JWT secret for token signing")
    jwt_algorithm: str = Field("HS256", description="JWT algorithm")
    jwt_expiration_seconds: int = Field(
        3600,
        description="JWT token expiration in seconds"
    )

    @validator('api_keys', pre=True)
    def parse_api_keys(cls, v):
        """Parse API keys from comma-separated string or environment variable."""
        if isinstance(v, str):
            return [key.strip() for key in v.split(',') if key.strip()]
        return v


class ProviderSettings(BaseModel):
    """Settings for LLM providers."""
    enabled_providers: List[str] = Field(
        ["openai", "anthropic"], 
        description="List of enabled LLM providers"
    )
    rate_limit_enabled: bool = Field(
        True,
        description="Enable rate limiting for provider APIs"
    )
    cache_enabled: bool = Field(
        True,
        description="Enable caching of provider responses"
    )
    cache_ttl_seconds: int = Field(
        3600,
        description="Time-to-live for cached responses in seconds"
    )

    # Provider-specific settings
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    
    # AWS Bedrock settings for Claude
    aws_region: Optional[str] = Field(None, description="AWS region for Bedrock API")
    aws_access_key_id: Optional[str] = Field(None, description="AWS Access Key ID")
    aws_secret_access_key: Optional[str] = Field(None, description="AWS Secret Access Key")
    aws_session_token: Optional[str] = Field(None, description="AWS Session Token (optional)")
    aws_profile: Optional[str] = Field(None, description="AWS profile name to use (optional)")
    aws_bedrock_endpoint_url: Optional[str] = Field(None, description="Custom AWS Bedrock endpoint URL (optional)")


class CacheSettings(BaseModel):
    """Cache settings for the application."""
    enabled: bool = Field(True, description="Enable caching")
    backend: str = Field("memory", description="Cache backend (memory, redis)")
    redis_url: Optional[str] = Field(None, description="Redis URL for cache")
    ttl_seconds: int = Field(3600, description="Default cache TTL in seconds")


class Settings(BaseSettings):
    """Main settings for the Conexus LLM Gateway."""
    app_name: str = Field("Conexus LLM Gateway", description="Application name")
    version: str = Field("0.1.0", description="Application version")
    
    # Component settings
    mode: ModeSettings = Field(default_factory=ModeSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)

    class Config:
        """Pydantic config for Settings."""
        env_prefix = "CONEXUS_LLM_"
        env_nested_delimiter = "__"
        case_sensitive = False
        
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            """Customize settings sources to include config file."""
            return (
                init_settings,
                env_settings,
                config_file_settings,
                file_secret_settings,
            )


def config_file_settings(settings: BaseSettings) -> Dict[str, Any]:
    """Load settings from config file."""
    config_path = os.environ.get("CONEXUS_LLM_CONFIG_FILE", DEFAULT_CONFIG_PATH)
    
    if not os.path.exists(config_path):
        logger.info(f"Config file not found at {config_path}, using defaults and env vars")
        return {}
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load config file: {e}")
        return {}


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings.
    
    This function is cached to avoid loading settings multiple times.
    """
    settings = Settings()
    
    # Configure logging based on settings
    logging.basicConfig(
        level=getattr(logging, settings.logging.level.upper()),
        format=settings.logging.format,
    )
    
    if settings.logging.file_path:
        file_handler = logging.FileHandler(settings.logging.file_path)
        file_handler.setFormatter(logging.Formatter(settings.logging.format))
        logging.getLogger().addHandler(file_handler)
    
    return settings