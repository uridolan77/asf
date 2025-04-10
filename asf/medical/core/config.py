"""
Configuration module for the Medical Research Synthesizer.

This module provides a centralized configuration using Pydantic.
It supports loading configuration from environment variables, .env files,
and external configuration sources.
"""

import os
import secrets
import logging
from typing import Optional, Dict, Any, List, Union, Callable
from functools import lru_cache
from pathlib import Path

from pydantic import BaseSettings, EmailStr, SecretStr, validator, Field, AnyHttpUrl, PostgresDsn
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load .env file if it exists
env_file = Path(os.environ.get("ENV_FILE", ".env"))
if env_file.exists():
    logger.info(f"Loading environment variables from {env_file}")
    load_dotenv(env_file)

class Settings(BaseSettings):
    """
    Settings for the Medical Research Synthesizer.

    This class uses Pydantic to load and validate configuration from environment variables.
    """

    # API settings
    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "Medical Research Synthesizer"
    DEBUG: bool = False

    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Security settings
    SECRET_KEY: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_urlsafe(32)))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # Database settings
    DATABASE_URL: str = "sqlite:///./medical_research_synthesizer.db"

    # Cache settings
    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 3600  # 1 hour

    # External API settings
    NCBI_EMAIL: EmailStr = "your.email@example.com"
    NCBI_API_KEY: Optional[SecretStr] = None

    # File paths
    IMPACT_FACTOR_SOURCE: str = "journal_impact_factors.csv"
    KB_DIR: str = "knowledge_bases"

    # Model settings
    USE_GPU: bool = True
    BIOMEDLM_MODEL: str = "microsoft/BioMedLM"

    # Ray settings
    RAY_ADDRESS: Optional[str] = None  # None means local mode
    RAY_NUM_CPUS: Optional[int] = None  # None means use all available
    RAY_NUM_GPUS: Optional[int] = None  # None means use all available

    # Logging settings
    LOG_LEVEL: str = "INFO"

    # Graph database settings
    GRAPH_DB_TYPE: str = "memgraph"  # memgraph or neo4j
    MEMGRAPH_HOST: str = "localhost"
    MEMGRAPH_PORT: int = 7687
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: SecretStr = SecretStr("neo4j")

    # Validators
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    class Config:
        """Pydantic config"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create settings instance with caching
@lru_cache()
def get_settings() -> Settings:
    """Get the settings instance."""
    settings = Settings()

    # Create required directories
    os.makedirs(settings.KB_DIR, exist_ok=True)

    return settings

# Create a singleton instance for backward compatibility
settings = get_settings()

# Export settings
__all__ = ["settings", "get_settings"]
