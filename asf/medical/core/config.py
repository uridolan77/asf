Configuration module for the Medical Research Synthesizer.

This module provides a centralized configuration using Pydantic.
It supports loading configuration from environment variables, .env files,
and external configuration sources.

import os
import secrets
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

from dotenv import load_dotenv
from pydantic import BaseSettings, Field, SecretStr, EmailStr, validator

logger = logging.getLogger(__name__)

env_file = Path(os.environ.get("ENV_FILE", ".env"))
if env_file.exists():
    logger.info(f"Loading environment variables from {env_file}")
    load_dotenv(env_file)

class Settings(BaseSettings):
    Settings for the Medical Research Synthesizer.
    
    This class uses Pydantic to load and validate configuration from environment variables.

    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "Medical Research Synthesizer"
    DEBUG: bool = False

    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    SECRET_KEY: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_urlsafe(32)))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    DATABASE_URL: str = "sqlite:///./medical_research_synthesizer.db"

    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 3600  # 1 hour

    NCBI_EMAIL: EmailStr = "your.email@example.com"
    NCBI_API_KEY: Optional[SecretStr] = None

    IMPACT_FACTOR_SOURCE: str = "journal_impact_factors.csv"
    KB_DIR: str = "knowledge_bases"

    USE_GPU: bool = True
    BIOMEDLM_MODEL: str = "microsoft/BioMedLM"

    RAY_ADDRESS: Optional[str] = None  # None means local mode
    RAY_NUM_CPUS: Optional[int] = None  # None means use all available
    RAY_NUM_GPUS: Optional[int] = None  # None means use all available

    LOG_LEVEL: str = "INFO"

    GRAPH_DB_TYPE: str = "memgraph"  # memgraph or neo4j
    MEMGRAPH_HOST: str = "localhost"
    MEMGRAPH_PORT: int = 7687
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: SecretStr = SecretStr("neo4j")

    # RabbitMQ settings
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USERNAME: str = "guest"
    RABBITMQ_PASSWORD: str = "guest"
    RABBITMQ_VHOST: str = "/"
    RABBITMQ_ENABLED: bool = False  # Set to True to enable RabbitMQ messaging

    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        Parse CORS origins from string or list.
        
        Args:
            cls: Description of cls
            v: Description of v
        
        
        Returns:
            Description of return value
    settings = Settings()

    os.makedirs(settings.KB_DIR, exist_ok=True)

    return settings

settings = get_settings()

__all__ = ["settings", "get_settings"]
