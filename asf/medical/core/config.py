"""
Configuration module for the Medical Research Synthesizer.

This module provides configuration settings and utilities for the application.

Classes:
    Settings: Defines application-wide configuration settings.

Functions:
    load_settings: Load configuration settings from environment variables or defaults.
"""
import os
import secrets
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

from dotenv import load_dotenv
from pydantic import Field, SecretStr, EmailStr, validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

env_file = Path(os.environ.get("ENV_FILE", ".env"))
if env_file.exists():
    logger.info(f"Loading environment variables from {env_file}")
    load_dotenv(env_file)

class Settings(BaseSettings):
    """
    Application configuration settings.

    This class defines configuration settings for the application, including
    database connections, API keys, and other environment-specific settings.

    Attributes:
        ENVIRONMENT (str): Environment name (development, staging, production).
        API_V1_STR (str): API version string.
        PROJECT_NAME (str): Name of the project.
        DEBUG (bool): Debug mode flag.
        CORS_ORIGINS (List[str]): List of allowed CORS origins.
        SECRET_KEY (SecretStr): Secret key for the application.
        ACCESS_TOKEN_EXPIRE_MINUTES (int): Access token expiration time in minutes.
        DATABASE_URL (str): Database connection URL.
        REDIS_URL (Optional[str]): Redis connection URL.
        CACHE_TTL (int): Cache time-to-live in seconds.
        NCBI_EMAIL (EmailStr): Email for NCBI API.
        NCBI_API_KEY (Optional[SecretStr]): API key for NCBI.
        IMPACT_FACTOR_SOURCE (str): Source file for journal impact factors.
        KB_DIR (str): Directory for knowledge bases.
        USE_GPU (bool): Whether to use GPU for computations.
        BIOMEDLM_MODEL (str): Name of the BioMedLM model to use.
        RAY_ADDRESS (Optional[str]): Address of the Ray cluster.
        RAY_NUM_CPUS (Optional[int]): Number of CPUs to allocate for Ray.
        RAY_NUM_GPUS (Optional[int]): Number of GPUs to allocate for Ray.
        LOG_LEVEL (str): Logging level for the application.
        GRAPH_DB_TYPE (str): Type of graph database (e.g., "memgraph", "neo4j").
        MEMGRAPH_HOST (str): Hostname for the Memgraph database.
        MEMGRAPH_PORT (int): Port for the Memgraph database.
        NEO4J_URI (str): URI for the Neo4j database.
        NEO4J_USER (str): Username for the Neo4j database.
        NEO4J_PASSWORD (SecretStr): Password for the Neo4j database.
        RABBITMQ_HOST (str): Hostname for the RabbitMQ server.
        RABBITMQ_PORT (int): Port for the RabbitMQ server.
        RABBITMQ_USERNAME (str): Username for the RabbitMQ server.
        RABBITMQ_PASSWORD (str): Password for the RabbitMQ server.
        RABBITMQ_VHOST (str): Virtual host for the RabbitMQ server.
        RABBITMQ_ENABLED (bool): Whether RabbitMQ messaging is enabled.
    """
    ENVIRONMENT: str = "development"
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
        """
        Assemble CORS origins from a string or list.

        Args:
            v (Union[str, List[str]]): CORS origins as a string or list.

        Returns:
            Union[List[str], str]: Parsed CORS origins.
        """
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v

def load_settings() -> Settings:
    """
    Load configuration settings from environment variables or defaults.

    Returns:
        Settings: The loaded configuration settings.
    """
    settings = Settings()

    os.makedirs(settings.KB_DIR, exist_ok=True)

    return settings

settings = load_settings()

__all__ = ["settings", "load_settings"]
